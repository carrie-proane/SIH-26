from __future__ import annotations

import csv
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .colmap import ColmapRunner, ExternalToolError, ReconstructionResult, write_matcher_benchmark
from .geo import geodetic_to_enu, robust_similarity, transform_ply
from .models import MatcherMetrics, RunRecord, RunStatus, StageEvent
from .report import build_quality_report, write_quality_report
from .storage import ProjectStore, atomic_json


class PipelineError(RuntimeError):
    pass


def _asset_path(store: ProjectStore, record: RunRecord, role: str) -> Path:
    project = store.get_project(record.project_id)
    matches = [asset for asset in project.assets if asset.role == role]
    if len(matches) != 1:
        raise PipelineError(f"Project must declare exactly one {role} asset")
    return store.project_dir(record.project_id) / matches[0].relative_path


def _synthetic_ply(path: Path) -> None:
    points = [
        (-2, -1, 0, 32, 191, 107), (-1, -1, 0.1, 32, 191, 107),
        (0, -1, 0.2, 245, 158, 11), (1, -1, 0.1, 245, 158, 11),
        (2, -1, 0, 239, 68, 68), (-2, 1, 0, 32, 191, 107),
        (-1, 1, 0.2, 32, 191, 107), (0, 1, 0.4, 245, 158, 11),
        (1, 1, 0.2, 239, 68, 68), (2, 1, 0, 239, 68, 68),
    ]
    header = (
        "ply\nformat ascii 1.0\ncomment SYNTHETIC_DEMO - not reconstruction evidence\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_text(header + "\n".join(" ".join(map(str, row)) for row in points) + "\n", encoding="utf-8")


class PipelineRunner:
    def __init__(self, store: ProjectStore, max_workers: int = 2) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sih-run")

    def submit(self, run_id: str) -> None:
        self.executor.submit(self.run, run_id)

    def _transition(
        self, record: RunRecord, stage: RunStatus, progress: int, message: str, event_status: str = "STARTED"
    ) -> None:
        record.stage = stage
        record.status = stage
        record.progress = progress
        record.events.append(
            StageEvent(stage=stage, status=event_status, progress=progress, message=message)  # type: ignore[arg-type]
        )
        self.store.save_run(record)

    def _probe(self, record: RunRecord) -> tuple[Path, list[dict[str, str]]]:
        video = _asset_path(self.store, record, "video")
        project = self.store.get_project(record.project_id)
        warnings: list[dict[str, str]] = []
        ffprobe = shutil.which("ffprobe")
        if record.synthetic_fixture:
            probe = {
                "format": {"filename": video.name, "format_name": "synthetic_fixture"},
                "streams": [],
            }
            warnings.append(
                {
                    "code": "FFPROBE_SKIPPED",
                    "message": "Synthetic demo did not inspect a real codec.",
                }
            )
        elif ffprobe:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise PipelineError(
                    "ffprobe could not read the video. Verify that the file is a supported, non-corrupt "
                    "MP4/MOV and inspect logs before retrying."
                )
            probe = json.loads(result.stdout)
        else:
            raise PipelineError(
                "ffprobe is not installed or not on PATH. Install FFmpeg and verify `ffprobe -version`; "
                "ingest will not continue with unknown frame timing."
            )
        payload = {
            "project_id": record.project_id,
            "run_id": record.run_id,
            "stage": "INGESTING",
            "status": "COMPLETED",
            "created_at": record.created_at,
            "config_version": record.config_version,
            "video_probe": probe,
            "input_assets": [asset.model_dump(mode="json") for asset in project.assets],
            "warnings": warnings,
        }
        path = self.store.run_dir(record.project_id, record.run_id) / "ingest_report.json"
        atomic_json(path, payload)
        return path, warnings

    def _preprocess_contract(self, record: RunRecord) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if record.synthetic_fixture:
            keyframes = [
                {"frame_index": i, "timestamp_s": i * 0.5, "selected": True, "source": "SYNTHETIC_DEMO"}
                for i in range(10)
            ]
            keyframes_path = run_dir / "keyframes.json"
            atomic_json(keyframes_path, {"synthetic_fixture": True, "frames": keyframes})
            scores_path = run_dir / "frame_scores.csv"
            with scores_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_index", "timestamp_s", "blur_score", "exposure_score", "selected"])
                for item in keyframes:
                    writer.writerow([item["frame_index"], item["timestamp_s"], 0.8, 0.9, True])
            telemetry_path = run_dir / "normalized_telemetry.csv"
            with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "timestamp_s",
                        "lat",
                        "lon",
                        "alt_m",
                        "alt_source",
                        "fix_quality",
                        "source_row",
                    ]
                )
                for i in range(10):
                    writer.writerow(
                        [
                            i * 0.5,
                            28.6139,
                            77.209 + i * 0.000005,
                            42,
                            "synthetic",
                            "ok",
                            i,
                        ]
                    )
            telemetry_meta_path = run_dir / "normalized_telemetry.meta.json"
            atomic_json(
                telemetry_meta_path,
                {
                    "schema_version": "1.0",
                    "source_file": "synthetic_telemetry.csv",
                    "source_format": "synthetic",
                    "source_dialect": "deterministic_smoke_fixture",
                    "parser_version": "0.1.0",
                    "row_count": 10,
                    "duration_s": 4.5,
                    "sample_rate_hz_estimated": 2.0,
                    "time_origin": "synthetic_video_start",
                    "coordinate_frame": "WGS84",
                    "altitude_reference": "relative_to_launch",
                    "warnings": [
                        {
                            "code": "SYNTHETIC_TELEMETRY",
                            "count": 10,
                            "detail": "Generated data; never present as a real flight.",
                        }
                    ],
                    "field_coverage": {"lat": 1.0, "lon": 1.0, "alt_m": 1.0},
                },
            )
            return [keyframes_path, scores_path, telemetry_path, telemetry_meta_path]
        if not record.config.preprocessing_run:
            raise PipelineError(
                "No preprocessing_run was supplied. Complete frame extraction/selection first and "
                "pass the path containing keyframes.json, frame_scores.csv, and frames/."
            )
        source = Path(record.config.preprocessing_run).resolve()
        required = [
            source / "keyframes.json",
            source / "frame_scores.csv",
            source / "normalized_telemetry.csv",
            source / "normalized_telemetry.meta.json",
            source / "frames",
        ]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise PipelineError(f"Preprocessing handoff is incomplete; missing: {', '.join(missing)}")
        shutil.copyfile(required[0], run_dir / "keyframes.json")
        shutil.copyfile(required[1], run_dir / "frame_scores.csv")
        shutil.copyfile(required[2], run_dir / "normalized_telemetry.csv")
        shutil.copyfile(required[3], run_dir / "normalized_telemetry.meta.json")
        for image in required[4].iterdir():
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copyfile(image, run_dir / "frames" / image.name)
        return [
            run_dir / "keyframes.json",
            run_dir / "frame_scores.csv",
            run_dir / "normalized_telemetry.csv",
            run_dir / "normalized_telemetry.meta.json",
        ]

    def _merge_telemetry_metadata(
        self, record: RunRecord, warnings: list[dict[str, str]]
    ) -> Path:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        meta_path = run_dir / "normalized_telemetry.meta.json"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(
                "normalized_telemetry.meta.json is missing or invalid; rerun the telemetry parser "
                "and preserve its schema/version/warnings sidecar."
            ) from exc
        if metadata.get("schema_version") != "1.0":
            raise PipelineError("Only normalized telemetry schema_version 1.0 is supported.")
        for warning in metadata.get("warnings", []):
            if not isinstance(warning, dict):
                continue
            warnings.append(
                {
                    "code": str(warning.get("code", "TELEMETRY_WARNING")),
                    "message": str(warning.get("detail", warning.get("code", "Telemetry warning"))),
                }
            )
        ingest_path = run_dir / "ingest_report.json"
        ingest = json.loads(ingest_path.read_text(encoding="utf-8"))
        ingest["telemetry_normalization"] = metadata
        ingest["warnings"] = warnings
        atomic_json(ingest_path, ingest)
        return ingest_path

    def _align_to_local_metric(self, record: RunRecord) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if record.synthetic_fixture:
            transform_path = run_dir / "local_transform.json"
            atomic_json(
                transform_path,
                {
                    "synthetic_fixture": True,
                    "coordinate_frame": "LOCAL_ENU_METRES",
                    "scale": 1.0,
                    "rotation": np.eye(3).tolist(),
                    "translation_m": [0.0, 0.0, 0.0],
                    "inlier_count": 10,
                    "residuals_m": [0.0] * 10,
                },
            )
            local_ply = run_dir / "sparse" / "sparse_local.ply"
            shutil.copyfile(run_dir / "sparse" / "sparse.ply", local_ply)
            return [transform_path, local_ply]

        keyframe_payload = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
        keyframes = (
            keyframe_payload.get("frames", [])
            if isinstance(keyframe_payload, dict)
            else keyframe_payload
        )
        if not isinstance(keyframes, list):
            raise PipelineError("keyframes.json must be a list or an object containing a frames list.")
        timestamps: dict[str, float] = {}
        for item in keyframes:
            if not isinstance(item, dict) or not item.get("selected", True):
                continue
            name = item.get("image_name") or item.get("filename")
            if name is not None and item.get("timestamp_s") is not None:
                timestamps[Path(str(name)).name] = float(item["timestamp_s"])
        if len(timestamps) < 3:
            raise PipelineError(
                "At least three selected keyframes must declare image_name/filename and timestamp_s "
                "for metric alignment."
            )

        telemetry: list[dict[str, str]] = []
        with (run_dir / "normalized_telemetry.csv").open(newline="", encoding="utf-8") as stream:
            telemetry = list(csv.DictReader(stream))
        required = {
            "timestamp_s",
            "lat",
            "lon",
            "alt_m",
            "alt_source",
            "fix_quality",
            "source_row",
        }
        if not telemetry or not required.issubset(telemetry[0]):
            raise PipelineError(
                "normalized_telemetry.csv does not match Yosha's schema v1 column contract."
            )
        telemetry = [
            row
            for row in telemetry
            if row["lat"].strip()
            and row["lon"].strip()
            and row["alt_m"].strip()
            and row["fix_quality"] != "missing"
        ]
        if len(telemetry) < 3:
            raise PipelineError(
                "Fewer than three telemetry rows contain a usable coordinate fix; metric alignment "
                "cannot be solved."
            )
        telemetry.sort(key=lambda row: float(row["timestamp_s"]))
        times = np.array([float(row["timestamp_s"]) for row in telemetry])
        if np.any(np.diff(times) <= 0):
            raise PipelineError("Normalized telemetry timestamps must be strictly increasing.")
        origin = record.config.local_origin or (
            float(telemetry[0]["lat"]),
            float(telemetry[0]["lon"]),
            float(telemetry[0]["alt_m"]),
        )
        enu = np.array(
            [
                geodetic_to_enu(float(row["lat"]), float(row["lon"]), float(row["alt_m"]), origin)
                for row in telemetry
            ]
        )

        pose_rows: list[dict[str, str]]
        pose_path = run_dir / "camera_poses.csv"
        with pose_path.open(newline="", encoding="utf-8") as stream:
            pose_rows = list(csv.DictReader(stream))
        sfm_points: list[list[float]] = []
        metric_points: list[np.ndarray] = []
        matched_rows: list[dict[str, str]] = []
        for row in pose_rows:
            timestamp = timestamps.get(Path(row["image_name"]).name)
            if timestamp is None or timestamp < times[0] or timestamp > times[-1]:
                continue
            right = int(np.searchsorted(times, timestamp, side="right"))
            if right == 0:
                position = enu[0]
            elif right == len(times):
                position = enu[-1]
            else:
                left = right - 1
                weight = (timestamp - times[left]) / (times[right] - times[left])
                position = enu[left] * (1 - weight) + enu[right] * weight
            sfm_points.append([float(row["sfm_x"]), float(row["sfm_y"]), float(row["sfm_z"])])
            metric_points.append(position)
            matched_rows.append(row | {"timestamp_s": str(timestamp)})
        if len(sfm_points) < 3:
            raise PipelineError(
                "Fewer than three registered cameras could be joined to telemetry timestamps; "
                "verify keyframe filenames and synchronization."
            )
        transform = robust_similarity(np.array(sfm_points), np.array(metric_points))
        aligned = transform.apply(np.array(sfm_points))
        with pose_path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(matched_rows[0]) + ["x_m", "y_m", "z_m", "alignment_residual_m", "alignment_inlier"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(matched_rows):
                row.update(
                    {
                        "x_m": aligned[index, 0],
                        "y_m": aligned[index, 1],
                        "z_m": aligned[index, 2],
                        "alignment_residual_m": transform.residuals_m[index],
                        "alignment_inlier": bool(transform.inliers[index]),
                    }
                )
                writer.writerow(row)
        transform_path = run_dir / "local_transform.json"
        atomic_json(
            transform_path,
            transform.as_dict()
            | {
                "coordinate_frame": "LOCAL_ENU_METRES",
                "origin_wgs84": {"lat": origin[0], "lon": origin[1], "alt_m": origin[2]},
                "outlier_image_names": [
                    matched_rows[index]["image_name"]
                    for index, keep in enumerate(transform.inliers)
                    if not keep
                ],
            },
        )
        local_ply = run_dir / "sparse" / "sparse_local.ply"
        transform_ply(run_dir / "sparse" / "sparse.ply", local_ply, transform)
        return [pose_path, transform_path, local_ply]

    def _reconstruct(self, record: RunRecord) -> ReconstructionResult:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if record.synthetic_fixture:
            ply = run_dir / "sparse" / "sparse.ply"
            _synthetic_ply(ply)
            poses = run_dir / "camera_poses.csv"
            with poses.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_index", "timestamp_s", "x_m", "y_m", "z_m", "source"])
                for i in range(10):
                    writer.writerow([i, i * 0.5, i * 0.5, 0, 2, "SYNTHETIC_DEMO"])
            metrics = MatcherMetrics(
                matcher="SIFT_SYNTHETIC_PLACEHOLDER",
                eligible_frames=10,
                registered_frames=9,
                median_reprojection_error_px=0.9,
                p95_reprojection_error_px=1.4,
                runtime_s=0.01,
            )
            return ReconstructionResult(metrics, [ply, poses], [])
        return ColmapRunner().run(run_dir / "frames", run_dir, record.config)

    def run(self, run_id: str) -> RunRecord:
        record = self.store.get_run(run_id)
        warnings: list[dict[str, str]] = []
        try:
            self._transition(record, RunStatus.INGESTING, 10, "Inspecting immutable input assets.")
            ingest, warnings = self._probe(record)
            self.store.register_artifacts(record, [ingest])
            self._transition(record, RunStatus.PREPROCESSING, 30, "Validating selected-frame handoff.")
            handoff = self._preprocess_contract(record)
            self.store.register_artifacts(record, handoff)
            updated_ingest = self._merge_telemetry_metadata(record, warnings)
            self.store.register_artifacts(record, [updated_ingest])
            self._transition(record, RunStatus.RECONSTRUCTING, 55, "Running sparse reconstruction.")
            result = self._reconstruct(record)
            self.store.register_artifacts(record, result.artifacts)
            alignment = self._align_to_local_metric(record)
            self.store.register_artifacts(record, alignment)
            metrics_path = self.store.run_dir(record.project_id, record.run_id) / "sparse_metrics.json"
            atomic_json(
                metrics_path,
                result.metrics.model_dump(mode="json")
                | {"registration_rate": result.metrics.registration_rate, "synthetic_fixture": record.synthetic_fixture},
            )
            benchmark_path = self.store.run_dir(record.project_id, record.run_id) / "matcher_benchmark.json"
            write_matcher_benchmark(benchmark_path, result.metrics, None)
            self.store.register_artifacts(record, [metrics_path, benchmark_path])
            self._transition(record, RunStatus.REPORTING, 85, "Generating trust and known-distance report.")
            quality_path = self.store.run_dir(record.project_id, record.run_id) / "quality_report.json"
            alignment_report = json.loads(
                (
                    self.store.run_dir(record.project_id, record.run_id)
                    / "local_transform.json"
                ).read_text(encoding="utf-8")
            )
            report = build_quality_report(
                record,
                result.metrics,
                warnings,
                synthetic=record.synthetic_fixture,
                alignment=alignment_report,
            )
            write_quality_report(quality_path, report)
            self.store.register_artifacts(record, [quality_path])
            self._transition(record, RunStatus.COMPLETED, 100, "Run completed.", "COMPLETED")
            return record
        except (PipelineError, ExternalToolError, ValueError, OSError, json.JSONDecodeError) as exc:
            record.failure_reason = str(exc)
            record.stage = RunStatus.FAILED
            record.status = RunStatus.FAILED
            record.events.append(
                StageEvent(stage=RunStatus.FAILED, status="FAILED", progress=record.progress, message=str(exc))
            )
            self.store.save_run(record)
            return record
