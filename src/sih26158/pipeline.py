from __future__ import annotations

import csv
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .infrastructure.storage import ProjectStore, atomic_json
from .models import (
    MatcherMetrics,
    OffsetSource,
    ProvenanceOrigin,
    RunRecord,
    RunStatus,
    StageEvent,
)
from .preprocessing.frames.contact_sheet import create_contact_sheet
from .preprocessing.frames.extractor import extract_frames
from .preprocessing.frames.selector import (
    FRAME_SCORE_COLUMNS,
    FrameQualityThresholds,
    SelectionWeights,
    select_keyframes,
)
from .preprocessing.scene_policy import analyze_scene
from .preprocessing.segmentation import run_optional_segmentation
from .preprocessing.telemetry.csv_parser import parse_csv
from .preprocessing.telemetry.models import sha256_file as telemetry_sha256_file
from .preprocessing.telemetry.models import write_csv as write_telemetry_csv
from .preprocessing.telemetry.srt_parser import parse_srt
from .reconstruction.colmap import (
    ColmapRunner,
    ExternalToolError,
    ReconstructionResult,
    write_matcher_benchmark,
)
from .reconstruction.confidence import validate_point_confidence_for_ply
from .reconstruction.dense import (
    DenseContext,
    UnavailableProvider,
    run_dense_stage,
    select_dense_provider,
)
from .reconstruction.geo import SimilarityTransform, geodetic_to_enu, transform_ply
from .reconstruction.surface_completion import (
    ExternalSurfaceCompletionProvider,
    SurfaceCompletionContext,
    UnavailableSurfaceCompletionProvider,
    run_surface_completion_stage,
)
from .reconstruction.sync import calibrate_telemetry_offset
from .reporting.report import build_quality_report, write_quality_report


class PipelineError(RuntimeError):
    pass


def _is_synthetic_demo(record: RunRecord) -> bool:
    return record.config.execution_mode == "SYNTHETIC_DEMO"


def _asset_path(store: ProjectStore, record: RunRecord, role: str) -> Path:
    project = store.get_project(record.project_id)
    matches = [asset for asset in project.assets if asset.role == role]
    if len(matches) != 1:
        raise PipelineError(f"Project must declare exactly one {role} asset")
    return store.project_dir(record.project_id) / matches[0].relative_path


def _synthetic_ply(path: Path) -> None:
    points = [
        (-2, -1, 0, 32, 191, 107),
        (-1, -1, 0.1, 32, 191, 107),
        (0, -1, 0.2, 245, 158, 11),
        (1, -1, 0.1, 245, 158, 11),
        (2, -1, 0, 239, 68, 68),
        (-2, 1, 0, 32, 191, 107),
        (-1, 1, 0.2, 32, 191, 107),
        (0, 1, 0.4, 245, 158, 11),
        (1, 1, 0.2, 239, 68, 68),
        (2, 1, 0, 239, 68, 68),
    ]
    header = (
        "ply\nformat ascii 1.0\ncomment SYNTHETIC_DEMO - not reconstruction evidence\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_text(
        header + "\n".join(" ".join(map(str, row)) for row in points) + "\n", encoding="utf-8"
    )


class PipelineRunner:
    def __init__(self, store: ProjectStore, max_workers: int = 2) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sih-run")

    def submit(self, run_id: str) -> None:
        self.executor.submit(self.run, run_id)

    def _transition(
        self,
        record: RunRecord,
        stage: RunStatus,
        progress: int,
        message: str,
        event_status: str = "STARTED",
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
        if _is_synthetic_demo(record):
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
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ],
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
            "source_provenance": record.source_provenance,
            "video_origin": record.video_origin,
            "telemetry_origin": record.telemetry_origin,
            "genuine_real_evidence": record.source_provenance == ProvenanceOrigin.REAL,
            "warnings": warnings,
        }
        path = self.store.run_dir(record.project_id, record.run_id) / "ingest_report.json"
        atomic_json(path, payload)
        return path, warnings

    def _preprocess_contract(self, record: RunRecord) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
            keyframes = [
                {
                    "frame_index": i,
                    "timestamp_s": i * 0.5,
                    "selected": True,
                    "source": "SYNTHETIC_DEMO",
                }
                for i in range(10)
            ]
            keyframes_path = run_dir / "keyframes.json"
            atomic_json(keyframes_path, {"synthetic_fixture": True, "frames": keyframes})
            scores_path = run_dir / "frame_scores.csv"
            with scores_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["frame_index", "timestamp_s", "blur_score", "exposure_score", "selected"]
                )
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
        handoff_problem: str | None = None
        if record.config.preprocessing_run:
            source = Path(record.config.preprocessing_run).resolve()
            required = [
                source / "keyframes.json",
                source / "frame_scores.csv",
                source / "normalized_telemetry.csv",
                source / "normalized_telemetry.meta.json",
                source / "frames",
            ]
            missing = [path.name for path in required if not path.exists()]
            if not missing:
                try:
                    payload = json.loads(required[0].read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    missing.append("valid keyframes.json")
                    payload = {}
                frame_rows = payload.get("frames", []) if isinstance(payload, dict) else []
                if not isinstance(frame_rows, list):
                    missing.append("keyframes.json frames list")
                    frame_rows = []
                selected_frame_rows = [
                    item
                    for item in frame_rows
                    if isinstance(item, dict) and item.get("selected", True)
                ]
                if len(selected_frame_rows) < 3:
                    missing.append("at least three selected keyframes")
                copied_frames: list[Path] = []
                for item in frame_rows:
                    if not isinstance(item, dict) or not item.get("selected", True):
                        continue
                    raw_name = item.get("image_name") or item.get("filename")
                    safe_name = Path(str(raw_name or "")).name
                    source_image = required[4] / safe_name
                    if not safe_name or not source_image.is_file():
                        missing.append(f"frames/{safe_name or '<missing image_name>'}")
                        continue
                    destination = run_dir / "frames" / safe_name
                    shutil.copyfile(source_image, destination)
                    copied_frames.append(destination)
                    item["image_name"] = safe_name
                    item["image_url"] = f"/api/runs/{record.run_id}/artifacts/frames/{safe_name}"
                    item.pop("path", None)
                    item.pop("frame_path", None)
                if missing:
                    handoff_problem = (
                        f"Configured handoff {source} was incomplete; missing: "
                        f"{', '.join(sorted(set(missing)))}. Generated a dataset-specific handoff "
                        "from this run's immutable inputs instead."
                    )
                    for copied in copied_frames:
                        copied.unlink(missing_ok=True)
                    return self._preprocess_uploaded_inputs(record, handoff_problem)
                atomic_json(run_dir / "keyframes.json", payload)
                shutil.copyfile(required[1], run_dir / "frame_scores.csv")
                shutil.copyfile(required[2], run_dir / "normalized_telemetry.csv")
                shutil.copyfile(required[3], run_dir / "normalized_telemetry.meta.json")
                artifacts = [
                    run_dir / "keyframes.json",
                    run_dir / "frame_scores.csv",
                    run_dir / "normalized_telemetry.csv",
                    run_dir / "normalized_telemetry.meta.json",
                    *copied_frames,
                ]
                for optional_name in ("frame_index.csv", "contact_sheet.png"):
                    optional = source / optional_name
                    if optional.is_file():
                        shutil.copyfile(optional, run_dir / optional_name)
                        artifacts.append(run_dir / optional_name)
                return artifacts
            handoff_problem = (
                f"Configured handoff {source} was incomplete; missing: {', '.join(missing)}. "
                "Generated a dataset-specific handoff from this run's immutable inputs instead."
            )
        return self._preprocess_uploaded_inputs(record, handoff_problem)

    def _preprocess_uploaded_inputs(
        self, record: RunRecord, handoff_problem: str | None = None
    ) -> list[Path]:
        """Create a scored, dataset-specific handoff from immutable uploaded inputs."""
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        video = _asset_path(self.store, record, "video")
        telemetry = _asset_path(self.store, record, "telemetry")
        preprocessing_dir = run_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        try:
            extraction = extract_frames(
                video,
                run_dir,
                frames_subdir="preprocessing/candidates",
            )
        except (ValueError, OSError) as exc:
            raise PipelineError(f"Automatic frame extraction failed: {exc}") from exc
        candidates = extraction.frames
        if len(candidates) < 3:
            raise PipelineError(
                "Automatic preprocessing extracted fewer than three frames; provide a longer "
                "video or a complete dataset-specific preprocessing handoff."
            )
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            36,
            f"Decoded {len(candidates)} timestamped candidate frames.",
        )
        target_frames = min(100, len(candidates))
        try:
            rows = select_keyframes(
                candidates,
                run_dir,
                target_frames=target_frames,
                weights=SelectionWeights(),
                force_include=set(record.config.force_include_frame_indices),
                force_exclude=set(record.config.force_exclude_frame_indices),
                quality_thresholds=FrameQualityThresholds(
                    min_laplacian_variance=record.config.frame_min_laplacian_variance,
                    min_exposure_score=record.config.frame_min_exposure_score,
                    relative_sharpness_floor=record.config.frame_relative_sharpness_floor,
                ),
            )
        except ValueError as exc:
            raise PipelineError(f"Automatic frame selection failed: {exc}") from exc
        contact_sheet_path = create_contact_sheet(rows, run_dir / "contact_sheet.png")
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            44,
            f"Scored candidates and selected {sum(bool(row['selected']) for row in rows)} frames.",
        )

        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        selected_paths: list[Path] = []
        keyframes: list[dict[str, object]] = []
        for row in rows:
            candidate = Path(str(row["frame_path"]))
            selected = bool(row["selected"])
            if selected:
                destination = frames_dir / candidate.name
                shutil.copyfile(candidate, destination)
                selected_paths.append(destination)
            safe_row = dict(row)
            safe_row["source_video"] = video.name
            safe_row["frame_path"] = f"preprocessing/candidates/{candidate.name}"
            safe_row["image_name"] = candidate.name
            safe_row["path"] = safe_row["frame_path"]
            safe_row["source"] = "AUTO_SCORED_SELECTION_FROM_UPLOADED_VIDEO"
            if selected:
                safe_row["image_url"] = (
                    f"/api/runs/{record.run_id}/artifacts/frames/{candidate.name}"
                )
            keyframes.append(safe_row)
        keyframes_path = run_dir / "keyframes.json"
        atomic_json(
            keyframes_path,
            {
                "schema_version": "1.0",
                "selection_method": "QUALITY_GATED_BLUR_EXPOSURE_REDUNDANCY",
                "frame_quality_gate": {
                    "minimum_laplacian_variance": (record.config.frame_min_laplacian_variance),
                    "minimum_exposure_score": record.config.frame_min_exposure_score,
                    "relative_sharpness_floor": (record.config.frame_relative_sharpness_floor),
                    "candidate_count": len(keyframes),
                    "eligible_count": sum(bool(item["quality_eligible"]) for item in keyframes),
                    "rejected_count": sum(not bool(item["quality_eligible"]) for item in keyframes),
                    "selected_count": sum(bool(item["selected"]) for item in keyframes),
                },
                "override_actions": {
                    "force_include": record.config.force_include_frame_indices,
                    "force_exclude": record.config.force_exclude_frame_indices,
                },
                "frames": keyframes,
            },
        )
        scores_path = run_dir / "frame_scores.csv"
        with scores_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_SCORE_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {column: item.get(column, "") for column in FRAME_SCORE_COLUMNS}
                for item in keyframes
            )

        suffix = telemetry.suffix.lower()
        parsed = parse_srt(telemetry) if suffix == ".srt" else parse_csv(telemetry)
        usable = [
            record for record in parsed.records if record.has_fix and record.alt_m is not None
        ]
        if len(usable) < 3:
            raise PipelineError(
                "Uploaded telemetry could not produce at least three usable latitude, longitude, "
                f"and altitude samples ({parsed.warnings.summary()})."
            )
        if handoff_problem:
            parsed.warnings.add("HANDOFF_FALLBACK", handoff_problem)
        duration_s = (
            extraction.source_frame_count / extraction.fps
            if extraction.source_frame_count > 0
            else max((frame.timestamp_s for frame in candidates), default=0.0)
        )
        selected_rows = [row for row in rows if row["selected"]]
        quality_eligible_rows = [row for row in rows if row["quality_eligible"]]
        if len(candidates) < 60:
            parsed.warnings.add(
                "TOO_FEW_FRAME_CANDIDATES",
                f"Only {len(candidates)} candidate frames were decoded; 60 or more are preferred.",
            )
        if len(selected_rows) < 60:
            parsed.warnings.add(
                "TOO_FEW_SELECTED_FRAMES",
                f"Only {len(selected_rows)} frames were selected; the target range is 60-120.",
            )
        rejected_count = len(rows) - len(quality_eligible_rows)
        if rejected_count:
            parsed.warnings.add(
                "FRAME_QUALITY_REJECTIONS",
                f"{rejected_count} candidate frame(s) were excluded by absolute/adaptive "
                "sharpness or exposure gates.",
            )
        if duration_s < 20:
            parsed.warnings.add(
                "VERY_SHORT_VIDEO",
                f"Video duration is approximately {duration_s:.2f}s; controlled passes should be 30-60s.",
            )
        averages = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in ("blur_score", "exposure_score", "redundancy_score")
        }
        if averages["blur_score"] < 0.3:
            parsed.warnings.add("EXCESSIVE_BLUR", "Mean normalized sharpness is below 0.30.")
        if averages["exposure_score"] < 0.4:
            parsed.warnings.add("POOR_EXPOSURE", "Mean exposure score is below 0.40.")
        if averages["redundancy_score"] < 0.25:
            parsed.warnings.add("HIGHLY_REDUNDANT_FOOTAGE", "Mean uniqueness score is below 0.25.")
        telemetry_duration = parsed.duration_s
        if telemetry_duration and abs(telemetry_duration - duration_s) > max(2.0, duration_s * 0.1):
            parsed.warnings.add(
                "TELEMETRY_DURATION_MISMATCH",
                f"Video is ~{duration_s:.2f}s but telemetry is ~{telemetry_duration:.2f}s.",
            )
        for warning in extraction.warnings:
            parsed.warnings.add(warning, "Decoded timestamps were unavailable for some frames.")
        telemetry_path = run_dir / "normalized_telemetry.csv"
        write_telemetry_csv(parsed.records, telemetry_path)
        telemetry_meta_path = run_dir / "normalized_telemetry.meta.json"
        metadata = parsed.meta(telemetry_sha256_file(telemetry))
        metadata["preprocessing_source"] = "AUTO_FROM_IMMUTABLE_RUN_INPUTS"
        metadata["frame_selection_method"] = "QUALITY_GATED_BLUR_EXPOSURE_REDUNDANCY"
        metadata["candidate_frame_count"] = len(candidates)
        metadata["selected_frame_count"] = len(selected_rows)
        metadata["video_duration_s"] = duration_s
        metadata["quality_score_means"] = averages
        metadata["frame_quality_gate"] = {
            "candidate_count": len(rows),
            "eligible_count": len(quality_eligible_rows),
            "rejected_count": rejected_count,
            "selected_count": len(selected_rows),
            "minimum_laplacian_variance": record.config.frame_min_laplacian_variance,
            "minimum_exposure_score": record.config.frame_min_exposure_score,
            "relative_sharpness_floor": record.config.frame_relative_sharpness_floor,
            "operator_override_count": sum(str(row["override"]) != "NONE" for row in rows),
        }
        atomic_json(telemetry_meta_path, metadata)
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            50,
            f"Normalized {len(parsed.records)} telemetry samples and finalized preprocessing.",
        )
        return [
            run_dir / "frame_index.csv",
            scores_path,
            keyframes_path,
            contact_sheet_path,
            telemetry_path,
            telemetry_meta_path,
            *selected_paths,
        ]

    def _merge_telemetry_metadata(self, record: RunRecord, warnings: list[dict[str, str]]) -> Path:
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
            if str(warning.get("code")) == "SYNTHETIC_TELEMETRY":
                record.telemetry_origin = ProvenanceOrigin.SYNTHETIC
                record.source_provenance = ProvenanceOrigin.SYNTHETIC
                record.synthetic_fixture = True
                self.store.save_run(record)
        ingest_path = run_dir / "ingest_report.json"
        ingest = json.loads(ingest_path.read_text(encoding="utf-8"))
        ingest["telemetry_normalization"] = metadata
        ingest["warnings"] = warnings
        ingest["source_provenance"] = record.source_provenance
        ingest["video_origin"] = record.video_origin
        ingest["telemetry_origin"] = record.telemetry_origin
        ingest["genuine_real_evidence"] = record.source_provenance == ProvenanceOrigin.REAL
        atomic_json(ingest_path, ingest)
        return ingest_path

    def _align_to_local_metric(
        self, record: RunRecord, warnings: list[dict[str, str]]
    ) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
            transform_path = run_dir / "local_transform.json"
            sync_path = run_dir / "sync_report.json"
            sync_report = {
                "schema_version": "1.0",
                "telemetry_offset_s": 0.0,
                "offset_source": OffsetSource.NOT_APPLICABLE,
                "rmse_before_m": 0.0,
                "rmse_after_m": 0.0,
                "matched_camera_count": 10,
                "inlier_count": 10,
                "synthetic_fixture": True,
            }
            atomic_json(sync_path, sync_report)
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
                    "telemetry_offset_s": 0.0,
                    "offset_source": OffsetSource.NOT_APPLICABLE,
                    "rmse_before_m": 0.0,
                    "rmse_after_m": 0.0,
                },
            )
            record.telemetry_offset_s = 0.0
            record.offset_source = OffsetSource.NOT_APPLICABLE
            record.rmse_before_m = 0.0
            record.rmse_after_m = 0.0
            record.matched_camera_count = 10
            record.inlier_count = 10
            self.store.save_run(record)
            local_ply = run_dir / "sparse" / "sparse_local.ply"
            shutil.copyfile(run_dir / "sparse" / "sparse.ply", local_ply)
            return [transform_path, sync_path, local_ply]

        keyframe_payload = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
        keyframes = (
            keyframe_payload.get("frames", [])
            if isinstance(keyframe_payload, dict)
            else keyframe_payload
        )
        if not isinstance(keyframes, list):
            raise PipelineError(
                "keyframes.json must be a list or an object containing a frames list."
            )
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
        frame_times: list[float] = []
        eligible_rows: list[dict[str, str]] = []
        for row in pose_rows:
            timestamp = timestamps.get(Path(row["image_name"]).name)
            if timestamp is None:
                continue
            sfm_points.append([float(row["sfm_x"]), float(row["sfm_y"]), float(row["sfm_z"])])
            frame_times.append(timestamp)
            eligible_rows.append(row | {"timestamp_s": str(timestamp)})
        if len(sfm_points) < 3:
            raise PipelineError(
                "Fewer than three registered cameras could be joined to telemetry timestamps; "
                "verify keyframe filenames and synchronization."
            )
        calibration = calibrate_telemetry_offset(
            np.array(sfm_points),
            np.array(frame_times),
            times,
            enu,
            manual_offset_s=record.config.telemetry_offset_s,
            manual_source=record.config.telemetry_offset_source,
        )
        selected = calibration.selected
        transform = selected.transform
        selected_sfm = np.array(sfm_points)[selected.matched_indices]
        aligned = transform.apply(selected_sfm)
        matched_rows = [eligible_rows[index] for index in selected.matched_indices]
        with pose_path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(matched_rows[0]) + [
                "telemetry_timestamp_s",
                "x_m",
                "y_m",
                "z_m",
                "alignment_residual_m",
                "alignment_inlier",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(matched_rows):
                row.update(
                    {
                        "telemetry_timestamp_s": (float(row["timestamp_s"]) + selected.offset_s),
                        "x_m": aligned[index, 0],
                        "y_m": aligned[index, 1],
                        "z_m": aligned[index, 2],
                        "alignment_residual_m": transform.residuals_m[index],
                        "alignment_inlier": bool(transform.inliers[index]),
                    }
                )
                writer.writerow(row)
        transform_path = run_dir / "local_transform.json"
        sync_path = run_dir / "sync_report.json"
        sync_report = calibration.as_report()
        selected_keyframes = [
            item for item in keyframes if isinstance(item, dict) and item.get("selected", True)
        ]
        out_of_range: list[int] = []
        latitudes = np.array([float(row["lat"]) for row in telemetry])
        longitudes = np.array([float(row["lon"]) for row in telemetry])
        altitudes = np.array([float(row["alt_m"]) for row in telemetry])
        for item in selected_keyframes:
            shifted_timestamp = float(item["timestamp_s"]) + selected.offset_s
            item["telemetry_timestamp_s"] = shifted_timestamp
            if shifted_timestamp < times[0] or shifted_timestamp > times[-1]:
                item["telemetry_status"] = "OUT_OF_RANGE"
                out_of_range.append(int(item["frame_index"]))
                continue
            item["telemetry_status"] = "INTERPOLATED"
            item["telemetry"] = {
                "lat": float(np.interp(shifted_timestamp, times, latitudes)),
                "lon": float(np.interp(shifted_timestamp, times, longitudes)),
                "alt_m": float(np.interp(shifted_timestamp, times, altitudes)),
            }
        sync_report["out_of_range_frame_indices"] = out_of_range
        sync_report["selected_keyframe_count"] = len(selected_keyframes)
        if out_of_range:
            warnings.append(
                {
                    "code": "TELEMETRY_EXTRAPOLATION_REJECTED",
                    "message": (
                        f"{len(out_of_range)} selected frame(s) fell outside telemetry coverage "
                        "after applying the run-specific offset; no extrapolated position was used."
                    ),
                }
            )
        atomic_json(run_dir / "keyframes.json", keyframe_payload)
        atomic_json(sync_path, sync_report)
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
                "telemetry_offset_s": selected.offset_s,
                "offset_source": calibration.source,
                "rmse_before_m": calibration.before.rmse_m,
                "rmse_after_m": selected.rmse_m,
            },
        )
        record.telemetry_offset_s = selected.offset_s
        record.offset_source = calibration.source
        record.rmse_before_m = calibration.before.rmse_m
        record.rmse_after_m = selected.rmse_m
        record.matched_camera_count = len(selected.matched_indices)
        record.inlier_count = selected.inlier_count
        self.store.save_run(record)
        local_ply = run_dir / "sparse" / "sparse_local.ply"
        transform_ply(run_dir / "sparse" / "sparse.ply", local_ply, transform)
        return [pose_path, transform_path, sync_path, local_ply, run_dir / "keyframes.json"]

    def _reconstruct(self, record: RunRecord) -> ReconstructionResult:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
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

    def _run_optional_dense(
        self,
        record: RunRecord,
        result: ReconstructionResult,
        alignment_report: dict[str, object],
        warnings: list[dict[str, str]],
    ) -> list[Path]:
        if not record.config.enable_dense_reconstruction:
            return []
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        residuals = np.asarray(alignment_report.get("residuals_m", []), dtype=float)
        inlier_count = int(alignment_report.get("inlier_count", 0))
        transform = SimilarityTransform(
            scale=float(alignment_report.get("scale", 1.0)),
            rotation=np.asarray(alignment_report.get("rotation", np.eye(3)), dtype=float),
            translation=np.asarray(
                alignment_report.get("translation_m", [0.0, 0.0, 0.0]), dtype=float
            ),
            inliers=np.ones(len(residuals), dtype=bool),
            residuals_m=residuals,
        )
        selection_path = run_dir / "sparse" / "model_selection.json"
        selected_model = ""
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_model = str(selection.get("selected_model", ""))
        sparse_model_dir = run_dir / "sparse" / "model" / selected_model
        scene_analysis_path = run_dir / "scene_analysis.json"
        scene_analysis = (
            json.loads(scene_analysis_path.read_text(encoding="utf-8"))
            if scene_analysis_path.is_file()
            else {}
        )
        mask_dir = run_dir / "masks" / "reconstruction"
        if scene_analysis.get("masking_decision") != "APPLIED":
            mask_dir = None
        context = DenseContext(
            run_dir=run_dir,
            frames_dir=run_dir / "frames",
            sparse_model_dir=sparse_model_dir,
            registered_images=result.metrics.registered_frames,
            transform=transform,
            mask_dir=mask_dir,
            reconstruction_target=record.config.reconstruction_target,
            scene_analysis=scene_analysis,
            profile=record.config.profile,
        )
        gate_reasons: list[str] = []
        if record.synthetic_fixture:
            gate_reasons.append("synthetic fixtures cannot produce real dense evidence")
        if result.metrics.registration_rate < 0.8:
            gate_reasons.append("sparse registration is below the 80% gate")
        if inlier_count < 3 or transform.scale <= 0:
            gate_reasons.append("local metric alignment gate did not produce three valid inliers")
        if not sparse_model_dir.is_dir() and not record.synthetic_fixture:
            gate_reasons.append("selected sparse COLMAP model directory is unavailable")
        if scene_analysis.get("dense_suitability") == "BLOCKED_WITHOUT_MASK" and mask_dir is None:
            gate_reasons.append(
                "scene diagnostics found extreme unstable/featureless content and no mask was applied"
            )
        if record.config.reconstruction_target == "PRIMARY_SUBJECT" and mask_dir is None:
            gate_reasons.append(
                "PRIMARY_SUBJECT reconstruction requires a complete applied mask set"
            )
        provider = (
            UnavailableProvider("; ".join(gate_reasons))
            if gate_reasons
            else select_dense_provider(
                record.config.dense_provider, require_masks=mask_dir is not None
            )
        )
        dense_result = run_dense_stage(context, provider)
        warnings.extend(dense_result.warnings)
        return dense_result.artifacts

    def _run_optional_surface_completion(
        self,
        record: RunRecord,
        result: ReconstructionResult,
        alignment_report: dict[str, object],
        warnings: list[dict[str, str]],
    ) -> list[Path]:
        if not record.config.enable_surface_completion:
            return []
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        dense_geometry = run_dir / "dense" / "fused.ply"
        sparse_geometry = run_dir / "sparse" / "sparse_local.ply"
        if dense_geometry.is_file():
            source_geometry = dense_geometry
            source_kind = "DENSE_OBSERVED_MVS"
        else:
            source_geometry = sparse_geometry
            source_kind = "SPARSE_OBSERVED_SFM"
        gate_reasons: list[str] = []
        if record.synthetic_fixture:
            gate_reasons.append("synthetic fixtures cannot produce completion evidence")
        if result.metrics.registration_rate < 0.8:
            gate_reasons.append("sparse registration is below the 80% gate")
        if int(alignment_report.get("inlier_count", 0)) < 3:
            gate_reasons.append("local metric alignment has fewer than three inliers")
        if not source_geometry.is_file():
            gate_reasons.append("no local-metric observed geometry is available")
        context = SurfaceCompletionContext(
            run_dir=run_dir,
            source_geometry=source_geometry,
            source_geometry_kind=source_kind,
            camera_poses=run_dir / "camera_poses.csv",
            selected_frames=run_dir / "keyframes.json",
            model_path=(
                Path(record.config.surface_completion_model_path).expanduser().resolve()
                if record.config.surface_completion_model_path
                else None
            ),
            sample_count=record.config.surface_completion_samples,
        )
        provider = (
            UnavailableSurfaceCompletionProvider("; ".join(gate_reasons))
            if gate_reasons
            else ExternalSurfaceCompletionProvider()
        )
        completion = run_surface_completion_stage(context, provider)
        warnings.extend(completion.warnings)
        return completion.artifacts

    def run(self, run_id: str) -> RunRecord:
        record = self.store.get_run(run_id)
        warnings: list[dict[str, str]] = []
        try:
            self._transition(record, RunStatus.INGESTING, 10, "Inspecting immutable input assets.")
            ingest, warnings = self._probe(record)
            self.store.register_artifacts(record, [ingest])
            self._transition(
                record,
                RunStatus.PREPROCESSING,
                30,
                "Preparing scored frame selection and normalized telemetry.",
            )
            handoff = self._preprocess_contract(record)
            self.store.register_artifacts(record, handoff)
            run_dir = self.store.run_dir(record.project_id, record.run_id)
            keyframes_path = run_dir / "keyframes.json"
            keyframe_payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
            frame_rows = keyframe_payload.get("frames", [])
            scene_path, _, scene_warnings = analyze_scene(
                run_dir,
                frame_rows,
                reconstruction_target=record.config.reconstruction_target,
                masking_mode=record.config.masking_mode,
            )
            warnings.extend(scene_warnings)
            self.store.register_artifacts(record, [scene_path])
            if record.config.enable_segmentation:
                segmentation_artifacts, segmentation_warnings = run_optional_segmentation(
                    run_dir,
                    record.run_id,
                    frame_rows,
                    record.config.segmentation_model_path,
                    reconstruction_target=record.config.reconstruction_target,
                    masking_mode=record.config.masking_mode,
                )
                warnings.extend(segmentation_warnings)
                atomic_json(keyframes_path, keyframe_payload)
                self.store.register_artifacts(
                    record, [keyframes_path, scene_path, *segmentation_artifacts]
                )
                blocking = [
                    warning["message"]
                    for warning in segmentation_warnings
                    if warning["code"].startswith("REQUIRED_SEGMENTATION")
                ]
                if blocking:
                    raise PipelineError(
                        "Required reconstruction masking is unavailable: " + blocking[0]
                    )
            elif scene_warnings:
                warnings.append(
                    {
                        "code": "SCENE_MASK_RECOMMENDATION_NOT_APPLIED",
                        "message": (
                            "Scene analysis recommended masking, but optional segmentation was "
                            "not enabled; reconstruction will remain unmasked and dense gates may block."
                        ),
                    }
                )
            updated_ingest = self._merge_telemetry_metadata(record, warnings)
            self.store.register_artifacts(record, [updated_ingest])
            self._transition(record, RunStatus.RECONSTRUCTING, 55, "Running sparse reconstruction.")
            result = self._reconstruct(record)
            self.store.register_artifacts(record, result.artifacts)
            alignment = self._align_to_local_metric(record, warnings)
            self.store.register_artifacts(record, alignment)
            metrics_path = (
                self.store.run_dir(record.project_id, record.run_id) / "sparse_metrics.json"
            )
            atomic_json(
                metrics_path,
                result.metrics.model_dump(mode="json")
                | {
                    "registration_rate": result.metrics.registration_rate,
                    "synthetic_fixture": record.synthetic_fixture,
                },
            )
            benchmark_path = (
                self.store.run_dir(record.project_id, record.run_id) / "matcher_benchmark.json"
            )
            write_matcher_benchmark(benchmark_path, result.metrics, None)
            self.store.register_artifacts(record, [metrics_path, benchmark_path])
            alignment_report = json.loads(
                (
                    self.store.run_dir(record.project_id, record.run_id) / "local_transform.json"
                ).read_text(encoding="utf-8")
            )
            if record.config.enable_dense_reconstruction:
                self._transition(
                    record,
                    RunStatus.RECONSTRUCTING,
                    75,
                    "Running optional dense visual reconstruction after sparse metric gates.",
                )
                dense_artifacts = self._run_optional_dense(
                    record, result, alignment_report, warnings
                )
                self.store.register_artifacts(record, dense_artifacts)
            if record.config.enable_surface_completion:
                self._transition(
                    record,
                    RunStatus.RECONSTRUCTING,
                    80,
                    "Running optional AI surface completion as non-measurable visual output.",
                )
                completion_artifacts = self._run_optional_surface_completion(
                    record, result, alignment_report, warnings
                )
                self.store.register_artifacts(record, completion_artifacts)
            self._transition(
                record, RunStatus.REPORTING, 85, "Generating trust and known-distance report."
            )
            quality_path = (
                self.store.run_dir(record.project_id, record.run_id) / "quality_report.json"
            )
            confidence_available = False
            if any(
                artifact.relative_path == "point_confidence.json" for artifact in record.artifacts
            ):
                try:
                    validate_point_confidence_for_ply(
                        self.store.run_dir(record.project_id, record.run_id)
                        / "point_confidence.json",
                        self.store.run_dir(record.project_id, record.run_id)
                        / "sparse"
                        / "sparse_local.ply",
                    )
                    confidence_available = True
                except ValueError as exc:
                    warnings.append({"code": "INVALID_CONFIDENCE_ARTIFACT", "message": str(exc)})
            report = build_quality_report(
                record,
                result.metrics,
                warnings,
                alignment=alignment_report,
                confidence_available=confidence_available,
                scene_analysis=(
                    json.loads(scene_path.read_text(encoding="utf-8"))
                    if scene_path.is_file()
                    else None
                ),
                frame_quality=(
                    json.loads(
                        (
                            self.store.run_dir(record.project_id, record.run_id) / "keyframes.json"
                        ).read_text(encoding="utf-8")
                    ).get("frame_quality_gate", {})
                ),
                surface_completion=(
                    json.loads((run_dir / "completion_report.json").read_text(encoding="utf-8"))
                    if (run_dir / "completion_report.json").is_file()
                    else None
                ),
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
                StageEvent(
                    stage=RunStatus.FAILED,
                    status="FAILED",
                    progress=record.progress,
                    message=str(exc),
                )
            )
            self.store.save_run(record)
            return record
