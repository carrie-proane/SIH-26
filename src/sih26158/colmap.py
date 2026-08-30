from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .confidence import classify_observed_point
from .models import MatcherMetrics, RunConfig


class ExternalToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReconstructionResult:
    metrics: MatcherMetrics
    artifacts: list[Path]
    commands: list[list[str]]


@dataclass(frozen=True)
class SparseModelCandidate:
    path: Path
    registered_images: int
    median_reprojection_error_px: float | None
    p95_reprojection_error_px: float | None

    def sort_key(self) -> tuple[int, float, str]:
        error = (
            self.median_reprojection_error_px
            if self.median_reprojection_error_px is not None
            else float("inf")
        )
        return (-self.registered_images, error, str(self.path))


def choose_matcher(sift: MatcherMetrics, learned: MatcherMetrics | None) -> tuple[str, str]:
    if learned is None:
        return (
            "SIFT",
            "Learned matcher was unavailable; the CPU-safe SIFT baseline remains selected.",
        )
    registration_gain = learned.registration_rate - sift.registration_rate
    reprojection_gain = sift.median_reprojection_error_px - learned.median_reprojection_error_px
    if registration_gain > 0.005 or (registration_gain >= 0 and reprojection_gain > 0.05):
        return learned.matcher, (
            "Learned matcher selected because it improved registered-frame rate or median "
            "reprojection error without reducing registration."
        )
    return "SIFT", (
        "SIFT retained because the learned matcher did not improve registered-frame rate or "
        "reprojection error."
    )


def write_matcher_benchmark(
    output: Path, sift: MatcherMetrics, learned: MatcherMetrics | None
) -> dict[str, object]:
    selected, reason = choose_matcher(sift, learned)
    report: dict[str, object] = {
        "baseline": sift.model_dump(mode="json") | {"registration_rate": sift.registration_rate},
        "learned": None
        if learned is None
        else learned.model_dump(mode="json") | {"registration_rate": learned.registration_rate},
        "selected_matcher": selected,
        "decision": reason,
        "policy": "Keep SuperPoint+LightGlue only when registration or reprojection improves.",
        "learned_status": "BLOCKED_UNAVAILABLE" if learned is None else "EXECUTED",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class ColmapRunner:
    def __init__(self, binary: str = "colmap") -> None:
        self.binary = binary

    def doctor(self) -> str:
        location = shutil.which(self.binary)
        if location is None:
            raise ExternalToolError(
                "COLMAP is not installed or not on PATH. Install COLMAP, verify `colmap -h`, "
                "then rerun. The pipeline will not substitute synthetic geometry for a real run."
            )
        return location

    def build_commands(self, frames: Path, run_dir: Path, config: RunConfig) -> list[list[str]]:
        database = run_dir / "sparse" / "database.db"
        model = run_dir / "sparse" / "model"
        model.mkdir(parents=True, exist_ok=True)
        gpu = "1" if config.use_gpu else "0"
        feature_command = [
            self.binary,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(frames),
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            gpu,
        ]
        mask_dir = run_dir / "masks" / "reconstruction"
        image_names = (
            [
                path.name
                for path in frames.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if frames.is_dir()
            else []
        )
        if image_names and all((mask_dir / f"{name}.png").is_file() for name in image_names):
            feature_command.extend(["--ImageReader.mask_path", str(mask_dir)])
        return [
            feature_command,
            [
                self.binary,
                "sequential_matcher",
                "--database_path",
                str(database),
                "--SequentialMatching.overlap",
                str(config.sequential_overlap),
                "--FeatureMatching.use_gpu",
                gpu,
            ],
            [
                self.binary,
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(frames),
                "--output_path",
                str(model),
            ],
        ]

    def _execute(self, command: list[str], log: Path) -> None:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode:
            raise ExternalToolError(
                f"COLMAP command failed with exit code {result.returncode}. Inspect {log.name}; "
                "verify overlap, intrinsics, blur, and selected-frame spacing."
            )

    @staticmethod
    def _read_registered_image_count(images_bin: Path) -> int:
        try:
            with images_bin.open("rb") as stream:
                raw = stream.read(8)
        except OSError as exc:
            raise ExternalToolError(f"Unable to inspect COLMAP model: {images_bin}") from exc
        if len(raw) != 8:
            raise ExternalToolError(f"Malformed COLMAP images.bin: {images_bin}")
        return int(struct.unpack("<Q", raw)[0])

    @staticmethod
    def _read_point_errors(points_bin: Path) -> list[float]:
        if not points_bin.is_file():
            return []
        errors: list[float] = []
        try:
            with points_bin.open("rb") as stream:
                count_raw = stream.read(8)
                if len(count_raw) != 8:
                    raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                point_count = struct.unpack("<Q", count_raw)[0]
                record_size = struct.calcsize("<QdddBBBd")
                for _ in range(point_count):
                    record = stream.read(record_size)
                    if len(record) != record_size:
                        raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                    error = float(struct.unpack("<QdddBBBd", record)[-1])
                    errors.append(error)
                    track_length_raw = stream.read(8)
                    if len(track_length_raw) != 8:
                        raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                    track_length = struct.unpack("<Q", track_length_raw)[0]
                    stream.seek(int(track_length) * 8, 1)
        except OSError as exc:
            raise ExternalToolError(f"Unable to inspect COLMAP model: {points_bin}") from exc
        return errors

    @classmethod
    def inspect_sparse_models(cls, model_root: Path) -> list[SparseModelCandidate]:
        candidates: list[SparseModelCandidate] = []
        for images_bin in sorted(model_root.glob("*/images.bin")):
            model_dir = images_bin.parent
            errors = cls._read_point_errors(model_dir / "points3D.bin")
            candidates.append(
                SparseModelCandidate(
                    path=model_dir,
                    registered_images=cls._read_registered_image_count(images_bin),
                    median_reprojection_error_px=(float(np.median(errors)) if errors else None),
                    p95_reprojection_error_px=(
                        float(np.percentile(errors, 95)) if errors else None
                    ),
                )
            )
        return candidates

    @classmethod
    def select_best_model(
        cls, model_root: Path, report_path: Path | None = None
    ) -> SparseModelCandidate:
        candidates = cls.inspect_sparse_models(model_root)
        if not candidates:
            raise ExternalToolError(
                "COLMAP completed without a sparse model. Check colmap.log for verified-match and "
                "camera-model diagnostics."
            )
        selected = min(candidates, key=SparseModelCandidate.sort_key)
        if report_path is not None:
            report = {
                "schema_version": "1.0",
                "selection_order": [
                    "highest registered_images",
                    "lowest median_reprojection_error_px",
                    "lexical path",
                ],
                "candidates": [
                    {
                        "path": str(candidate.path.relative_to(model_root)),
                        "registered_images": candidate.registered_images,
                        "median_reprojection_error_px": candidate.median_reprojection_error_px,
                        "p95_reprojection_error_px": candidate.p95_reprojection_error_px,
                    }
                    for candidate in sorted(candidates, key=lambda item: str(item.path))
                ],
                "selected_model": str(selected.path.relative_to(model_root)),
                "rationale": (
                    f"Selected {selected.path.relative_to(model_root)} with "
                    f"{selected.registered_images} registered images. Candidates are ranked by "
                    "registered-image count descending, median reprojection error ascending, then "
                    "lexical path."
                ),
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return selected

    def run(self, frames: Path, run_dir: Path, config: RunConfig) -> ReconstructionResult:
        self.doctor()
        images = [p for p in frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(images) < 3:
            raise ExternalToolError(
                "Fewer than three selected images were provided. Supply Yosha's retained frames "
                "and keyframes.json before reconstruction."
            )
        log = run_dir / "logs" / "colmap.log"
        commands = self.build_commands(frames, run_dir, config)
        started = time.monotonic()
        for command in commands:
            self._execute(command, log)
        selection_path = run_dir / "sparse" / "model_selection.json"
        selected = self.select_best_model(run_dir / "sparse" / "model", selection_path)
        model_dir = selected.path
        text_model = run_dir / "sparse" / "text_model"
        text_model.mkdir(exist_ok=True)
        self._execute(
            [
                self.binary,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(text_model),
                "--output_type",
                "TXT",
            ],
            log,
        )
        ply = run_dir / "sparse" / "sparse.ply"
        confidence_path = run_dir / "point_confidence.json"
        self._export_points_and_confidence(
            text_model / "points3D.txt",
            text_model / "images.txt",
            ply,
            confidence_path,
        )
        poses = run_dir / "camera_poses.csv"
        self._export_camera_poses(text_model / "images.txt", poses)
        analysis = subprocess.run(
            [self.binary, "model_analyzer", "--path", str(model_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        (run_dir / "sparse" / "model_analysis.txt").write_text(
            analysis.stdout + analysis.stderr, encoding="utf-8"
        )
        values = {
            k.lower().replace(" ", "_"): v
            for k, v in re.findall(r"^([^:]+):\s*(.+)$", analysis.stdout, re.MULTILINE)
        }
        registered = selected.registered_images
        analyzer_error = float(str(values.get("mean_reprojection_error", "0")).split()[0])
        median_error = selected.median_reprojection_error_px
        p95_error = selected.p95_reprojection_error_px
        metrics = MatcherMetrics(
            matcher=config.matcher,
            eligible_frames=len(images),
            registered_frames=registered,
            median_reprojection_error_px=max(
                0.0, analyzer_error if median_error is None else median_error
            ),
            p95_reprojection_error_px=max(0.0, analyzer_error if p95_error is None else p95_error),
            runtime_s=time.monotonic() - started,
        )
        return ReconstructionResult(
            metrics,
            [
                ply,
                poses,
                log,
                run_dir / "sparse" / "model_analysis.txt",
                selection_path,
                confidence_path,
            ],
            commands,
        )

    @staticmethod
    def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        norm = np.linalg.norm([qw, qx, qy, qz])
        if norm == 0:
            raise ExternalToolError("COLMAP returned a zero-length camera quaternion.")
        qw, qx, qy, qz = np.array([qw, qx, qy, qz]) / norm
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    @classmethod
    def _export_camera_poses(cls, images_txt: Path, output: Path) -> None:
        if not images_txt.is_file():
            raise ExternalToolError("COLMAP text export did not produce images.txt.")
        rows: list[list[object]] = []
        image_line = True
        for raw in images_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            if image_line:
                values = raw.split()
                if len(values) < 10:
                    raise ExternalToolError("Malformed COLMAP images.txt camera record.")
                qw, qx, qy, qz = map(float, values[1:5])
                translation = np.array(list(map(float, values[5:8])))
                rotation = cls._quaternion_rotation(qw, qx, qy, qz)
                center = -(rotation.T @ translation)
                rows.append([values[0], values[9], *center.tolist(), qw, qx, qy, qz])
            image_line = not image_line
        if not rows:
            raise ExternalToolError("COLMAP sparse model contains no registered camera poses.")
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["image_id", "image_name", "sfm_x", "sfm_y", "sfm_z", "qw", "qx", "qy", "qz"]
            )
            writer.writerows(rows)

    @classmethod
    def _camera_centres(cls, images_txt: Path) -> dict[int, np.ndarray]:
        centres: dict[int, np.ndarray] = {}
        image_line = True
        for raw in images_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            if image_line:
                values = raw.split()
                if len(values) < 10:
                    raise ExternalToolError("Malformed COLMAP images.txt camera record.")
                qw, qx, qy, qz = map(float, values[1:5])
                translation = np.array(list(map(float, values[5:8])))
                rotation = cls._quaternion_rotation(qw, qx, qy, qz)
                centres[int(values[0])] = -(rotation.T @ translation)
            image_line = not image_line
        return centres

    @staticmethod
    def _triangulation_angle(
        point: np.ndarray, image_ids: list[int], camera_centres: dict[int, np.ndarray]
    ) -> float:
        rays = []
        for image_id in sorted(set(image_ids)):
            centre = camera_centres.get(image_id)
            if centre is None:
                continue
            ray = centre - point
            norm = float(np.linalg.norm(ray))
            if norm > 0:
                rays.append(ray / norm)
        maximum = 0.0
        for first_index, first in enumerate(rays):
            for second in rays[first_index + 1 :]:
                cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
                maximum = max(maximum, float(np.degrees(np.arccos(cosine))))
        return maximum

    @classmethod
    def _export_points_and_confidence(
        cls, points_txt: Path, images_txt: Path, ply_path: Path, confidence_path: Path
    ) -> None:
        """Export PLY and confidence together in ascending COLMAP point-ID order."""
        if not points_txt.is_file() or not images_txt.is_file():
            raise ExternalToolError("COLMAP text export lacks points3D.txt or images.txt.")
        camera_centres = cls._camera_centres(images_txt)
        points: list[tuple[int, np.ndarray, tuple[int, int, int], float, list[int]]] = []
        for raw in points_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            values = raw.split()
            if len(values) < 8 or (len(values) - 8) % 2:
                raise ExternalToolError("Malformed COLMAP points3D.txt point record.")
            source_id = int(values[0])
            position = np.array(list(map(float, values[1:4])))
            color = tuple(map(int, values[4:7]))
            error = float(values[7])
            image_ids = [int(values[index]) for index in range(8, len(values), 2)]
            points.append((source_id, position, color, error, image_ids))
        if not points:
            raise ExternalToolError("COLMAP sparse model contains no triangulated points.")
        points.sort(key=lambda item: item[0])
        ply_lines = [
            "ply",
            "format ascii 1.0",
            "comment vertex order is ascending COLMAP POINT3D_ID",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        confidence_records: list[dict[str, object]] = []
        for vertex_id, (_, position, color, error, image_ids) in enumerate(points):
            angle = cls._triangulation_angle(position, image_ids, camera_centres)
            track_length = len(image_ids)
            ply_lines.append(
                " ".join(
                    [*(f"{value:.12g}" for value in position), *(str(value) for value in color)]
                )
            )
            confidence_records.append(
                {
                    "point_id": vertex_id,
                    "supporting_views": len(set(image_ids)),
                    "track_length": track_length,
                    "reprojection_error": error,
                    "triangulation_angle": angle,
                    "confidence_class": classify_observed_point(track_length, error, angle).value,
                }
            )
        ply_path.write_text("\n".join(ply_lines) + "\n", encoding="utf-8")
        confidence_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "point_order": "PLY_VERTEX_ORDER",
                    "points": confidence_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
