from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import MatcherMetrics, RunConfig


class ExternalToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReconstructionResult:
    metrics: MatcherMetrics
    artifacts: list[Path]
    commands: list[list[str]]


def choose_matcher(
    sift: MatcherMetrics, learned: MatcherMetrics | None
) -> tuple[str, str]:
    if learned is None:
        return "SIFT", "Learned matcher was unavailable; the CPU-safe SIFT baseline remains selected."
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
        return [
            [
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
                "--SiftExtraction.use_gpu",
                gpu,
            ],
            [
                self.binary,
                "sequential_matcher",
                "--database_path",
                str(database),
                "--SequentialMatching.overlap",
                str(config.sequential_overlap),
                "--SiftMatching.use_gpu",
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
        models = sorted((run_dir / "sparse" / "model").glob("*/images.bin"))
        if not models:
            raise ExternalToolError(
                "COLMAP completed without a sparse model. Check colmap.log for verified-match and "
                "camera-model diagnostics."
            )
        model_dir = models[0].parent
        ply = run_dir / "sparse" / "sparse.ply"
        self._execute(
            [self.binary, "model_converter", "--input_path", str(model_dir), "--output_path", str(ply), "--output_type", "PLY"],
            log,
        )
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
        values = {k.lower().replace(" ", "_"): v for k, v in re.findall(r"^([^:]+):\s*(.+)$", analysis.stdout, re.MULTILINE)}
        registered = int(values.get("registered_images", values.get("images", len(images))))
        mean_error = float(str(values.get("mean_reprojection_error", "0")).split()[0])
        metrics = MatcherMetrics(
            matcher=config.matcher,
            eligible_frames=len(images),
            registered_frames=registered,
            median_reprojection_error_px=max(0.0, mean_error),
            p95_reprojection_error_px=max(0.0, mean_error),
            runtime_s=time.monotonic() - started,
        )
        return ReconstructionResult(
            metrics,
            [ply, poses, log, run_dir / "sparse" / "model_analysis.txt"],
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
            writer.writerow(["image_id", "image_name", "sfm_x", "sfm_y", "sfm_z", "qw", "qx", "qy", "qz"])
            writer.writerows(rows)
