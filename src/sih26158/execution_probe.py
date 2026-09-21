from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import utc_now


def run_colmap_execution_probe(*, use_gpu: bool, timeout_s: float = 120) -> dict[str, Any]:
    """Opt-in, bounded feature/matching execution against generated non-evidence images."""

    binary = shutil.which("colmap")
    if binary is None:
        return {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "status": "BLOCKED",
            "reason": "COLMAP executable is unavailable",
            "probe_scope": "actual feature extraction and matching; not reconstruction evidence",
        }
    with tempfile.TemporaryDirectory(prefix="sih-colmap-probe-") as temporary:
        root = Path(temporary)
        images = root / "images"
        images.mkdir()
        rng = np.random.default_rng(26158)
        base = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for index, shift in enumerate((0, 8, 16)):
            matrix = np.float32([[1, 0, shift], [0, 1, shift // 2]])
            image = cv2.warpAffine(base, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
            if not cv2.imwrite(str(images / f"probe_{index}.jpg"), image):
                raise OSError("Could not write generated probe image")
        database = root / "database.db"
        commands = [
            [
                binary,
                "feature_extractor",
                "--database_path",
                str(database),
                "--image_path",
                str(images),
                "--ImageReader.single_camera",
                "1",
                "--FeatureExtraction.use_gpu",
                "1" if use_gpu else "0",
                "--FeatureExtraction.max_image_size",
                "640",
            ],
            [
                binary,
                "exhaustive_matcher",
                "--database_path",
                str(database),
                "--FeatureMatching.use_gpu",
                "1" if use_gpu else "0",
            ],
        ]
        results: list[dict[str, Any]] = []
        started = time.monotonic()
        for command in commands:
            command_started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(1.0, timeout_s - (time.monotonic() - started)),
                    env=os.environ.copy(),
                )
                results.append(
                    {
                        "command": command[1],
                        "returncode": completed.returncode,
                        "runtime_s": time.monotonic() - command_started,
                        "output_tail": (completed.stdout + completed.stderr)[-2000:],
                    }
                )
                if completed.returncode:
                    break
            except subprocess.TimeoutExpired as exc:
                results.append(
                    {
                        "command": command[1],
                        "returncode": None,
                        "runtime_s": time.monotonic() - command_started,
                        "output_tail": str(exc),
                        "timed_out": True,
                    }
                )
                break
        passed = len(results) == len(commands) and all(item["returncode"] == 0 for item in results)
        return {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "status": "PASS" if passed else "FAILED",
            "probe_scope": (
                "Actual COLMAP feature extraction and matching on generated images; this does not "
                "validate sparse mapping, a real dataset, accuracy, completeness, or runtime targets."
            ),
            "requested_backend": "GPU" if use_gpu else "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "total_runtime_s": time.monotonic() - started,
            "database_created": database.is_file() and database.stat().st_size > 0,
            "commands": results,
        }
