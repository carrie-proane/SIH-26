import csv
import json
import struct
from pathlib import Path

import numpy as np

from sih26158.colmap import ColmapRunner
from sih26158.models import RunConfig


def _write_binary_model(path: Path, registered_images: int, errors: list[float]) -> None:
    path.mkdir(parents=True)
    (path / "images.bin").write_bytes(struct.pack("<Q", registered_images))
    points = bytearray(struct.pack("<Q", len(errors)))
    for point_id, error in enumerate(errors):
        points.extend(struct.pack("<QdddBBBd", point_id, 0, 0, 0, 0, 0, 0, error))
        points.extend(struct.pack("<Q", 0))
    (path / "points3D.bin").write_bytes(points)


def test_colmap_uses_current_4_1_command_options(tmp_path: Path) -> None:
    commands = ColmapRunner().build_commands(tmp_path / "frames", tmp_path, RunConfig())
    flattened = [item for command in commands for item in command]
    assert "--FeatureExtraction.use_gpu" in flattened
    assert "--FeatureMatching.use_gpu" in flattened
    assert "--SiftExtraction.use_gpu" not in flattened
    assert "--SiftMatching.use_gpu" not in flattened


def test_best_sparse_model_prefers_91_registered_frames(tmp_path: Path) -> None:
    model_root = tmp_path / "sparse" / "model"
    _write_binary_model(model_root / "0", 10, [0.2, 0.3])
    _write_binary_model(model_root / "1", 91, [0.4, 0.5, 0.6])
    report_path = tmp_path / "model_selection.json"

    selected = ColmapRunner.select_best_model(model_root, report_path)

    assert selected.path == model_root / "1"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selected_model"] == "1"
    assert {item["registered_images"] for item in report["candidates"]} == {10, 91}
    assert "registered-image count descending" in report["rationale"]


def test_sparse_model_ties_use_error_then_lexical_path(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_binary_model(model_root / "z", 91, [0.2])
    _write_binary_model(model_root / "a", 91, [0.2])
    _write_binary_model(model_root / "lower_priority", 91, [0.3])

    selected = ColmapRunner.select_best_model(model_root)

    assert selected.path == model_root / "a"


def test_colmap_text_camera_pose_export(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "1 1 0 0 0 -2 -3 -4 1 frame_0001.jpg\n"
        "1.0 2.0 -1\n",
        encoding="utf-8",
    )
    output = tmp_path / "poses.csv"
    ColmapRunner._export_camera_poses(images, output)
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["image_name"] == "frame_0001.jpg"
    assert np.allclose([float(row["sfm_x"]), float(row["sfm_y"]), float(row["sfm_z"])], [2, 3, 4])
