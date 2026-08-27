import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sih26158.colmap import ColmapRunner
from sih26158.confidence import classify_observed_point, validate_point_confidence_for_ply
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


def test_deterministic_ply_and_confidence_share_point_order(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(
        "# cameras\n"
        "1 1 0 0 0 -1 0 0 1 a.jpg\n0 0 -1\n"
        "2 1 0 0 0 1 0 0 1 b.jpg\n0 0 -1\n"
        "3 1 0 0 0 0 -1 0 1 c.jpg\n0 0 -1\n"
        "4 1 0 0 0 0 1 0 1 d.jpg\n0 0 -1\n",
        encoding="utf-8",
    )
    points = tmp_path / "points3D.txt"
    points.write_text(
        "10 0 0 0 10 20 30 0.5 1 0 2 0 3 0 4 0\n"
        "3 2 0 0 40 50 60 3.0 1 0 3 0\n",
        encoding="utf-8",
    )
    ply = tmp_path / "sparse.ply"
    confidence = tmp_path / "point_confidence.json"
    ColmapRunner._export_points_and_confidence(points, images, ply, confidence)

    payload = json.loads(confidence.read_text(encoding="utf-8"))
    assert [item["point_id"] for item in payload["points"]] == [0, 1]
    assert payload["points"][0]["confidence_class"] == "OBSERVED_LOW"
    assert payload["points"][1]["confidence_class"] == "OBSERVED_HIGH"
    ply_text = ply.read_text(encoding="utf-8")
    assert ply_text.split("end_header\n", 1)[1].splitlines()[0] == "2 0 0 40 50 60"
    validate_point_confidence_for_ply(confidence, ply)


def test_confidence_thresholds_are_geometric_not_rgb() -> None:
    assert classify_observed_point(4, 1.0, 5.0).value == "OBSERVED_HIGH"
    assert classify_observed_point(3, 2.0, 2.0).value == "OBSERVED_MEDIUM"
    assert classify_observed_point(2, 0.1, 30.0).value == "OBSERVED_LOW"


def test_confidence_count_mismatch_is_rejected(tmp_path: Path) -> None:
    ply = tmp_path / "cloud.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n1 1 1\n",
        encoding="utf-8",
    )
    confidence = tmp_path / "point_confidence.json"
    confidence.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "point_order": "PLY_VERTEX_ORDER",
                "points": [
                    {
                        "point_id": 0,
                        "supporting_views": 3,
                        "track_length": 3,
                        "reprojection_error": 0.5,
                        "triangulation_angle": 4.0,
                        "confidence_class": "OBSERVED_MEDIUM",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="point count"):
        validate_point_confidence_for_ply(confidence, ply)
