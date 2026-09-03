import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sih26158.colmap import CameraModelAttempt, ColmapRunner, SparseModelCandidate
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


def test_accuracy_profile_uses_exhaustive_guided_matching_and_stricter_mapper(
    tmp_path: Path,
) -> None:
    commands = ColmapRunner().build_commands(
        tmp_path / "frames", tmp_path, RunConfig(profile="accurate")
    )

    assert commands[1][1] == "exhaustive_matcher"
    assert "--FeatureMatching.guided_matching" in commands[1]
    assert "--SiftExtraction.estimate_affine_shape" in commands[0]
    assert commands[2][commands[2].index("--Mapper.filter_max_reproj_error") + 1] == "3"


def test_sequential_loop_detection_requires_declared_local_vocabulary_tree(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "vocab.bin"
    tree.write_bytes(b"fixture")
    command = ColmapRunner().build_commands(
        tmp_path / "frames",
        tmp_path,
        RunConfig(matching_strategy="SEQUENTIAL", vocab_tree_path=str(tree)),
    )[1]

    assert command[1] == "sequential_matcher"
    assert command[command.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert command[command.index("--SequentialMatching.vocab_tree_path") + 1] == str(tree)


def test_trusted_camera_calibration_can_be_fixed(tmp_path: Path) -> None:
    commands = ColmapRunner().build_commands(
        tmp_path / "frames",
        tmp_path,
        RunConfig(
            camera_model="OPENCV",
            camera_params="1200,1200,960,540,0,0,0,0",
            refine_intrinsics=False,
        ),
    )

    assert commands[0][commands[0].index("--ImageReader.camera_params") + 1].startswith("1200")
    assert commands[2][commands[2].index("--Mapper.ba_refine_focal_length") + 1] == "0"
    assert commands[2][commands[2].index("--Mapper.ba_refine_extra_params") + 1] == "0"


def test_trusted_camera_parameters_force_fixed_model_policy() -> None:
    config = RunConfig(camera_params="1200,1200,960,540,0,0,0,0")

    assert config.camera_model_policy == "FIXED"
    assert ColmapRunner._camera_model_candidates(config) == ["SIMPLE_RADIAL"]


def test_auto_camera_policy_is_bounded_and_deterministic() -> None:
    assert ColmapRunner._camera_model_candidates(RunConfig(camera_model="RADIAL")) == [
        "RADIAL",
        "SIMPLE_RADIAL",
        "OPENCV",
    ]


def test_camera_intrinsic_diagnostics_reject_unstable_distortion(tmp_path: Path) -> None:
    plausible = tmp_path / "plausible.txt"
    plausible.write_text("1 SIMPLE_RADIAL 1920 1080 1400 960 540 0.05\n", encoding="utf-8")
    unstable = tmp_path / "unstable.txt"
    unstable.write_text("1 SIMPLE_RADIAL 1920 1080 1400 960 540 9.0\n", encoding="utf-8")

    assert ColmapRunner._inspect_intrinsics(plausible)["plausible"] is True
    rejected = ColmapRunner._inspect_intrinsics(unstable)
    assert rejected["plausible"] is False
    assert any(
        str(reason).startswith("UNSTABLE_RADIAL_DISTORTION")
        for reason in rejected["rejection_reasons"]
    )


def test_bounded_recovery_uses_dense_sift_and_exhaustive_matching(tmp_path: Path) -> None:
    commands = ColmapRunner().build_commands(
        tmp_path / "frames",
        tmp_path,
        RunConfig(matching_strategy="SEQUENTIAL"),
        workspace=tmp_path / "attempt",
        recovery=True,
    )

    assert commands[0][commands[0].index("--SiftExtraction.max_num_features") + 1] == "32768"
    assert commands[0][commands[0].index("--SiftExtraction.peak_threshold") + 1] == "0.003"
    assert commands[1][1] == "exhaustive_matcher"
    assert commands[2][commands[2].index("--Mapper.max_reg_trials") + 1] == "5"


def test_camera_attempt_discards_private_workspace_left_by_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        (frames / f"frame_{index:04d}.jpg").write_bytes(b"image")
    workspace = tmp_path / "sparse" / "attempts" / "00_simple_radial"
    workspace.mkdir(parents=True)
    stale_database = workspace / "sparse" / "database.db"
    stale_database.parent.mkdir()
    stale_database.write_bytes(b"partial database")

    runner = ColmapRunner()
    monkeypatch.setattr(runner, "build_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runner,
        "select_best_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("stop after reset")),
    )

    attempt = runner._run_camera_attempt(
        frames,
        tmp_path,
        RunConfig(),
        attempt_id="00_simple_radial",
        camera_model="SIMPLE_RADIAL",
        recovery=False,
        log=tmp_path / "logs" / "colmap.log",
    )

    assert attempt.status == "FAILED"
    assert workspace.is_dir()
    assert not stale_database.exists()


def test_recovery_triggers_only_when_initial_sparse_gates_are_weak(tmp_path: Path) -> None:
    good_sparse = SparseModelCandidate(tmp_path / "good", 90, 0.7, 1.8)
    weak_sparse = SparseModelCandidate(tmp_path / "weak", 50, 2.0, 5.0)
    good = CameraModelAttempt(
        "good",
        "SIMPLE_RADIAL",
        tmp_path,
        False,
        "COMPLETED",
        [],
        sparse_model=good_sparse,
        intrinsics={"plausible": True},
    )
    weak = CameraModelAttempt(
        "weak",
        "RADIAL",
        tmp_path,
        False,
        "COMPLETED",
        [],
        sparse_model=weak_sparse,
        intrinsics={"plausible": True},
    )

    assert ColmapRunner._retry_required(good, 100) == (False, [])
    retry, reasons = ColmapRunner._retry_required(weak, 100)
    assert retry is True
    assert "REGISTRATION_BELOW_80_PERCENT" in reasons
    assert "P95_REPROJECTION_ABOVE_4_PX" in reasons


def test_colmap_sparse_feature_extraction_consumes_complete_masks(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    masks = tmp_path / "masks" / "reconstruction"
    frames.mkdir()
    masks.mkdir(parents=True)
    (frames / "frame.jpg").write_bytes(b"image")
    (masks / "frame.jpg.png").write_bytes(b"mask")

    feature_command = ColmapRunner().build_commands(frames, tmp_path, RunConfig())[0]

    assert feature_command[-2:] == ["--ImageReader.mask_path", str(masks)]


def test_colmap_ignores_incomplete_operational_mask_set(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    masks = tmp_path / "masks" / "reconstruction"
    frames.mkdir()
    masks.mkdir(parents=True)
    (frames / "a.jpg").write_bytes(b"image")
    (frames / "b.jpg").write_bytes(b"image")
    (masks / "a.jpg.png").write_bytes(b"mask")

    feature_command = ColmapRunner().build_commands(frames, tmp_path, RunConfig())[0]

    assert "--ImageReader.mask_path" not in feature_command


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


def test_geometry_diagnostics_expose_weak_baseline_without_claiming_scale(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(
        "1 1 0 0 0 0 0 0 1 a.jpg\n0 0 -1\n"
        "2 1 0 0 0 -0.001 0 0 1 b.jpg\n0 0 -1\n"
        "3 1 0 0 0 -0.002 0 0 1 c.jpg\n0 0 -1\n",
        encoding="utf-8",
    )
    points = tmp_path / "points3D.txt"
    points.write_text(
        "1 0 0 10 1 2 3 0.4 1 0 2 0 3 0\n2 10 0 10 4 5 6 0.5 1 0 2 0 3 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "geometry.json"

    report = ColmapRunner._write_geometry_diagnostics(points, images, output, eligible_images=3)

    assert report["measurement_geometry_gate_passed"] is False
    assert report["gates"]["camera_baseline_ratio_0_01"] is False
    assert "do not independently verify metric scale" in str(report["interpretation"])


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
        "10 0 0 0 10 20 30 0.5 1 0 2 0 3 0 4 0\n3 2 0 0 40 50 60 3.0 1 0 3 0\n",
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
