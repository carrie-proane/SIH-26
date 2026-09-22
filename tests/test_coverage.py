import numpy as np
import pytest

from sih26158.coverage import DepthView, analyze_surface_support, depth_array_sha256

VIDEO_HASH = "a" * 64


def view(name, depth, *, selected=False, video_hash=VIDEO_HASH):
    pose = np.eye(4)
    pose[2, 3] = 2
    intrinsics = np.array([[20, 0, 30], [0, 20, 30], [0, 0, 1]], dtype=float)
    return DepthView(name, video_hash, pose, intrinsics, depth, depth_array_sha256(depth), selected)


def test_frustum_membership_does_not_prove_visibility(observed_ring):
    root, source, _ = observed_ring
    occluder = view("blocked", np.ones((60, 60)), selected=True)
    report = analyze_surface_support(root, source, (occluder,), video_sha256=VIDEO_HASH)
    assert report.views[0]["frustum_sample_count"] == 16
    assert report.views[0]["occluded_sample_count"] == 16
    assert report.available_support_statistics["selected_view_supported_centroids"] == 0
    assert report.observed_surface_completeness_percent is None
    assert report.denominator is None and report.unavailable_reasons


def test_additional_frames_require_new_depth_consistent_support(observed_ring):
    root, source, _ = observed_ring
    blocked = view("selected", np.ones((60, 60)), selected=True)
    first = view("candidate-a", np.full((60, 60), 2.0))
    duplicate = view("candidate-b", np.full((60, 60), 2.0))
    report = analyze_surface_support(
        root, source, (blocked, duplicate, first), video_sha256=VIDEO_HASH
    )
    assert [p["frame_id"] for p in report.additional_frame_proposals] == ["candidate-a"]
    assert report.additional_frame_proposals[0]["additional_supported_sample_count"] == 16
    assert report.observed_surface_area_m2 is None


def test_unknown_depth_and_free_space_are_not_observed_surface_support(observed_ring):
    root, source, _ = observed_ring
    for depth in (np.zeros((60, 60)), np.full((60, 60), np.nan), np.full((60, 60), 3.0)):
        report = analyze_surface_support(
            root, source, (view("unknown", depth, selected=True),), video_sha256=VIDEO_HASH
        )
        assert report.available_support_statistics["selected_view_supported_centroids"] == 0


def test_cross_video_depth_hash_relative_depth_and_scaled_pose_are_rejected(observed_ring):
    from dataclasses import replace

    root, source, _ = observed_ring
    good = view("frame", np.full((60, 60), 2.0))
    bad_pose = np.eye(4) * 2
    bad_pose[3, 3] = 1
    for bad in (
        replace(good, video_sha256="b" * 64),
        replace(good, depth_sha256="0" * 64),
        replace(good, depth_semantics="RELATIVE_DEPTH"),
        replace(good, world_to_camera=bad_pose),
    ):
        with pytest.raises(ValueError):
            analyze_surface_support(root, source, (bad,), video_sha256=VIDEO_HASH)


def test_without_declared_mesh_or_depth_coverage_is_unavailable(tmp_path):
    report = analyze_surface_support(tmp_path, None, (), video_sha256=VIDEO_HASH)
    assert report.status == "UNAVAILABLE"
    assert report.observed_surface_completeness_percent is None


def test_changed_source_mesh_returns_unavailable_statistics(observed_ring):
    root, source, _ = observed_ring
    (root / source.artifact.path).write_bytes(b"tampered")
    report = analyze_surface_support(
        root, source, (view("frame", np.full((60, 60), 2.0)),), video_sha256=VIDEO_HASH
    )
    assert report.status == "UNAVAILABLE"
    assert report.available_support_statistics == {}
    assert "checksum" in report.unavailable_reasons[-1]
