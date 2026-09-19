from pathlib import Path

import cv2
import numpy as np

from sih26158.segmentation import run_optional_segmentation


def test_mocked_segmentation_declares_masks_without_model_download(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame_000001.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 1, "image_name": image_path.name, "selected": True}
    ]

    def provider(_: Path) -> np.ndarray:
        mask = np.zeros((20, 30), np.uint8)
        mask[:, :3] = 255
        return mask

    artifacts, warnings = run_optional_segmentation(
        tmp_path, "run_test", keyframes, None, provider=provider
    )
    assert not warnings
    assert {path.name for path in artifacts} == {
        "frame_000001_dynamic.png",
        "segmentation_comparison.json",
    }
    assert keyframes[0]["dynamic_mask_fraction"] == 0.1
    assert str(keyframes[0]["mask_url"]).startswith("/api/runs/run_test/artifacts/masks/")


def test_missing_optional_model_falls_back_to_unmasked_frames(tmp_path: Path) -> None:
    artifacts, warnings = run_optional_segmentation(
        tmp_path, "run_test", [], str(tmp_path / "missing.pt")
    )
    assert artifacts == []
    assert warnings[0]["code"] == "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES"


def test_generated_masks_do_not_claim_reconstruction_effect(tmp_path: Path) -> None:
    import json
    from sih26158.colmap import ColmapRunner
    from sih26158.models import ArtifactEntry, MatcherMetrics, RunConfig, RunRecord
    from sih26158.report import build_quality_report

    (tmp_path / "frames").mkdir()
    artifacts, _ = run_optional_segmentation(
        tmp_path, "r", [{"image_name": "a.jpg"}], None,
        provider=lambda _: np.zeros((10, 10), np.uint8),
    )
    segmentation = json.loads(artifacts[-1].read_text())
    commands = ColmapRunner().build_commands(tmp_path / "frames", tmp_path, RunConfig())
    assert commands.masking_applied is False
    record = RunRecord(project_id="p", run_id="r", config=RunConfig(), artifacts=[
        ArtifactEntry(name="segmentation", relative_path=artifacts[-1].name,
                      media_type="application/json", size_bytes=0, sha256="0" * 64)
    ])
    metrics = MatcherMetrics(matcher="SIFT", eligible_frames=10, registered_frames=9,
        median_reprojection_error_px=1, p95_reprojection_error_px=1, runtime_s=1,
        masking_applied=commands.masking_applied)
    report = build_quality_report(record, metrics, [])
    assert report["masking_applied"] is False
    assert report["masking_note"] == segmentation["masking_note"]
    assert "not passed to COLMAP" in report["masking_note"]
