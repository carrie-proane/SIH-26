import json
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
        "frame_000001.jpg.png",
        "frame_000001.jpg.mask.png",
        "segmentation_comparison.json",
    }
    assert keyframes[0]["dynamic_mask_fraction"] == 0.1
    assert str(keyframes[0]["mask_url"]).startswith("/api/runs/run_test/artifacts/masks/")


def test_missing_optional_model_falls_back_to_unmasked_frames(tmp_path: Path) -> None:
    artifacts, warnings = run_optional_segmentation(
        tmp_path, "run_test", [], str(tmp_path / "missing.pt")
    )
    assert [path.name for path in artifacts] == ["segmentation_comparison.json"]
    assert warnings[0]["code"] == "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES"


def test_primary_subject_missing_model_is_a_blocker(tmp_path: Path) -> None:
    artifacts, warnings = run_optional_segmentation(
        tmp_path,
        "run_test",
        [],
        str(tmp_path / "missing.pt"),
        reconstruction_target="PRIMARY_SUBJECT",
        masking_mode="AUTO",
    )
    report = json.loads(artifacts[0].read_text())
    assert report["status"] == "BLOCKED"
    assert warnings[0]["code"] == "REQUIRED_SEGMENTATION_UNAVAILABLE"


def test_operational_masks_invert_review_exclusion_semantics(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "image_name": image_path.name, "selected": True}
    ]

    def provider(_: Path) -> np.ndarray:
        mask = np.zeros((10, 10), np.uint8)
        mask[:, :2] = 255
        return mask

    run_optional_segmentation(tmp_path, "run_test", keyframes, None, provider=provider)
    review = cv2.imread(str(tmp_path / "masks/frame_dynamic.png"), cv2.IMREAD_GRAYSCALE)
    operational = cv2.imread(
        str(tmp_path / "masks/reconstruction/frame.jpg.mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert np.all(review[:, :2] == 255)
    assert np.all(operational[:, :2] == 0)
    assert np.all(operational[:, 2:] == 255)


def test_primary_subject_rejects_trivial_keep_everything_mask(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "image_name": image_path.name, "selected": True}
    ]

    artifacts, warnings = run_optional_segmentation(
        tmp_path,
        "run_test",
        keyframes,
        None,
        provider=lambda _: np.zeros((10, 10), np.uint8),
        reconstruction_target="PRIMARY_SUBJECT",
        masking_mode="AUTO",
    )

    assert [path.name for path in artifacts] == ["segmentation_comparison.json"]
    assert warnings[0]["code"] == "REQUIRED_SEGMENTATION_FAILED"
