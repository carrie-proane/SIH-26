import json
from pathlib import Path

import cv2
import numpy as np

from sih26158.scene_policy import analyze_scene


def _frame(run_dir: Path, name: str, image: np.ndarray) -> dict[str, object]:
    frames = run_dir / "frames"
    frames.mkdir(exist_ok=True)
    cv2.imwrite(str(frames / name), image)
    return {"image_name": name, "selected": True}


def test_primary_subject_analysis_requires_explicit_subject_mask(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "frame.jpg", np.full((100, 160, 3), 100, np.uint8))
    path, report, warnings = analyze_scene(
        tmp_path,
        [frame],
        reconstruction_target="PRIMARY_SUBJECT",
        masking_mode="AUTO",
    )

    assert path.is_file()
    assert report["recommendation"] == "SUBJECT_MASK_REQUIRED"
    assert warnings[0]["code"] == "SCENE_MASK_RECOMMENDED"


def test_extreme_sky_candidate_blocks_dense_without_mask(tmp_path: Path) -> None:
    blue_sky = np.full((100, 160, 3), (255, 160, 80), np.uint8)
    frame = _frame(tmp_path, "sky.jpg", blue_sky)
    path, report, _ = analyze_scene(
        tmp_path,
        [frame],
        reconstruction_target="FULL_SCENE",
        masking_mode="AUTO",
    )

    persisted = json.loads(path.read_text())
    assert report["dense_suitability"] == "BLOCKED_WITHOUT_MASK"
    assert persisted["recommendation"] == "MASK_RECOMMENDED"
