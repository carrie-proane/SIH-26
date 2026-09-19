"""Contract and behavior tests for the Day-2 frame pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from frames.contact_sheet import create_contact_sheet
from frames.extractor import detect_rotation, extract_frames
from frames.scoring import blur_scores, exposure_score, redundancy_scores
from frames.selector import FRAME_SCORE_COLUMNS, SelectionWeights, select_indices, select_keyframes


def make_video(path: Path, frame_count: int = 30, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 96))
    assert writer.isOpened()
    for index in range(frame_count):
        image = np.full((96, 160, 3), 35, np.uint8)
        cv2.rectangle(image, (index * 4 % 125, 18), (index * 4 % 125 + 35, 76), (210, 210, 210), -1)
        cv2.putText(image, str(index), (55, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(image)
    writer.release()


def test_extraction_produces_expected_count_and_index(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    make_video(video)
    result = extract_frames(video, tmp_path / "run", every_nth=3)
    assert len(result.frames) == 10
    assert result.frames[1].frame_index == 3
    assert result.frames[1].timestamp_s == 0.3
    with (tmp_path / "run/frame_index.csv").open(newline="") as handle:
        assert next(csv.reader(handle)) == ["frame_index", "timestamp_s", "source_video"]


def test_negative_90_display_matrix_is_applied_once(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "rotated.avi"
    make_video(video, frame_count=1)
    monkeypatch.setattr("frames.extractor.detect_rotation", lambda _: 270)
    result = extract_frames(video, tmp_path / "run", every_nth=1)
    image = cv2.imread(result.frames[0].frame_path)
    assert image.shape[:2] == (160, 96)


def test_rotation_detector_reads_negative_display_matrix(monkeypatch) -> None:
    completed = subprocess.CompletedProcess([], 0, '{"streams":[{"side_data_list":[{"rotation":-90}]}]}', "")
    monkeypatch.setattr("frames.extractor.subprocess.run", lambda *args, **kwargs: completed)
    assert detect_rotation("portrait.mp4") == 270


def test_blur_score_prefers_sharp_image() -> None:
    sharp = np.zeros((120, 120, 3), np.uint8)
    sharp[::4, :] = 255
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    scores = blur_scores([sharp, blurred])
    assert scores[0] > scores[1]


def test_exposure_penalizes_blown_and_crushed_images() -> None:
    well_exposed = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    well_exposed = cv2.cvtColor(well_exposed, cv2.COLOR_GRAY2BGR)
    assert exposure_score(well_exposed) > exposure_score(np.full_like(well_exposed, 255))
    assert exposure_score(well_exposed) > exposure_score(np.zeros_like(well_exposed))


def test_redundancy_marks_duplicate_as_low_value() -> None:
    first = np.full((80, 80, 3), 60, np.uint8)
    duplicate = first.copy()
    different = np.zeros_like(first)
    different[:, :40] = 255
    scores = redundancy_scores([first, duplicate, different])
    assert scores[1] < scores[2]


def test_selector_respects_minimum_temporal_spacing() -> None:
    chosen = select_indices([0.9, 1.0, 0.8, 0.7], [0.0, 0.2, 1.0, 2.0], 3, 0.75)
    times = [[0.0, 0.2, 1.0, 2.0][index] for index in chosen]
    assert all(abs(a - b) >= 0.75 for position, a in enumerate(times) for b in times[position + 1:])


def test_pipeline_artifacts_match_contract_and_contact_sheet_is_valid(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    run = tmp_path / "run"
    make_video(video, frame_count=20)
    extraction = extract_frames(video, run, every_nth=2)
    rows = select_keyframes(extraction.frames, run, target_frames=4, weights=SelectionWeights(), min_spacing_s=0.3)
    create_contact_sheet(rows, run / "contact_sheet.png", columns=3, thumbnail_width=120)
    with (run / "frame_scores.csv").open(newline="") as handle:
        assert next(csv.reader(handle)) == FRAME_SCORE_COLUMNS
    payload = json.loads((run / "keyframes.json").read_text())
    assert payload["schema_version"] == "1.0"
    assert sum(frame["selected"] for frame in payload["frames"]) == 4
    assert all(set(FRAME_SCORE_COLUMNS + ["image_name", "path"]) <= set(frame) for frame in payload["frames"])
    assert cv2.imread(str(run / "contact_sheet.png")) is not None


def test_weights_must_sum_to_one() -> None:
    try:
        SelectionWeights(0.5, 0.5, 0.5)
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("invalid weights were accepted")


def test_overrides_are_recorded_and_preserve_minimum_selection(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    run = tmp_path / "run"
    make_video(video, frame_count=12)
    extraction = extract_frames(video, run, every_nth=1)
    rows = select_keyframes(
        extraction.frames,
        run,
        target_frames=4,
        force_include={11},
        force_exclude={0},
    )
    decisions = {row["frame_index"]: row for row in rows}
    assert decisions[11]["override"] == "FORCE_INCLUDE"
    assert decisions[11]["selected"] is True
    assert decisions[0]["override"] == "FORCE_EXCLUDE"
    assert decisions[0]["selected"] is False
    assert sum(bool(row["selected"]) for row in rows) >= 3


def test_streaming_scores_preserve_original_formulas(tmp_path: Path) -> None:
    from frames.scoring import load_images, stream_frame_scores
    video = tmp_path / "scoring.avi"
    make_video(video, frame_count=10)
    frames = extract_frames(video, tmp_path / "frames", every_nth=1).frames
    paths = [frame.frame_path for frame in frames]
    images = load_images(paths)
    actual = stream_frame_scores(paths, None)
    expected = (blur_scores(images), [exposure_score(image) for image in images], redundancy_scores(images))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b)


def test_only_selected_previews_are_restored_at_full_resolution(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "native.avi"
    make_video(video, frame_count=12)
    monkeypatch.setattr("frames.extractor.detect_rotation", lambda _: 270)
    frames = extract_frames(video, tmp_path / "frames", every_nth=1, preview_max_dimension=80).frames
    assert all(max(cv2.imread(frame.frame_path).shape[:2]) == 80 for frame in frames)
    rows = select_keyframes(frames, tmp_path / "selected", target_frames=4)
    for row in rows:
        shape = cv2.imread(str(row["frame_path"])).shape[:2]
        assert shape == ((160, 96) if row["selected"] else (80, 48))


def test_scoring_holds_only_adjacent_bounded_images(tmp_path: Path, monkeypatch) -> None:
    import weakref

    from frames import scoring
    video = tmp_path / "stream.avi"
    make_video(video, frame_count=12)
    frames = extract_frames(video, tmp_path / "frames", every_nth=1).frames
    original_read, original_ssim = scoring.cv2.imread, scoring._ssim
    live_images = []
    compared_shapes = []

    def read(path):
        image = original_read(path)
        live_images.append(weakref.ref(image))
        assert sum(ref() is not None for ref in live_images) <= 2
        return image

    def compare(first, second):
        compared_shapes.extend([first.shape, second.shape])
        return original_ssim(first, second)

    monkeypatch.setattr(scoring.cv2, "imread", read)
    monkeypatch.setattr(scoring, "_ssim", compare)
    scoring.stream_frame_scores([frame.frame_path for frame in frames], max_dimension=80)
    assert compared_shapes and all(max(shape[:2]) <= 80 for shape in compared_shapes)


def test_real_video_preview_selection_and_peak_memory(tmp_path: Path) -> None:
    import sys

    import pytest

    from frames.scoring import SCORING_MAX_DIMENSION
    root = Path(__file__).parents[1]
    video = root / "DJI_0574.MP4"
    if not video.is_file():
        pytest.skip("Real-video resource acceptance requires local DJI_0574.MP4; no fixture is fabricated")
    report_path = tmp_path / "benchmark.json"
    subprocess.run([sys.executable, str(root / "scripts/benchmark_frame_previews.py"),
        "--video", str(video), "--dimension", str(SCORING_MAX_DIMENSION), "--output", str(report_path)],
        check=True, capture_output=True, text=True, timeout=240)
    report = json.loads(report_path.read_text())
    assert report["exact_selection_overlap"] >= 0.9
    assert report["peak_rss_reduction"] >= 0.2
    assert report["preview"]["candidate_count"] == report["baseline"]["candidate_count"]
    assert set(map(tuple, report["preview"]["selected_shapes"])) == {(2160, 3840)}
