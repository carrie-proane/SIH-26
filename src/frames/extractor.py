"""Extract consistently oriented video frames and record their source timestamps."""

from __future__ import annotations

import csv
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class ExtractedFrame:
    """One decoded frame retained from the source video."""

    frame_index: int
    timestamp_s: float
    source_video: str
    frame_path: str


@dataclass(frozen=True)
class ExtractionResult:
    """Metadata and retained frames from one extraction pass."""

    frames: list[ExtractedFrame]
    fps: float
    source_frame_count: int
    rotation_degrees: int
    warnings: list[str]


def detect_rotation(video_path: str | Path) -> int:
    """Read display rotation metadata with ffprobe, returning a clockwise angle."""

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", str(video_path),
    ]
    try:
        payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        warnings.warn("Video rotation metadata unavailable; frames retain decoded orientation.")
        return 0
    stream = next(iter(payload.get("streams", [])), {})
    values = [stream.get("tags", {}).get("rotate")]
    values.extend(item.get("rotation") for item in stream.get("side_data_list", []))
    for value in values:
        if value is not None:
            try:
                return int(round(float(value))) % 360
            except (TypeError, ValueError):
                continue
    return 0


def _rotate(frame, degrees: int):
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_nth: int | None = None,
    target_fps: float | None = None,
) -> ExtractionResult:
    """Decode a video with OpenCV and retain frames at a deterministic interval."""

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if every_nth is not None and every_nth < 1:
        raise ValueError("every_nth must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    # OpenCV may otherwise apply Display Matrix rotation itself. Disable that
    # behavior so metadata rotation is handled exactly once below.
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    messages: list[str] = []
    if fps <= 0:
        capture.release()
        raise ValueError("Video reports an invalid frame rate")
    duration_s = source_count / fps if source_count > 0 else 0.0
    # The contract targets ~2 fps for ordinary 30-60 s passes, but the
    # 16.16 s primary needs ~6 fps (every fifth source frame at 30 fps) so
    # enough candidates exist for a 60-120-frame selection.
    effective_target_fps = target_fps or (6.0 if duration_s and duration_s < 20.0 else 2.0)
    if effective_target_fps <= 0:
        capture.release()
        raise ValueError("target_fps must be positive")
    interval = every_nth or max(1, round(fps / effective_target_fps))
    rotation = detect_rotation(video_path)
    retained: list[ExtractedFrame] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % interval == 0:
            destination = frames_dir / f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(destination), _rotate(frame, rotation)):
                capture.release()
                raise OSError(f"Could not write extracted frame: {destination}")
            retained.append(ExtractedFrame(index, index / fps, str(video_path), str(destination)))
        index += 1
    capture.release()
    if not retained:
        messages.append("NO_FRAMES_EXTRACTED")
        warnings.warn("No frames were decoded from the video.")
    index_path = output_dir / "frame_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_s", "source_video"])
        writer.writerows((f.frame_index, f"{f.timestamp_s:.6f}", f.source_video) for f in retained)
    return ExtractionResult(retained, fps, source_count, rotation, messages)
