"""Extract consistently oriented video frames and record their source timestamps."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractedFrame:
    """One decoded frame retained from the source video."""

    frame_index: int
    timestamp_s: float
    source_video: str
    frame_path: str
    source_width: int | None = None
    source_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    resize_scale: float = 1.0


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
                return round(float(value)) % 360
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
    frames_subdir: str = "frames",
    max_candidates: int = 240,
    max_image_dimension: int | None = None,
) -> ExtractionResult:
    """Decode a video with OpenCV and retain frames at a deterministic interval."""

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir)
    frames_dir = output_dir / frames_subdir
    frames_dir.mkdir(parents=True, exist_ok=True)
    if every_nth is not None and every_nth < 1:
        raise ValueError("every_nth must be at least 1")
    if max_candidates < 3:
        raise ValueError("max_candidates must be at least 3")
    if max_image_dimension is not None and max_image_dimension < 1:
        raise ValueError("max_image_dimension must be positive")
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
    # Sample uniformly across the complete duration and bound retained images.
    # Decoding remains streaming: only the current frame and retained JPEGs exist.
    adaptive_fps = max_candidates / duration_s if duration_s > 0 else 2.0
    effective_target_fps = target_fps or min(8.0, max(0.1, adaptive_fps))
    if effective_target_fps <= 0:
        capture.release()
        raise ValueError("target_fps must be positive")
    interval = every_nth or max(1, math.ceil(fps / effective_target_fps))
    if every_nth is None and source_count > 0:
        interval = max(interval, math.ceil(source_count / max_candidates))
    rotation = detect_rotation(video_path)
    retained: list[ExtractedFrame] = []
    sampled_count = 0
    reservoir_rng = np.random.default_rng(26158)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % interval == 0:
            sampled_count += 1
            if len(retained) < max_candidates:
                retained_slot = len(retained)
            else:
                retained_slot = int(reservoir_rng.integers(0, sampled_count))
                if retained_slot >= max_candidates:
                    index += 1
                    continue
            destination = frames_dir / f"frame_{index:06d}.jpg"
            source_height, source_width = frame.shape[:2]
            output_frame = _rotate(frame, rotation)
            output_height, output_width = output_frame.shape[:2]
            resize_scale = 1.0
            if max_image_dimension is not None and max(output_height, output_width) > max_image_dimension:
                resize_scale = max_image_dimension / max(output_height, output_width)
                output_width = max(1, round(output_width * resize_scale))
                output_height = max(1, round(output_height * resize_scale))
                output_frame = cv2.resize(
                    output_frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            if not cv2.imwrite(str(destination), output_frame):
                capture.release()
                raise OSError(f"Could not write extracted frame: {destination}")
            decoded_timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(decoded_timestamp_s) or decoded_timestamp_s < 0:
                decoded_timestamp_s = index / fps
                if "DECODED_TIMESTAMP_UNAVAILABLE" not in messages:
                    messages.append("DECODED_TIMESTAMP_UNAVAILABLE")
            extracted = ExtractedFrame(
                index,
                round(decoded_timestamp_s, 9),
                video_path.name,
                str(destination),
                source_width,
                source_height,
                output_width,
                output_height,
                resize_scale,
            )
            if retained_slot == len(retained):
                retained.append(extracted)
            else:
                Path(retained[retained_slot].frame_path).unlink(missing_ok=True)
                retained[retained_slot] = extracted
        index += 1
    capture.release()
    retained.sort(key=lambda item: item.frame_index)
    if sampled_count > max_candidates:
        messages.append("CANDIDATE_RESERVOIR_BOUND_APPLIED")
    if not retained:
        messages.append("NO_FRAMES_EXTRACTED")
        warnings.warn("No frames were decoded from the video.")
    index_path = output_dir / "frame_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_s",
                "source_video",
                "source_width",
                "source_height",
                "output_width",
                "output_height",
                "resize_scale",
                "rotation_degrees",
            ]
        )
        writer.writerows(
            (
                f.frame_index,
                f"{f.timestamp_s:.6f}",
                f.source_video,
                f.source_width,
                f.source_height,
                f.output_width,
                f.output_height,
                f"{f.resize_scale:.9f}",
                rotation,
            )
            for f in retained
        )
    return ExtractionResult(retained, fps, source_count, rotation, messages)
