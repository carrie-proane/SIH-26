"""Rank frames by reconstruction value and select a temporally distributed subset."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .extractor import ExtractedFrame
from .scoring import blur_scores, exposure_scores, load_images, redundancy_scores

FRAME_SCORE_COLUMNS = [
    "frame_index", "timestamp_s", "source_video", "frame_path", "blur_score",
    "exposure_score", "redundancy_score", "composite_score", "selected_automatically",
    "override", "selected",
]


@dataclass(frozen=True)
class SelectionWeights:
    blur: float = 0.4
    exposure: float = 0.3
    redundancy: float = 0.3

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Selection weights cannot be negative")
        if abs(sum(asdict(self).values()) - 1.0) > 1e-6:
            raise ValueError("Selection weights must sum to 1")


def select_indices(scores: Sequence[float], timestamps: Sequence[float], target: int, min_spacing_s: float) -> set[int]:
    """Greedily select highest scores while enforcing minimum timestamp spacing."""

    if target < 0 or min_spacing_s < 0:
        raise ValueError("target and min_spacing_s must be non-negative")
    selected: set[int] = set()
    for candidate in sorted(range(len(scores)), key=lambda i: (-scores[i], timestamps[i], i)):
        if len(selected) >= min(target, len(scores)):
            break
        if all(abs(timestamps[candidate] - timestamps[chosen]) + 1e-9 >= min_spacing_s for chosen in selected):
            selected.add(candidate)
    return selected


def select_keyframes(
    frames: Sequence[ExtractedFrame], output_dir: str | Path, target_frames: int = 80,
    weights: SelectionWeights | None = None, min_spacing_s: float | None = None,
    force_include: set[int] | None = None, force_exclude: set[int] | None = None,
) -> list[dict[str, object]]:
    """Score all extracted frames and write the CSV/JSON handoff artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or SelectionWeights()
    force_include = force_include or set()
    force_exclude = force_exclude or set()
    if force_include & force_exclude:
        raise ValueError("A frame cannot be both force-included and force-excluded")
    known_indices = {frame.frame_index for frame in frames}
    unknown = (force_include | force_exclude) - known_indices
    if unknown:
        raise ValueError(f"Frame overrides reference unknown indices: {sorted(unknown)}")
    images = load_images([frame.frame_path for frame in frames])
    blur = blur_scores(images)
    exposure = exposure_scores(images)
    redundancy = redundancy_scores(images)
    composite = [weights.blur * b + weights.exposure * e + weights.redundancy * r for b, e, r in zip(blur, exposure, redundancy)]
    timestamps = [frame.timestamp_s for frame in frames]
    if min_spacing_s is None:
        duration = max(timestamps, default=0.0) - min(timestamps, default=0.0)
        min_spacing_s = duration / max(1, target_frames - 1) * 0.8 if target_frames > 1 else 0.0
    selected_positions = select_indices(composite, timestamps, target_frames, min_spacing_s)
    rows: list[dict[str, object]] = []
    for position, frame in enumerate(frames):
        selected_automatically = position in selected_positions
        if frame.frame_index in force_include:
            override = "FORCE_INCLUDE"
            selected = True
        elif frame.frame_index in force_exclude:
            override = "FORCE_EXCLUDE"
            selected = False
        else:
            override = "NONE"
            selected = selected_automatically
        rows.append({
            "frame_index": frame.frame_index, "timestamp_s": round(frame.timestamp_s, 6),
            "source_video": frame.source_video, "frame_path": frame.frame_path,
            "blur_score": round(blur[position], 6), "exposure_score": round(exposure[position], 6),
            "redundancy_score": round(redundancy[position], 6), "composite_score": round(composite[position], 6),
            "selected_automatically": selected_automatically,
            "override": override,
            "selected": selected,
        })
    if sum(bool(row["selected"]) for row in rows) < 3:
        raise ValueError("Frame overrides must preserve at least three selected images")
    with (output_dir / "frame_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    decisions = [
        {
            **row,
            "image_name": Path(str(row["frame_path"])).name,
            "path": row["frame_path"],
        }
        for row in rows
    ]
    payload = {
        "schema_version": "1.0", "source_video": frames[0].source_video if frames else None,
        "selection": {"target_frames": target_frames, "min_spacing_s": round(min_spacing_s, 6), "weights": asdict(weights)},
        "frames": decisions,
    }
    (output_dir / "keyframes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return rows
