"""Rank frames by reconstruction value and select a temporally distributed subset."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .extractor import ExtractedFrame
from .scoring import (
    exposure_scores,
    laplacian_variances,
    load_images,
    normalize_scores,
    redundancy_scores,
)

FRAME_SCORE_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "source_video",
    "frame_path",
    "blur_score",
    "exposure_score",
    "redundancy_score",
    "composite_score",
    "laplacian_variance",
    "quality_eligible",
    "quality_rejection_reasons",
    "selected_automatically",
    "override",
    "selected",
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


@dataclass(frozen=True)
class FrameQualityThresholds:
    """Dataset-independent floors plus a bounded dataset-relative sharpness floor."""

    min_laplacian_variance: float = 40.0
    min_exposure_score: float = 0.18
    relative_sharpness_floor: float = 0.60

    def __post_init__(self) -> None:
        if self.min_laplacian_variance < 0:
            raise ValueError("min_laplacian_variance cannot be negative")
        if not 0 <= self.min_exposure_score <= 1:
            raise ValueError("min_exposure_score must be between 0 and 1")
        if not 0 <= self.relative_sharpness_floor <= 1:
            raise ValueError("relative_sharpness_floor must be between 0 and 1")


def select_indices(
    scores: Sequence[float], timestamps: Sequence[float], target: int, min_spacing_s: float
) -> set[int]:
    """Greedily select highest scores while enforcing minimum timestamp spacing."""

    if target < 0 or min_spacing_s < 0:
        raise ValueError("target and min_spacing_s must be non-negative")
    selected: set[int] = set()
    for candidate in sorted(range(len(scores)), key=lambda i: (-scores[i], timestamps[i], i)):
        if len(selected) >= min(target, len(scores)):
            break
        if all(
            abs(timestamps[candidate] - timestamps[chosen]) + 1e-9 >= min_spacing_s
            for chosen in selected
        ):
            selected.add(candidate)
    return selected


def select_keyframes(
    frames: Sequence[ExtractedFrame],
    output_dir: str | Path,
    target_frames: int = 80,
    weights: SelectionWeights | None = None,
    min_spacing_s: float | None = None,
    force_include: set[int] | None = None,
    force_exclude: set[int] | None = None,
    quality_thresholds: FrameQualityThresholds | None = None,
) -> list[dict[str, object]]:
    """Score all extracted frames and write the CSV/JSON handoff artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or SelectionWeights()
    quality_thresholds = quality_thresholds or FrameQualityThresholds()
    force_include = force_include or set()
    force_exclude = force_exclude or set()
    if force_include & force_exclude:
        raise ValueError("A frame cannot be both force-included and force-excluded")
    known_indices = {frame.frame_index for frame in frames}
    unknown = (force_include | force_exclude) - known_indices
    if unknown:
        raise ValueError(f"Frame overrides reference unknown indices: {sorted(unknown)}")
    images = load_images([frame.frame_path for frame in frames])
    sharpness = laplacian_variances(images)
    blur = normalize_scores(sharpness)
    exposure = exposure_scores(images)
    redundancy = redundancy_scores(images)
    composite = [
        weights.blur * b + weights.exposure * e + weights.redundancy * r
        for b, e, r in zip(blur, exposure, redundancy)
    ]
    timestamps = [frame.timestamp_s for frame in frames]
    if min_spacing_s is None:
        duration = max(timestamps, default=0.0) - min(timestamps, default=0.0)
        min_spacing_s = duration / max(1, target_frames - 1) * 0.8 if target_frames > 1 else 0.0
    median_sharpness = float(np.median(sharpness)) if sharpness else 0.0
    effective_sharpness_floor = max(
        quality_thresholds.min_laplacian_variance,
        median_sharpness * quality_thresholds.relative_sharpness_floor,
    )
    rejection_reasons: list[list[str]] = []
    for sharpness_value, exposure_value in zip(sharpness, exposure, strict=True):
        reasons: list[str] = []
        if sharpness_value < effective_sharpness_floor:
            reasons.append("BELOW_SHARPNESS_GATE")
        if exposure_value < quality_thresholds.min_exposure_score:
            reasons.append("BELOW_EXPOSURE_GATE")
        rejection_reasons.append(reasons)
    eligible_positions = [index for index, reasons in enumerate(rejection_reasons) if not reasons]
    eligible_scores = [composite[index] for index in eligible_positions]
    eligible_timestamps = [timestamps[index] for index in eligible_positions]
    chosen_eligible = select_indices(
        eligible_scores,
        eligible_timestamps,
        min(target_frames, len(eligible_positions)),
        min_spacing_s,
    )
    selected_positions = {eligible_positions[index] for index in chosen_eligible}
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
        rows.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_s": round(frame.timestamp_s, 6),
                "source_video": frame.source_video,
                "frame_path": frame.frame_path,
                "blur_score": round(blur[position], 6),
                "exposure_score": round(exposure[position], 6),
                "redundancy_score": round(redundancy[position], 6),
                "composite_score": round(composite[position], 6),
                "laplacian_variance": round(sharpness[position], 6),
                "quality_eligible": not rejection_reasons[position],
                "quality_rejection_reasons": ";".join(rejection_reasons[position]),
                "selected_automatically": selected_automatically,
                "override": override,
                "selected": selected,
            }
        )
    if sum(bool(row["selected"]) for row in rows) < 3:
        raise ValueError(
            "Fewer than three frames passed the sharpness/exposure gates; "
            "capture clearer footage or explicitly review frame overrides"
        )
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
        "schema_version": "1.0",
        "source_video": frames[0].source_video if frames else None,
        "selection": {
            "target_frames": target_frames,
            "min_spacing_s": round(min_spacing_s, 6),
            "weights": asdict(weights),
            "quality_thresholds": asdict(quality_thresholds),
            "effective_sharpness_floor": round(effective_sharpness_floor, 6),
            "quality_eligible_frames": len(eligible_positions),
        },
        "frames": decisions,
    }
    (output_dir / "keyframes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return rows
