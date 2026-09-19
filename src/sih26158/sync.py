from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geo import SimilarityTransform, robust_similarity
from .models import OffsetSource

# Variance ratio 100 means at least a 10:1 spread in principal directions.
# Use the two largest 3D eigenvalues so a planar orbit remains identifiable.
MAX_TRAJECTORY_SPREAD_RATIO = 100.0


def trajectory_identifiability(points: NDArray[np.float64]) -> dict[str, object]:
    points = np.asarray(points, dtype=float)
    centered = points - points.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(points)), 0)
    largest, second = float(eigenvalues[-1]), float(eigenvalues[-2])
    ratio = largest / second if second > np.finfo(float).eps else None
    degenerate = ratio is None or ratio > MAX_TRAJECTORY_SPREAD_RATIO
    return {
        "alignment_identifiability": "degenerate" if degenerate else "well_conditioned",
        "trajectory_spread_ratio": ratio,
        "trajectory_spread_ratio_threshold": MAX_TRAJECTORY_SPREAD_RATIO,
        "trajectory_spread_eigenvalues": eigenvalues.tolist(),
    }


@dataclass(frozen=True)
class OffsetEvaluation:
    offset_s: float
    rmse_m: float
    transform: SimilarityTransform
    matched_indices: NDArray[np.int64]
    metric_points: NDArray[np.float64]

    @property
    def identifiability(self) -> dict[str, object]:
        return trajectory_identifiability(self.metric_points[self.transform.inliers])

    @property
    def inlier_count(self) -> int:
        return int(self.transform.inliers.sum())


@dataclass(frozen=True)
class OffsetCalibration:
    selected: OffsetEvaluation
    before: OffsetEvaluation | None
    source: OffsetSource
    searched_offsets: tuple[tuple[float, float, int], ...]

    def as_report(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            **self.selected.identifiability,
            "residual_label": "camera-to-telemetry consistency metric",
            "telemetry_offset_s": self.selected.offset_s,
            "offset_source": self.source.value,
            "rmse_before_m": self.before.rmse_m if self.before is not None else None,
            "rmse_after_m": self.selected.rmse_m,
            "matched_camera_count": len(self.selected.matched_indices),
            "inlier_count": self.selected.inlier_count,
            "search_candidates": [
                {"offset_s": offset, "robust_rmse_m": rmse, "inlier_count": inliers}
                for offset, rmse, inliers in self.searched_offsets
            ],
        }


def _evaluate_offset(
    sfm_points: NDArray[np.float64],
    frame_times: NDArray[np.float64],
    telemetry_times: NDArray[np.float64],
    telemetry_points: NDArray[np.float64],
    offset_s: float,
) -> OffsetEvaluation:
    shifted = frame_times + offset_s
    matched_indices = np.flatnonzero(
        (shifted >= telemetry_times[0]) & (shifted <= telemetry_times[-1])
    ).astype(np.int64)
    if len(matched_indices) < 3:
        raise ValueError("Fewer than three camera timestamps overlap telemetry at this offset")
    matched_times = shifted[matched_indices]
    metric_points = np.column_stack(
        [
            np.interp(matched_times, telemetry_times, telemetry_points[:, axis])
            for axis in range(3)
        ]
    )
    transform = robust_similarity(sfm_points[matched_indices], metric_points)
    residuals = transform.residuals_m[transform.inliers]
    if not len(residuals):
        raise ValueError("Offset alignment produced no robust inliers")
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    return OffsetEvaluation(
        offset_s=float(offset_s),
        rmse_m=rmse,
        transform=transform,
        matched_indices=matched_indices,
        metric_points=metric_points,
    )


def calibrate_telemetry_offset(
    sfm_points: NDArray[np.float64],
    frame_times: NDArray[np.float64],
    telemetry_times: NDArray[np.float64],
    telemetry_points: NDArray[np.float64],
    *,
    manual_offset_s: float | None = None,
    manual_source: str | None = None,
    search_min_s: float = -1.0,
    search_max_s: float = 1.0,
    search_step_s: float = 0.05,
) -> OffsetCalibration:
    """Calibrate one run only; no dataset-specific offset is encoded here."""
    sfm_points = np.asarray(sfm_points, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    telemetry_times = np.asarray(telemetry_times, dtype=float)
    telemetry_points = np.asarray(telemetry_points, dtype=float)
    if sfm_points.shape != (len(frame_times), 3):
        raise ValueError("Camera centres and frame timestamps must have matching lengths")
    if telemetry_points.shape != (len(telemetry_times), 3):
        raise ValueError("Telemetry positions and timestamps must have matching lengths")
    if len(sfm_points) < 3 or len(telemetry_points) < 3:
        raise ValueError("At least three camera and telemetry positions are required")
    if np.any(np.diff(telemetry_times) <= 0):
        raise ValueError("Telemetry timestamps must be strictly increasing")

    try:
        before = _evaluate_offset(
            sfm_points, frame_times, telemetry_times, telemetry_points, 0.0
        )
    except ValueError:
        # An unavailable baseline must not prevent a valid shifted fit.
        before = None
    if manual_offset_s is not None:
        selected = _evaluate_offset(
            sfm_points,
            frame_times,
            telemetry_times,
            telemetry_points,
            manual_offset_s,
        )
        source = OffsetSource(manual_source or OffsetSource.MANUAL.value)
        return OffsetCalibration(
            selected=selected,
            before=before,
            source=source,
            searched_offsets=((selected.offset_s, selected.rmse_m, selected.inlier_count),),
        )

    if search_step_s <= 0 or search_min_s > search_max_s:
        raise ValueError("Invalid bounded telemetry offset search")
    count = round((search_max_s - search_min_s) / search_step_s)
    offsets = [round(search_min_s + index * search_step_s, 10) for index in range(count + 1)]
    evaluations: list[OffsetEvaluation] = []
    for offset in offsets:
        try:
            evaluations.append(
                _evaluate_offset(
                    sfm_points,
                    frame_times,
                    telemetry_times,
                    telemetry_points,
                    offset,
                )
            )
        except ValueError:
            continue
    if not evaluations:
        raise ValueError("No bounded telemetry offset candidate produced a valid alignment")
    selected = min(
        evaluations,
        key=lambda item: (
            item.rmse_m,
            -item.inlier_count,
            abs(item.offset_s),
            item.offset_s,
        ),
    )
    return OffsetCalibration(
        selected=selected,
        before=before,
        source=OffsetSource.AUTOMATIC,
        searched_offsets=tuple(
            (item.offset_s, item.rmse_m, item.inlier_count) for item in evaluations
        ),
    )
