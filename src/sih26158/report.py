from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ConfidenceLabel, MatcherMetrics, RunRecord, utc_now


def known_distance_metrics(reference_m: float | None, measured_m: float | None) -> dict[str, Any]:
    if reference_m is None or measured_m is None:
        return {
            "status": "NOT_PROVIDED",
            "reference_m": reference_m,
            "measured_m": measured_m,
            "absolute_error_m": None,
            "percent_error": None,
            "passes_10_percent_gate": None,
        }
    absolute = abs(measured_m - reference_m)
    percent = 100 * absolute / reference_m
    return {
        "status": "MEASURED",
        "reference_m": reference_m,
        "measured_m": measured_m,
        "absolute_error_m": absolute,
        "signed_error_m": measured_m - reference_m,
        "percent_error": percent,
        "passes_10_percent_gate": percent <= 10,
    }


def build_quality_report(
    record: RunRecord,
    metrics: MatcherMetrics,
    warnings: list[dict[str, str]],
    *,
    synthetic: bool,
    alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registration_rate = metrics.registration_rate
    alignment = alignment or {}
    residuals = [float(value) for value in alignment.get("residuals_m", [])]
    inlier_count = int(alignment.get("inlier_count", 0))
    sorted_residuals = sorted(residuals)
    median_residual = (
        sorted_residuals[len(sorted_residuals) // 2] if sorted_residuals else None
    )
    p95_residual = (
        sorted_residuals[min(len(sorted_residuals) - 1, int(0.95 * len(sorted_residuals)))]
        if sorted_residuals
        else None
    )
    return {
        "schema_version": "1.0",
        "project_id": record.project_id,
        "run_id": record.run_id,
        "stage": "REPORTING",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "config_version": record.config_version,
        "run_configuration": record.config.model_dump(mode="json"),
        "synthetic_fixture": synthetic,
        "metrics": {
            "eligible_frames": metrics.eligible_frames,
            "registered_frames": metrics.registered_frames,
            "registered_frame_rate": registration_rate,
            "registered_frame_gate_80_percent": registration_rate >= 0.8,
            "median_reprojection_error_px": metrics.median_reprojection_error_px,
            "p95_reprojection_error_px": metrics.p95_reprojection_error_px,
            "reprojection_gate_1_5_px": metrics.median_reprojection_error_px <= 1.5,
            "runtime_s": metrics.runtime_s,
            "metric_alignment": {
                "scale": alignment.get("scale"),
                "origin_wgs84": alignment.get("origin_wgs84"),
                "camera_pairs": len(residuals),
                "inlier_count": inlier_count,
                "outlier_count": max(0, len(residuals) - inlier_count),
                "median_camera_prior_residual_m": median_residual,
                "p95_camera_prior_residual_m": p95_residual,
            },
            "known_distance": known_distance_metrics(
                record.config.known_distance_m, record.config.measured_distance_m
            ),
            "coverage": {"status": "NOT_EVALUATED", "reason": "Reference visible-region mask not supplied."},
        },
        "confidence_contract": [label.value for label in ConfidenceLabel],
        "warnings": warnings,
        "limitations": [
            "Geometry is defensible only where observed by multiple source frames.",
            "Ordinary GNSS is a soft alignment prior and is not survey-grade ground truth.",
            "AI-assisted geometry is excluded from verified measurement.",
            "Unseen surfaces are not reconstructed or claimed as measured.",
        ],
    }


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
