from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .confidence import confidence_contract
from .models import EvidenceVerdict, MatcherMetrics, ProvenanceOrigin, RunRecord, utc_now


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


def reconstructed_known_distance(record: RunRecord, run_dir: Path | None) -> dict[str, Any]:
    """Read endpoints from this run's declared metric sparse cloud; never trust typed distances."""
    config = record.config
    endpoints = [config.known_distance_endpoint_a, config.known_distance_endpoint_b]
    measured = None
    computed_at = None
    cloud_path = "sparse/sparse_local.ply"
    cloud = next((a for a in record.artifacts if a.relative_path == cloud_path), None)
    coords = None
    note = "No independent reference distance supplied."
    if config.known_distance_m is not None:
        note = "Reconstructed distance unavailable: supply two PLY vertex endpoints for this run."
        if all(endpoint is not None for endpoint in endpoints):
            note = "This run has no available declared metric sparse point cloud."
            if run_dir is not None and cloud is not None:
                try:
                    ids = [endpoint.point_id for endpoint in endpoints]
                    if ids[0] == ids[1]:
                        raise ValueError("Reference endpoints must be distinct vertices")
                    coords = _read_sparse_endpoints(run_dir / cloud_path, ids)
                    measured = math.dist(*coords)
                    computed_at = utc_now()
                    note = "Computed from this run's metric sparse PLY vertices; telemetry determined scale independently of this reference."
                except (OSError, ValueError, UnicodeError) as exc:
                    note = f"Reconstructed distance unavailable: {exc}"
    result = known_distance_metrics(config.known_distance_m, measured)
    result.update({
        "reference_value_m": config.known_distance_m,
        "reference_source": config.known_distance_reference_source,
        "endpoint_a": endpoints[0].model_dump() if endpoints[0] else None,
        "endpoint_b": endpoints[1].model_dump() if endpoints[1] else None,
        "reconstructed_distance_m": measured,
        "computed_at": computed_at,
        "error_pct": result["percent_error"],
        "gate_threshold_pct": 10.0,
        "run_id": record.run_id,
        "cloud_artifact": cloud_path if measured is not None else None,
        "cloud_sha256": cloud.sha256 if measured is not None and cloud else None,
        "endpoint_coordinates_m": coords,
        "reported_input_measured_m": config.measured_distance_m,
        "note": note,
    })
    return result


def _read_sparse_endpoints(path: Path, point_ids: list[int]) -> list[list[float]]:
    # Our COLMAP sparse exporter writes ASCII. Unknown formats stay unvalidated.
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Not a PLY point cloud")
        count = 0
        names: list[str] = []
        in_vertices = False
        ascii_format = False
        for line in stream:
            parts = line.split()
            if parts == ["end_header"]:
                break
            if parts[:1] == ["format"]:
                ascii_format = parts[1] == "ascii"
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
                in_vertices = True
            elif parts[:1] == ["element"]:
                in_vertices = False
            if in_vertices and parts[:1] == ["property"]:
                if len(parts) != 3:
                    raise ValueError("Unsupported PLY vertex property")
                names.append(parts[2])
        else:
            raise ValueError("Missing PLY header terminator")
        if not ascii_format or not {"x", "y", "z"} <= set(names):
            raise ValueError("Expected an ASCII metric sparse cloud with xyz vertices")
        if max(point_ids) >= count:
            raise ValueError("Endpoint vertex ID is outside this run's point cloud")
        indices = [names.index(axis) for axis in ("x", "y", "z")]
        found = {}
        for index in range(max(point_ids) + 1):
            values = stream.readline().split()
            if index in point_ids:
                if len(values) != len(names):
                    raise ValueError("Truncated PLY vertex")
                point = [float(values[i]) for i in indices]
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("Non-finite endpoint coordinates")
                found[index] = point
        return [found[index] for index in point_ids]


def build_quality_report(
    record: RunRecord,
    metrics: MatcherMetrics,
    warnings: list[dict[str, str]],
    *,
    alignment: dict[str, Any] | None = None,
    confidence_available: bool = False,
    run_dir: Path | None = None,
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
    report_warnings = list(warnings)
    if not confidence_available:
        report_warnings.append(
            {
                "code": "CONFIDENCE_UNAVAILABLE",
                "message": (
                    "No valid explicit point-confidence artifact was declared; photographic RGB "
                    "must not be interpreted as verification confidence."
                ),
            }
        )
    genuine_real_evidence = record.source_provenance == ProvenanceOrigin.REAL
    known_distance = reconstructed_known_distance(record, run_dir)
    gates = {
        "registration": registration_rate >= 0.8,
        "reprojection": metrics.median_reprojection_error_px <= 1.5,
        "metric_alignment": (
            float(alignment["scale"]) > 0 and inlier_count >= 3
            if alignment.get("scale") is not None else None
        ),
        "known_distance": known_distance["passes_10_percent_gate"],
        "known_distance_provenance": True if (record.config.known_distance_reference_source or "").strip() else None,
        "alignment_identifiability": True if alignment.get("alignment_identifiability") == "well_conditioned" else None,
        "altitude_reference": True if alignment.get("vertical_alignment_verdict") == "PASSED" else None,
        "real_evidence": True if genuine_real_evidence and not alignment.get("synthetic_fixture") else None,
    }
    verdict = (
        EvidenceVerdict.FAILED if any(value is False for value in gates.values()) else
        EvidenceVerdict.NOT_VALIDATED if any(value is None for value in gates.values()) else
        EvidenceVerdict.PASSED
    )
    return {
        "schema_version": "1.0",
        "project_id": record.project_id,
        "run_id": record.run_id,
        "stage": "REPORTING",
        "status": "COMPLETED",
        "evidence_verdict": verdict,
        "evidence_gates": gates,
        "created_at": utc_now(),
        "config_version": record.config_version,
        "run_configuration": record.config.model_dump(mode="json"),
        "synthetic_fixture": record.source_provenance == ProvenanceOrigin.SYNTHETIC,
        "source_provenance": record.source_provenance,
        "video_origin": record.video_origin,
        "telemetry_origin": record.telemetry_origin,
        "genuine_real_evidence": genuine_real_evidence,
        "alignment_identifiability": alignment.get("alignment_identifiability", "not_validated"),
        "altitude_reference": alignment.get("altitude_reference", "unknown"),
        "altitude_reference_source": alignment.get("altitude_reference_source"),
        "altitude_reference_assumed": alignment.get("altitude_reference_assumed", True),
        "vertical_alignment_verdict": alignment.get("vertical_alignment_verdict", "NOT_VALIDATED"),
        "matcher_actually_used": metrics.matcher_actually_used,
        "matcher_fallback_reason": metrics.matcher_fallback_reason,
        "masking_applied": metrics.masking_applied,
        "masking_note": (
            None if metrics.masking_applied else
            "masks generated but not passed to COLMAP in this build"
            if any(a.relative_path == "segmentation_comparison.json" for a in record.artifacts)
            else "no masks passed to COLMAP"
        ),
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
                "residual_label": "camera-to-telemetry consistency metric",
                "alignment_identifiability": alignment.get("alignment_identifiability", "not_validated"),
                "trajectory_spread_ratio": alignment.get("trajectory_spread_ratio"),
                "trajectory_spread_ratio_threshold": alignment.get("trajectory_spread_ratio_threshold"),
                "scale": alignment.get("scale"),
                "origin_wgs84": alignment.get("origin_wgs84"),
                "camera_pairs": len(residuals),
                "inlier_count": inlier_count,
                "outlier_count": max(0, len(residuals) - inlier_count),
                "median_camera_prior_residual_m": median_residual,
                "p95_camera_prior_residual_m": p95_residual,
            },
            "telemetry_sync": {
                "residual_label": "camera-to-telemetry consistency metric",
                "telemetry_offset_s": record.telemetry_offset_s,
                "offset_source": record.offset_source,
                "rmse_before_m": record.rmse_before_m,
                "rmse_after_m": record.rmse_after_m,
                "matched_camera_count": record.matched_camera_count,
                "inlier_count": record.inlier_count,
            },
            "known_distance": known_distance,
            "coverage": {"status": "NOT_EVALUATED", "reason": "Reference visible-region mask not supplied."},
        },
        "confidence_artifact": {
            "available": confidence_available,
            "measurement_confidence_available": confidence_available,
            "reason": (
                None
                if confidence_available
                else "Confidence unavailable for this run"
            ),
            "contract": confidence_contract(),
        },
        "warnings": report_warnings,
        "limitations": [
            "Geometry is defensible only where observed by multiple source frames.",
            "Camera-to-telemetry consistency residuals measure fit to the supplied prior, not independent positional validation.",
            "Ordinary GNSS is a soft alignment prior and is not survey-grade ground truth.",
            "AI-assisted geometry is excluded from measurement evidence.",
            "Confidence-qualified measurement is unavailable without a valid explicit confidence artifact.",
            "Unseen surfaces are not reconstructed or claimed as measured.",
        ],
    }


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
