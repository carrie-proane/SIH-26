from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .confidence import confidence_contract
from .dataset import CheckpointCorrespondenceSet, DatasetManifest
from .models import MatcherMetrics, MeasurementRecord, ProvenanceOrigin, RunRecord, utc_now


def known_distance_metrics(reference_m: float | None, measured_m: float | None) -> dict[str, Any]:
    if reference_m is None or measured_m is None:
        return {
            "status": "NOT_PROVIDED",
            "reference_m": reference_m,
            "measured_m": measured_m,
            "absolute_error_m": None,
            "percent_error": None,
            "passes_10_percent_gate": None,
            "gate_source": "legacy engineering threshold; not the official SIH spatial target",
            "reconstructed_value_source": "MANUAL_INPUT" if measured_m is not None else None,
            "independent_evaluation_eligible": False,
            "independent_evaluation_reason": "Legacy input has no declared control/evaluation role.",
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
        "gate_source": "legacy engineering threshold; not the official SIH spatial target",
        "reconstructed_value_source": "MANUAL_INPUT",
        "independent_evaluation_eligible": False,
        "independent_evaluation_reason": "Legacy input has no declared control/evaluation role.",
    }


def _error_statistics(measurements: list[MeasurementRecord]) -> dict[str, Any]:
    eligible = [
        item
        for item in measurements
        if item.measurement_eligible
        and item.reference_role == "HELD_OUT_EVALUATION"
        and item.reference_value_m is not None
    ]
    excluded_controls = sum(item.reference_role == "SCALE_CONTROL" for item in measurements)
    if not eligible:
        return {
            "status": "UNAVAILABLE",
            "sample_count": 0,
            "excluded_scale_control_count": excluded_controls,
            "reason": "No eligible held-out references were supplied.",
        }
    absolute = [abs(item.backend_distance_m - float(item.reference_value_m)) for item in eligible]
    signed = [item.backend_distance_m - float(item.reference_value_m) for item in eligible]
    relative = [
        100 * error / float(item.reference_value_m)
        for error, item in zip(absolute, eligible, strict=True)
    ]
    ordered = sorted(absolute)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "status": "EVALUATED",
        "sample_count": len(eligible),
        "units": "m",
        "mean_signed_error_m": sum(signed) / len(signed),
        "median_absolute_error_m": median(absolute),
        "rmse_m": math.sqrt(sum(value * value for value in signed) / len(signed)),
        "p95_absolute_error_m": ordered[p95_index],
        "maximum_absolute_error_m": max(absolute),
        "median_relative_error_percent": median(relative),
        "excluded_scale_control_count": excluded_controls,
    }


def measurement_evaluation(measurements: list[MeasurementRecord]) -> dict[str, Any]:
    """Summarize only held-out, backend-computed, measurement-eligible references."""

    kinds = {
        "horizontal": "HORIZONTAL",
        "vertical": "VERTICAL",
        "spatial_3d": "DISTANCE_3D",
        "relative_dimension": "RELATIVE_DIMENSION",
    }
    return {
        "schema_version": "1.0",
        "official_requirement": {
            "threshold_m": 1.0,
            "source": "SIH26158 desired-output table",
            "compliance_status": "PROTOCOL_UNDEFINED",
            "ambiguity": (
                "The source does not define the statistic, coordinate dimensions, reference "
                "accuracy, or sampling protocol; no single favorable statistic is labelled compliant."
            ),
        },
        "by_kind": {
            label: _error_statistics(
                [item for item in measurements if item.measurement_kind == kind]
            )
            for label, kind in kinds.items()
        },
        "all_held_out": _error_statistics(measurements),
    }


def _distribution(values: list[float], *, signed: bool = False) -> dict[str, Any]:
    ordered = sorted(abs(value) for value in values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    result: dict[str, Any] = {
        "count": len(values),
        "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
        "median_absolute_error_m": median(ordered),
        "p95_absolute_error_m": ordered[p95_index],
        "maximum_absolute_error_m": ordered[-1],
    }
    if signed:
        result["mean_signed_error_m"] = sum(values) / len(values)
    return result


def _transform_point(
    matrix: tuple[tuple[float, ...], ...], point: tuple[float, ...]
) -> tuple[float, float, float]:
    vector = (*point, 1.0)
    result = tuple(
        sum(matrix[row][column] * vector[column] for column in range(4)) for row in range(3)
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("coordinate transform produced non-finite coordinates")
    return result  # type: ignore[return-value]


def positional_checkpoint_evaluation(
    manifest: DatasetManifest,
    correspondences: CheckpointCorrespondenceSet | None,
) -> dict[str, Any]:
    """Evaluate declared held-out checkpoints without fitting to evaluation observations."""

    references = manifest.references.positional_checkpoints
    controls = [item for item in references if item.role == "SCALE_CONTROL"]
    heldout = [item for item in references if item.role == "HELD_OUT_EVALUATION"]
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "units": "m",
        "held_out_reference_count": len(heldout),
        "excluded_scale_control_count": len(controls),
        "shape_aligned_evaluation": {
            "status": "NOT_PERFORMED",
            "reason": "Shape-aligned evaluation must be reported separately from georeferenced accuracy.",
        },
        "official_requirement": {
            "threshold_m": 1.0,
            "compliance_status": "PROTOCOL_UNDEFINED",
            "reason": (
                "The SIH desired output does not define the statistic, coordinate dimensions, "
                "reference accuracy, or checkpoint sampling protocol."
            ),
        },
    }
    if not heldout:
        return base | {
            "status": "UNVERIFIED",
            "reason": "No held-out positional checkpoints were declared.",
        }
    if correspondences is None:
        return base | {
            "status": "UNVERIFIED",
            "reason": "No reconstructed checkpoint correspondences were supplied.",
        }
    coordinate_system = manifest.references.coordinate_system
    if coordinate_system is None:
        return base | {
            "status": "UNAVAILABLE_INCOMPATIBLE_FRAME",
            "reason": "Reference coordinate-system metadata is missing.",
        }
    mismatched_reference_frames = [
        item.id for item in references if item.coordinate_frame != coordinate_system.frame
    ]
    if mismatched_reference_frames:
        return base | {
            "status": "UNAVAILABLE_INCOMPATIBLE_FRAME",
            "reason": "Checkpoint coordinate frames do not match the declared reference frame.",
            "mismatched_checkpoint_ids": mismatched_reference_frames,
        }

    transform = coordinate_system.reference_to_reconstruction_transform
    same_frame = coordinate_system.frame == correspondences.reconstruction_coordinate_frame
    if not same_frame:
        if transform is None:
            return base | {
                "status": "UNAVAILABLE_INCOMPATIBLE_FRAME",
                "reason": "Reference and reconstruction frames differ and no explicit transform was declared.",
                "reference_frame": coordinate_system.frame,
                "reconstruction_frame": correspondences.reconstruction_coordinate_frame,
            }
        if (
            transform.source_frame != coordinate_system.frame
            or transform.target_frame != correspondences.reconstruction_coordinate_frame
        ):
            return base | {
                "status": "UNAVAILABLE_INCOMPATIBLE_FRAME",
                "reason": "The declared transform does not map the reference frame to the reconstruction frame.",
            }
        heldout_ids = {item.id for item in heldout}
        leaked = sorted(heldout_ids.intersection(transform.fitted_using_checkpoint_ids))
        if leaked:
            return base | {
                "status": "UNAVAILABLE_CONTROL_LEAKAGE",
                "reason": "The frame transform was fitted using held-out evaluation checkpoints.",
                "leaked_checkpoint_ids": leaked,
            }
    elif coordinate_system.units == "degrees":
        return base | {
            "status": "UNAVAILABLE_INCOMPATIBLE_UNITS",
            "reason": "Angular coordinates cannot be compared directly with reconstruction metres.",
        }

    scales = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    correspondence_by_id = {
        item.checkpoint_id: item.reconstructed_coordinates for item in correspondences.checkpoints
    }
    per_checkpoint: list[dict[str, Any]] = []
    residuals: list[tuple[float, float, float]] = []
    missing: list[str] = []
    vertical_compatible = all(
        item.altitude_reference == correspondences.altitude_reference
        and item.altitude_reference.upper() != "UNKNOWN"
        for item in heldout
    )
    for reference in heldout:
        reconstructed = correspondence_by_id.get(reference.id)
        if reconstructed is None:
            missing.append(reference.id)
            continue
        if transform is not None and not same_frame:
            expected = _transform_point(transform.matrix_4x4, reference.coordinates)
        else:
            scale = scales.get(reference.units)
            if scale is None:
                return base | {
                    "status": "UNAVAILABLE_INCOMPATIBLE_UNITS",
                    "reason": f"Checkpoint {reference.id} uses non-metric units without a transform.",
                }
            expected = tuple(value * scale for value in reference.coordinates)
        residual = tuple(reconstructed[index] - expected[index] for index in range(3))
        residuals.append(residual)  # type: ignore[arg-type]
        per_checkpoint.append(
            {
                "checkpoint_id": reference.id,
                "reference_role": reference.role,
                "expected_reconstruction_coordinates_m": expected,
                "reconstructed_coordinates_m": reconstructed,
                "signed_axis_residual_m": {
                    "x": residual[0],
                    "y": residual[1],
                    "z": residual[2] if vertical_compatible else None,
                },
                "horizontal_error_m": math.hypot(residual[0], residual[1]),
                "vertical_error_m": abs(residual[2]) if vertical_compatible else None,
                "spatial_3d_error_m": math.sqrt(sum(value * value for value in residual))
                if vertical_compatible
                else None,
                "stated_reference_accuracy_m": reference.stated_accuracy_m,
            }
        )
    if not residuals:
        return base | {
            "status": "UNVERIFIED",
            "reason": "No held-out checkpoint has a reconstructed correspondence.",
            "missing_checkpoint_ids": missing,
        }

    horizontal = [math.hypot(item[0], item[1]) for item in residuals]
    axes: dict[str, Any] = {
        "x": _distribution([item[0] for item in residuals], signed=True),
        "y": _distribution([item[1] for item in residuals], signed=True),
        "z": (
            _distribution([item[2] for item in residuals], signed=True)
            if vertical_compatible
            else {
                "status": "UNAVAILABLE_INCOMPATIBLE_VERTICAL_DATUM",
                "reason": "Checkpoint and reconstruction altitude references are missing or differ.",
            }
        ),
    }
    spatial = [math.sqrt(sum(value * value for value in item)) for item in residuals]
    metrics = {
        "horizontal": _distribution(horizontal),
        "vertical": (
            _distribution([item[2] for item in residuals])
            if vertical_compatible
            else {"status": "UNAVAILABLE_INCOMPATIBLE_VERTICAL_DATUM"}
        ),
        "spatial_3d": (
            _distribution(spatial)
            if vertical_compatible
            else {"status": "UNAVAILABLE_INCOMPATIBLE_VERTICAL_DATUM"}
        ),
        "signed_axes": axes,
    }
    return base | {
        "status": "EVALUATED" if vertical_compatible else "PARTIALLY_EVALUATED_HORIZONTAL_ONLY",
        "evaluated_checkpoint_count": len(residuals),
        "missing_checkpoint_ids": missing,
        "reference_frame": coordinate_system.frame,
        "reconstruction_frame": correspondences.reconstruction_coordinate_frame,
        "transform_applied": transform.model_dump(mode="json")
        if transform and not same_frame
        else None,
        "metrics": metrics,
        "per_checkpoint": per_checkpoint,
        "provisional_threshold_interpretations": {
            "label": "PROVISIONAL_NOT_ORGANIZER_COMPLIANCE",
            "horizontal_rmse_at_most_1m": metrics["horizontal"]["rmse_m"] <= 1.0,
            "spatial_3d_rmse_at_most_1m": (
                metrics["spatial_3d"]["rmse_m"] <= 1.0 if vertical_compatible else None
            ),
        },
    }


def build_dataset_evaluation(
    manifest: DatasetManifest,
    measurements: list[MeasurementRecord],
    correspondences: CheckpointCorrespondenceSet | None,
) -> dict[str, Any]:
    declared_distances = {item.id: item for item in manifest.references.relative_distances}
    bound_measurements: list[MeasurementRecord] = []
    rejected_measurements: list[dict[str, Any]] = []
    for item in measurements:
        reference = declared_distances.get(item.reference_id or "")
        if reference is None:
            rejected_measurements.append(
                {
                    "measurement_id": item.measurement_id,
                    "reason": "Measurement has no reference_id bound to this dataset manifest.",
                }
            )
            continue
        if item.reference_role != reference.role:
            rejected_measurements.append(
                {
                    "measurement_id": item.measurement_id,
                    "reason": "Measurement role differs from its manifest reference role.",
                }
            )
            continue
        if item.reference_value_m is None or not math.isclose(
            item.reference_value_m, reference.distance_m, rel_tol=0, abs_tol=1e-9
        ):
            rejected_measurements.append(
                {
                    "measurement_id": item.measurement_id,
                    "reason": "Measurement reference value differs from the hash-bound manifest value.",
                }
            )
            continue
        bound_measurements.append(item)
    relative_evaluation = measurement_evaluation(bound_measurements)
    relative_evaluation["manifest_binding"] = {
        "status": (
            "BOUND"
            if bound_measurements
            else (
                "UNVERIFIED_NO_MATCHED_MEASUREMENTS"
                if declared_distances
                else "UNVERIFIED_NO_DECLARED_REFERENCES"
            )
        ),
        "declared_reference_count": len(declared_distances),
        "bound_measurement_count": len(bound_measurements),
        "rejected_measurements": rejected_measurements,
    }
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "dataset_id": manifest.dataset_id,
        "dataset_role": manifest.dataset_role,
        "relative_distance": relative_evaluation,
        "positional_checkpoints": positional_checkpoint_evaluation(manifest, correspondences),
        "surface_completeness": {
            "status": "UNAVAILABLE",
            "reason": "No visible-surface reference domain and denominator were supplied.",
            "registered_frame_rate_is_not_surface_completeness": True,
        },
    }


def build_quality_report(
    record: RunRecord,
    metrics: MatcherMetrics,
    warnings: list[dict[str, str]],
    *,
    alignment: dict[str, Any] | None = None,
    confidence_available: bool = False,
    scene_analysis: dict[str, Any] | None = None,
    frame_quality: dict[str, Any] | None = None,
    geometry_diagnostics: dict[str, Any] | None = None,
    camera_model_selection: dict[str, Any] | None = None,
    temporal_coverage: dict[str, Any] | None = None,
    dense_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registration_rate = metrics.registration_rate
    alignment = alignment or {}
    scene_analysis = scene_analysis or {}
    frame_quality = frame_quality or {}
    geometry_diagnostics = geometry_diagnostics or {}
    camera_model_selection = camera_model_selection or {}
    temporal_coverage = temporal_coverage or {}
    dense_report = dense_report or {}
    residuals = [float(value) for value in alignment.get("residuals_m", [])]
    inlier_count = int(alignment.get("inlier_count", 0))
    sorted_residuals = sorted(residuals)
    median_residual = sorted_residuals[len(sorted_residuals) // 2] if sorted_residuals else None
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
    report_warnings.extend(geometry_diagnostics.get("warnings", []))
    genuine_real_evidence = record.source_provenance == ProvenanceOrigin.REAL
    geometry_gate = bool(geometry_diagnostics.get("measurement_geometry_gate_passed", False))
    metric_validated = (
        genuine_real_evidence
        and geometry_gate
        and inlier_count >= 3
        and isinstance(alignment.get("scale"), (int, float))
        and math.isfinite(float(alignment["scale"]))
        and float(alignment["scale"]) > 0
    )
    dense_status = (
        str(dense_report.get("status", "UNAVAILABLE"))
        if record.config.enable_dense_reconstruction
        else "NOT_REQUESTED"
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
        "synthetic_fixture": record.source_provenance == ProvenanceOrigin.SYNTHETIC,
        "source_provenance": record.source_provenance,
        "video_origin": record.video_origin,
        "telemetry_origin": record.telemetry_origin,
        "genuine_real_evidence": genuine_real_evidence,
        "completion_summary": {
            "sparse_status": "SUCCESS",
            "dense_status": dense_status,
            "technical_completion": "COMPLETED",
            "metric_validation_status": "VALIDATED" if metric_validated else "LIMITED",
            "partial_success": bool(
                not metric_validated
                or (
                    record.config.enable_dense_reconstruction
                    and dense_status not in {"COMPLETED", "SUCCESS"}
                )
            ),
        },
        "execution": {
            "requested_matcher": record.requested_matcher or record.config.matcher,
            "executed_matcher": record.executed_matcher,
            "stage_timings_s": record.stage_timings_s,
            "processing_started_at": record.processing_started_at,
            "processing_completed_at": record.processing_completed_at,
            "actual_sparse_backend": (
                "COLMAP_GPU" if record.effective_sparse_gpu else "COLMAP_CPU"
            ),
            "actual_dense_backend": record.selected_dense_provider,
            "resource_settings": {
                "worker_threads": record.config.worker_threads,
                "max_candidate_frames": record.config.max_candidate_frames,
                "max_selected_frames": record.config.max_selected_frames,
                "processing_max_image_dimension": record.config.processing_max_image_dimension,
            },
        },
        "official_targets": {
            "runtime": {
                "requirement": "strictly below 900 seconds for a 600-second input video",
                "source": "SIH26158 desired-output table",
                "status": "NOT_VALIDATED_BY_THIS_RUN",
                "reason": "Only a fresh representative 600-second benchmark can establish compliance.",
            },
            "spatial_accuracy": {
                "requirement": "at most 1 metre",
                "source": "SIH26158 desired-output table",
                "status": "PROTOCOL_UNDEFINED",
                "reason": (
                    "The official statistic, dimensions, sampling, and reference protocol are unspecified."
                ),
            },
            "visible_scene_coverage": {
                "requirement": "entire visible scene",
                "status": "UNAVAILABLE",
                "reason": "No defensible visible-surface reference domain and denominator were supplied.",
            },
        },
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
            "telemetry_sync": {
                "telemetry_offset_s": record.telemetry_offset_s,
                "offset_source": record.offset_source,
                "rmse_before_m": record.rmse_before_m,
                "rmse_after_m": record.rmse_after_m,
                "matched_camera_count": record.matched_camera_count,
                "inlier_count": record.inlier_count,
            },
            "known_distance": known_distance_metrics(
                record.config.known_distance_m, record.config.measured_distance_m
            ),
            "coverage": {
                "status": "NOT_EVALUATED",
                "reason": "Reference visible-region mask not supplied.",
                "temporal_diagnostics": temporal_coverage,
                "observed_geometry_only": True,
                "inferred_geometry_included": False,
            },
            "frame_quality_gate": frame_quality,
            "geometry_diagnostics": geometry_diagnostics,
            "camera_model_selection": camera_model_selection,
            "reconstruction_policy": {
                "target": record.config.reconstruction_target,
                "masking_mode": record.config.masking_mode,
                "masking_decision": scene_analysis.get("masking_decision", "NOT_EVALUATED"),
                "scene_recommendation": scene_analysis.get("recommendation"),
                "dense_suitability": scene_analysis.get("dense_suitability"),
                "operational_masks_declared": any(
                    artifact.relative_path.startswith("masks/reconstruction/")
                    for artifact in record.artifacts
                ),
            },
        },
        "scene_analysis": scene_analysis,
        "confidence_artifact": {
            "available": confidence_available,
            "measurement_confidence_available": confidence_available,
            "reason": (None if confidence_available else "Confidence unavailable for this run"),
            "contract": confidence_contract(),
        },
        "warnings": report_warnings,
        "limitations": [
            "Geometry is defensible only where observed by multiple source frames.",
            "Ordinary GNSS is a soft alignment prior and is not survey-grade ground truth.",
            "AI-assisted geometry is excluded from measurement evidence.",
            "Confidence-qualified measurement is unavailable without a valid explicit confidence artifact.",
            "Unseen surfaces are not reconstructed or claimed as measured.",
        ],
    }


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
