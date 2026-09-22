from __future__ import annotations

import math
import sys
from datetime import datetime
from typing import Any

from .models import ProvenanceOrigin, RunCheckpoint, RunRecord, utc_now


def _seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return max(
            0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        )
    except ValueError:
        return None


def _peak_rss() -> dict[str, Any]:
    try:
        import resource

        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (AttributeError, ImportError, OSError, ValueError):
        return {"status": "UNAVAILABLE"}
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {
        "status": "AVAILABLE_PROCESS_PEAK",
        "orchestrator_peak_rss_bytes": int(self_rss * multiplier),
        "completed_children_peak_rss_bytes": int(child_rss * multiplier),
        "scope": (
            "OS-reported process maxima sampled at report time; this is not a synchronized "
            "whole-process-tree peak and may omit externally managed descendants."
        ),
    }


def build_benchmark_report(
    record: RunRecord,
    checkpoint: RunCheckpoint,
    ingest: dict[str, Any],
    keyframes: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    reused_stages: list[str],
    active_report_duration_s: float | None = None,
    export_duration_s: float | None = None,
    export_report: dict[str, Any] | None = None,
    dense_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    streams = ingest.get("video_probe", {}).get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    duration_s = float(
        ingest.get("video_probe", {}).get("format", {}).get("duration")
        or keyframes.get("temporal_coverage", {}).get("video_duration_s")
        or 0
    )
    processing_wall_s = _seconds_between(
        record.processing_started_at, record.processing_completed_at
    )
    stage_times = {name: completed.duration_s for name, completed in checkpoint.completed.items()}
    if active_report_duration_s is not None:
        stage_times["REPORT"] = max(0.0, active_report_duration_s)
    stage_execution_s = sum(value for value in stage_times.values() if value is not None)
    reconstruction_execution_s = sum(stage_times.get(name) or 0.0 for name in ("SPARSE", "DENSE"))
    selected = [
        item
        for item in keyframes.get("frames", [])
        if isinstance(item, dict) and item.get("selected", True)
    ]
    registered = [
        item
        for item in keyframes.get("frames", [])
        if isinstance(item, dict) and item.get("reconstruction_status") == "REGISTERED"
    ]
    representative = 595 <= duration_s <= 605
    fresh = not reused_stages
    target_passed = bool(
        representative
        and fresh
        and record.source_provenance == ProvenanceOrigin.REAL
        and processing_wall_s is not None
        and processing_wall_s < 900
    )
    if not representative:
        target_status = "NOT_VALIDATED_NON_REPRESENTATIVE_DURATION"
    elif not fresh:
        target_status = "NOT_VALIDATED_CACHED_STAGE_REUSE"
    elif record.source_provenance != ProvenanceOrigin.REAL:
        target_status = (
            "NOT_VALIDATED_SYNTHETIC_FIXTURE"
            if record.synthetic_fixture
            else "NOT_VALIDATED_NON_REAL_PROVENANCE"
        )
    else:
        target_status = "PASS" if target_passed else "FAIL"
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "run_id": record.run_id,
        "timing_contract": {
            "start": "PREPROCESS stage start after immutable upload/queue",
            "end": "quality and export-readiness reports produced after requested reconstruction stages",
            "queue_time_s": _seconds_between(record.created_at, record.processing_started_at),
            "upload_time_s": None,
            "upload_time_reason": "Upload timing is not currently instrumented.",
            "processing_wall_s": processing_wall_s,
            "stage_execution_s": stage_execution_s,
            "stage_timings_s": stage_times,
            "reconstruction_execution_s": reconstruction_execution_s,
            "export_execution_s": export_duration_s,
            "export_timing_boundary": (
                "build_export_readiness entry through validated, atomically published export packages"
            ),
            "export_included_in_processing_wall": True,
        },
        "cache_classification": "COLD_FRESH" if fresh else "WARM_OR_RESUMED",
        "reused_stages": reused_stages,
        "input": {
            "dataset_id": record.config.dataset_id,
            "dataset_role": record.config.dataset_role,
            "dataset_manifest_sha256": record.config.dataset_manifest_sha256,
            "dataset_benchmark_config_sha256": record.config.dataset_benchmark_config_sha256,
            "assets": [
                {
                    "role": item.get("role"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                    "origin": item.get("origin"),
                }
                for item in ingest.get("input_assets", [])
                if isinstance(item, dict)
            ],
            "duration_s": duration_s,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "candidate_frames": len(keyframes.get("frames", [])),
            "selected_frames": len(selected),
            "processing_max_image_dimension": record.config.processing_max_image_dimension,
        },
        "official_runtime_gate": {
            "input_duration_target_s": 600,
            "processing_time_limit_s": 900,
            "comparison": "strictly_less_than",
            "status": target_status,
            "passed": target_passed if target_status in {"PASS", "FAIL"} else None,
        },
        "execution": {
            "requested_matcher": record.requested_matcher or record.config.matcher,
            "executed_matcher": record.executed_matcher,
            "source_revision": record.environment.get("git_revision"),
            "source_dirty": record.environment.get("git_dirty"),
            "tool_versions": record.environment,
            "sparse_backend": "COLMAP_GPU" if record.effective_sparse_gpu else "COLMAP_CPU",
            "dense_backend": record.selected_dense_provider,
            "worker_threads_requested": record.config.worker_threads,
            "worker_threads_effective": capabilities.get("resource_policy", {}).get(
                "worker_threads_effective"
            ),
            "heavy_job_limit": capabilities.get("resource_policy", {}).get("heavy_job_limit"),
            "capabilities": capabilities,
            "effective_configuration": record.config.model_dump(mode="json"),
            "peak_memory": _peak_rss(),
        },
        "outcomes": {
            "sparse": {
                "status": "COMPLETED" if "SPARSE" in checkpoint.completed else "UNAVAILABLE",
                "selected_frames": len(selected),
                "registered_frames": len(registered),
            },
            "dense": dense_report
            or {
                "status": "NOT_REQUESTED"
                if not record.config.enable_dense_reconstruction
                else "UNAVAILABLE"
            },
            "exports": {
                "status": "RECORDED" if export_report else "UNAVAILABLE",
                "available_validated_formats": (export_report or {}).get(
                    "available_validated_formats", []
                ),
                "formats": (export_report or {}).get("formats", {}),
            },
            "surface_completeness": {
                "status": "UNAVAILABLE",
                "reason": "No visible-surface reference denominator was supplied.",
                "registered_frame_rate_is_not_surface_completeness": True,
            },
        },
        "artifacts": [
            {"path": item.relative_path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in record.artifacts
        ],
    }


def _input_signature(report: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    assets = report.get("input", {}).get("assets", [])
    signature = tuple(
        sorted(
            (str(item.get("role")), str(item.get("sha256")))
            for item in assets
            if item.get("role") in {"video", "telemetry"} and item.get("sha256")
        )
    )
    if {role for role, _ in signature} != {"video", "telemetry"}:
        return ()
    return signature


def _configuration_differences(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append({"path": path, "left": left.get(key), "right": right.get(key)})
            else:
                differences.extend(_configuration_differences(left[key], right[key], path))
        return differences
    if left != right:
        return [{"path": prefix, "left": left, "right": right}]
    return []


def compare_benchmark_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare only reports bound to identical video and telemetry hashes."""

    if len(reports) < 2:
        raise ValueError("At least two benchmark reports are required")
    baseline = reports[0]
    baseline_signature = _input_signature(baseline)
    entries: list[dict[str, Any]] = []
    for report in reports:
        signature = _input_signature(report)
        same_inputs = bool(baseline_signature) and signature == baseline_signature
        same_cache_class = report.get("cache_classification") == baseline.get(
            "cache_classification"
        )
        differences = _configuration_differences(
            baseline.get("execution", {}).get("effective_configuration", {}),
            report.get("execution", {}).get("effective_configuration", {}),
        )
        processing = report.get("timing_contract", {}).get("processing_wall_s")
        entry = {
            "run_id": report.get("run_id"),
            "same_input_hashes": same_inputs,
            "cache_classification": report.get("cache_classification"),
            "configuration_differences_from_baseline": differences,
            "processing_wall_s": processing if same_inputs else None,
            "comparison_status": (
                "COMPARABLE"
                if same_inputs and same_cache_class
                else ("INCOMPATIBLE_CACHE_CLASS" if same_inputs else "INCOMPATIBLE_INPUTS")
            ),
            "excluded_from_runtime_comparison": not (same_inputs and same_cache_class),
        }
        entries.append(entry)
    comparable_times = [
        float(item["processing_wall_s"])
        for item in entries
        if not item["excluded_from_runtime_comparison"]
        and isinstance(item["processing_wall_s"], (int, float))
        and math.isfinite(float(item["processing_wall_s"]))
    ]
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "baseline_run_id": baseline.get("run_id"),
        "input_signature": baseline_signature,
        "entries": entries,
        "comparable_run_count": len(comparable_times),
        "runtime_summary_s": (
            {
                "minimum": min(comparable_times),
                "median": sorted(comparable_times)[len(comparable_times) // 2],
                "maximum": max(comparable_times),
            }
            if comparable_times
            else {"status": "UNAVAILABLE"}
        ),
        "interpretation_warnings": [
            "Higher point or face counts are not evidence of higher positional accuracy.",
            "Warm/resumed and cold/fresh timings are not directly comparable.",
            "No runtime is extrapolated from a shorter input to the 600-second target.",
        ],
    }
