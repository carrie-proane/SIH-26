from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .confidence import confidence_contract, validate_point_confidence_for_ply
from .models import ProvenanceOrigin, RunRecord

CONFIDENCE_LEGEND = [
    {"label": "OBSERVED_HIGH", "color": "#20bf6b", "measurement": "ALLOWED"},
    {"label": "OBSERVED_MEDIUM", "color": "#f5a30b", "measurement": "CAUTION"},
    {"label": "OBSERVED_LOW", "color": "#ef4444", "measurement": "CONFIRM"},
    {
        "label": "AI_ASSISTED_NOT_MEASURABLE",
        "color": "#a855f7",
        "measurement": "DISABLED",
    },
    {"label": "UNSEEN", "color": "#64748b", "measurement": "DISABLED"},
]


class ViewerManifestUnavailable(RuntimeError):
    """The run does not yet expose the minimum safe viewer artifact set."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewerManifestUnavailable(f"Viewer artifact is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ViewerManifestUnavailable(f"Viewer artifact must contain an object: {path.name}")
    return payload


def build_viewer_manifest(record: RunRecord, run_dir: Path) -> dict[str, Any]:
    """Render Arnav's stable browser payload from Jay's declared artifacts.

    The endpoint deliberately uses only artifacts already declared on the run.
    It does not guess files, publish the run directory, or invent successful
    metrics when a reconstruction is incomplete.
    """

    artifacts = {item.relative_path: item for item in record.artifacts}
    cloud_path = next(
        (
            candidate
            for candidate in ("sparse/sparse_local.ply", "sparse/sparse.ply")
            if candidate in artifacts
        ),
        None,
    )
    required = {
        "cloud": cloud_path,
        "camera path": "camera_poses.csv" if "camera_poses.csv" in artifacts else None,
        "selected frames": "keyframes.json" if "keyframes.json" in artifacts else None,
        "quality report": "quality_report.json" if "quality_report.json" in artifacts else None,
    }
    missing = [name for name, relative_path in required.items() if relative_path is None]
    if missing:
        raise ViewerManifestUnavailable(
            "Viewer is not ready; missing declared artifacts: " + ", ".join(missing)
        )

    quality = _read_json(run_dir / "quality_report.json")
    known_distance = quality.get("metrics", {}).get("known_distance", {})
    if not isinstance(known_distance, dict):
        known_distance = {}

    coordinate_frame = (
        "LOCAL_ENU_METRES" if cloud_path == "sparse/sparse_local.ply" else "COLMAP_SFM"
    )
    confidence_path = (
        "point_confidence.json" if "point_confidence.json" in artifacts else None
    )
    confidence_reason = "Confidence unavailable for this run"
    confidence_available = False
    if confidence_path is not None:
        try:
            validate_point_confidence_for_ply(
                run_dir / confidence_path, run_dir / cloud_path
            )
            confidence_available = True
            confidence_reason = "Explicit point-confidence artifact validated."
        except ValueError as exc:
            confidence_reason = str(exc)
    measurement_reference = {
        "label": "Independent known distance",
        "status": known_distance.get("status", "NOT_MEASURED"),
        "reference_m": known_distance.get("reference_m"),
        "measured_m": known_distance.get("measured_m"),
        "percent_error": known_distance.get("percent_error"),
        "passes_10_percent_gate": known_distance.get("passes_10_percent_gate"),
        "synthetic_fixture": record.synthetic_fixture,
    }
    dense_cloud_path = "dense/fused.ply" if "dense/fused.ply" in artifacts else None
    textured_mesh_path = next(
        (
            path
            for path in sorted(artifacts)
            if path.startswith("dense/textured/") and path.lower().endswith(".ply")
        ),
        None,
    )
    texture_paths = [
        path
        for path in sorted(artifacts)
        if path.startswith("dense/textured/")
        and path.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    return {
        "schema_version": "1.0",
        "project_id": record.project_id,
        "run_id": record.run_id,
        "stage": record.stage,
        "status": record.status,
        "synthetic_fixture": record.synthetic_fixture,
        "source_provenance": record.source_provenance,
        "video_origin": record.video_origin,
        "telemetry_origin": record.telemetry_origin,
        "genuine_real_evidence": record.source_provenance == ProvenanceOrigin.REAL,
        "cloud": {
            "url": artifacts[cloud_path].url,
            "format": "PLY",
            "coordinate_frame": coordinate_frame,
            "color_mode": "PHOTOGRAPHIC_RGB",
            "color_mode_label": "Photographic RGB",
        },
        "visual_models": {
            "evidence_cloud": {
                "available": True,
                "url": artifacts[cloud_path].url,
                "format": "PLY",
                "coordinate_frame": coordinate_frame,
                "measurement_eligible": True,
                "default": True,
            },
            "dense_cloud": {
                "available": dense_cloud_path is not None,
                "url": artifacts[dense_cloud_path].url if dense_cloud_path else None,
                "format": "PLY" if dense_cloud_path else None,
                "coordinate_frame": "LOCAL_ENU_METRES" if dense_cloud_path else None,
                "measurement_eligible": False,
            },
            "textured_mesh": {
                "available": textured_mesh_path is not None,
                "url": artifacts[textured_mesh_path].url if textured_mesh_path else None,
                "format": "PLY" if textured_mesh_path else None,
                "texture_urls": [artifacts[path].url for path in texture_paths],
                "coordinate_frame": "LOCAL_ENU_METRES" if textured_mesh_path else None,
                "measurement_eligible": False,
                "statement": "Visual model - not used for verified measurement",
            },
            "gaussian_splat": {
                "available": False,
                "url": None,
                "format": None,
                "measurement_eligible": False,
                "statement": "Photoreal View unavailable; no Gaussian Splatting was installed or run.",
            },
            "dense_report_url": (
                artifacts["dense_report.json"].url
                if "dense_report.json" in artifacts
                else None
            ),
        },
        "camera_path": {
            "url": artifacts["camera_poses.csv"].url,
            "coordinate_frame": coordinate_frame,
        },
        "selected_frames": {"url": artifacts["keyframes.json"].url},
        "confidence_legend": CONFIDENCE_LEGEND,
        "confidence": {
            "available": confidence_available,
            "url": artifacts[confidence_path].url if confidence_available else None,
            "format": "POINT_CONFIDENCE_JSON" if confidence_available else None,
            "reason": confidence_reason,
            "contract": confidence_contract(),
        },
        "measurement_reference": measurement_reference,
        "quality_report_url": artifacts["quality_report.json"].url,
        "ingest_report_url": (
            artifacts["ingest_report.json"].url if "ingest_report.json" in artifacts else None
        ),
        "ai_overlay": {
            "available": False,
            "label": "AI_ASSISTED_NOT_MEASURABLE",
            "measurement": "DISABLED",
            "reason": "No declared Depth Anything overlay artifact exists for this run.",
        },
    }
