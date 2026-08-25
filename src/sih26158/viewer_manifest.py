from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import RunRecord

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
    measurement_reference = {
        "label": "Independent known distance",
        "status": known_distance.get("status", "NOT_MEASURED"),
        "reference_m": known_distance.get("reference_m"),
        "measured_m": known_distance.get("measured_m"),
        "percent_error": known_distance.get("percent_error"),
        "passes_10_percent_gate": known_distance.get("passes_10_percent_gate"),
        "synthetic_fixture": record.synthetic_fixture,
    }

    return {
        "schema_version": "1.0",
        "project_id": record.project_id,
        "run_id": record.run_id,
        "stage": record.stage,
        "status": record.status,
        "synthetic_fixture": record.synthetic_fixture,
        "cloud": {
            "url": artifacts[cloud_path].url,
            "format": "PLY",
            "coordinate_frame": coordinate_frame,
        },
        "camera_path": {
            "url": artifacts["camera_poses.csv"].url,
            "coordinate_frame": coordinate_frame,
        },
        "selected_frames": {"url": artifacts["keyframes.json"].url},
        "confidence_legend": CONFIDENCE_LEGEND,
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
