from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..models import ConfidenceLabel, PointConfidenceArtifact

CONFIDENCE_REQUIRED_FIELDS = [
    "point_id",
    "supporting_views",
    "track_length",
    "reprojection_error",
    "triangulation_angle",
    "confidence_class",
]

OBSERVED_CONFIDENCE_THRESHOLDS = {
    "OBSERVED_HIGH": {
        "min_track_length": 4,
        "max_reprojection_error_px": 1.0,
        "min_triangulation_angle_deg": 5.0,
    },
    "OBSERVED_MEDIUM": {
        "min_track_length": 3,
        "max_reprojection_error_px": 2.0,
        "min_triangulation_angle_deg": 2.0,
    },
}


def classify_observed_point(
    track_length: int, reprojection_error: float, triangulation_angle: float
) -> ConfidenceLabel:
    """Classify only COLMAP-observed points using documented geometric evidence."""
    high = OBSERVED_CONFIDENCE_THRESHOLDS["OBSERVED_HIGH"]
    if (
        track_length >= high["min_track_length"]
        and reprojection_error <= high["max_reprojection_error_px"]
        and triangulation_angle >= high["min_triangulation_angle_deg"]
    ):
        return ConfidenceLabel.OBSERVED_HIGH
    medium = OBSERVED_CONFIDENCE_THRESHOLDS["OBSERVED_MEDIUM"]
    if (
        track_length >= medium["min_track_length"]
        and reprojection_error <= medium["max_reprojection_error_px"]
        and triangulation_angle >= medium["min_triangulation_angle_deg"]
    ):
        return ConfidenceLabel.OBSERVED_MEDIUM
    return ConfidenceLabel.OBSERVED_LOW


def load_point_confidence(path: Path) -> PointConfidenceArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PointConfidenceArtifact.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid explicit point-confidence artifact: {path.name}: {exc}") from exc


def ply_vertex_count(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            header = stream.read(64 * 1024).split(b"end_header", 1)[0].decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"PLY header is unreadable: {path.name}") from exc
    match = next(
        (line for line in header.splitlines() if line.startswith("element vertex ")),
        None,
    )
    if match is None:
        raise ValueError(f"PLY has no vertex declaration: {path.name}")
    return int(match.split()[2])


def validate_point_confidence_for_ply(
    confidence_path: Path, ply_path: Path
) -> PointConfidenceArtifact:
    artifact = load_point_confidence(confidence_path)
    if len(artifact.points) != ply_vertex_count(ply_path):
        raise ValueError(
            "Explicit confidence point count does not match the declared PLY vertex count"
        )
    return artifact


def confidence_contract() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "supported_artifact": "point_confidence.json",
        "point_order": "PLY_VERTEX_ORDER",
        "required_fields": CONFIDENCE_REQUIRED_FIELDS,
        "valid_classes": [label.value for label in ConfidenceLabel],
        "rgb_derivation_prohibited": True,
        "observed_thresholds": OBSERVED_CONFIDENCE_THRESHOLDS,
    }
