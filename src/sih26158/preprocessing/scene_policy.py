"""Conservative, auditable scene diagnostics for reconstruction policy decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..infrastructure.storage import atomic_json


def _selected_images(
    run_dir: Path, keyframes: list[dict[str, object]], limit: int = 24
) -> list[Path]:
    images = [
        run_dir / "frames" / Path(str(frame.get("image_name", ""))).name
        for frame in keyframes
        if frame.get("selected", True) and frame.get("image_name")
    ]
    images = [path for path in images if path.is_file()]
    if len(images) <= limit:
        return images
    indices = np.linspace(0, len(images) - 1, limit, dtype=int)
    return [images[int(index)] for index in indices]


def analyze_scene(
    run_dir: Path,
    keyframes: list[dict[str, object]],
    *,
    reconstruction_target: str,
    masking_mode: str,
) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    """Measure image-level risk signals without pretending to understand semantics.

    These diagnostics intentionally use conservative names such as ``sky_candidate`` and
    ``appearance_change``.  They recommend policy; they do not claim that an object or surface
    has been semantically identified without an explicit segmentation artifact.
    """

    sampled = _selected_images(run_dir, keyframes)
    featureless: list[float] = []
    sky_candidates: list[float] = []
    specular: list[float] = []
    appearance_change: list[float] = []
    previous: np.ndarray | None = None

    for path in sampled:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)
        featureless.append(float(np.mean(gradient < 10.0)))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        upper = hsv[: max(1, hsv.shape[0] // 2)]
        blue = (
            (upper[:, :, 0] >= 85)
            & (upper[:, :, 0] <= 135)
            & (upper[:, :, 1] >= 35)
            & (upper[:, :, 2] >= 70)
        )
        pale_bright = (upper[:, :, 1] < 35) & (upper[:, :, 2] > 225)
        sky_candidates.append(float(np.mean(blue | pale_bright)))
        specular.append(float(np.mean((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 242))))

        if previous is not None:
            difference = cv2.absdiff(gray, previous)
            appearance_change.append(float(np.mean(difference > 32)))
        previous = gray

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    mean_featureless = mean(featureless)
    mean_sky = mean(sky_candidates)
    mean_specular = mean(specular)
    mean_change = mean(appearance_change)
    extreme_unstable = bool(
        mean_featureless is not None
        and mean_change is not None
        and mean_featureless >= 0.85
        and mean_change >= 0.35
    )
    extreme_sky = bool(mean_sky is not None and mean_sky >= 0.70)

    if reconstruction_target == "PRIMARY_SUBJECT":
        recommendation = "SUBJECT_MASK_REQUIRED"
        rationale = "A primary-subject output cannot be defined without an explicit subject mask."
    elif any(
        value is not None and value >= threshold
        for value, threshold in (
            (mean_sky, 0.18),
            (mean_specular, 0.12),
            (mean_featureless, 0.60),
        )
    ):
        recommendation = "MASK_RECOMMENDED"
        rationale = (
            "Conservative image diagnostics found substantial sky-candidate, bright/specular, "
            "or low-texture content."
        )
    else:
        recommendation = "UNMASKED_ACCEPTABLE"
        rationale = "No conservative image diagnostic crossed the mask-recommendation threshold."

    dense_suitability = (
        "BLOCKED_WITHOUT_MASK" if extreme_unstable or extreme_sky else "PROCEED_WITH_GATES"
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "reconstruction_target": reconstruction_target,
        "masking_mode": masking_mode,
        "sampled_frame_count": len(featureless),
        "signals": {
            "mean_featureless_fraction": mean_featureless,
            "mean_upper_frame_sky_candidate_fraction": mean_sky,
            "mean_bright_specular_candidate_fraction": mean_specular,
            "mean_interframe_appearance_change_fraction": mean_change,
        },
        "recommendation": recommendation,
        "rationale": rationale,
        "dense_suitability": dense_suitability,
        "masking_decision": "NOT_EVALUATED",
        "limitations": [
            "Image diagnostics are heuristics, not semantic classification.",
            "Inter-frame appearance change includes intentional camera motion.",
            "A mask is authoritative only when its generated artifact is declared.",
        ],
    }
    warnings: list[dict[str, str]] = []
    if recommendation in {"MASK_RECOMMENDED", "SUBJECT_MASK_REQUIRED"}:
        warnings.append(
            {
                "code": "SCENE_MASK_RECOMMENDED",
                "message": rationale,
            }
        )
    report_path = run_dir / "scene_analysis.json"
    atomic_json(report_path, payload)
    return report_path, payload, warnings


def update_masking_decision(run_dir: Path, decision: str, reason: str) -> Path:
    path = run_dir / "scene_analysis.json"
    payload = {}
    if path.is_file():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["masking_decision"] = decision
    payload["masking_reason"] = reason
    atomic_json(path, payload)
    return path
