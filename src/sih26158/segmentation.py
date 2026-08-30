"""Optional scene-aware segmentation with a no-download, fail-honestly contract."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .scene_policy import update_masking_decision
from .storage import atomic_json

MaskProvider = Callable[[Path], np.ndarray]
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
SKY_CLASSES = {"sky"}


def _ultralytics_provider(model_path: Path, reconstruction_target: str) -> MaskProvider:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional segmentation dependency group") from exc
    model = YOLO(str(model_path))

    def segment(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read selected frame: {image_path.name}")
        candidates: list[tuple[str, np.ndarray]] = []
        results = model.predict(source=str(image_path), verbose=False)
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            for class_id, tensor in zip(
                result.boxes.cls.tolist(), result.masks.data.cpu().numpy(), strict=True
            ):
                resized = cv2.resize(
                    tensor.astype(np.float32),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                candidates.append((str(result.names[int(class_id)]).lower(), resized >= 0.5))

        if reconstruction_target == "PRIMARY_SUBJECT":
            if not candidates:
                raise ValueError("No primary-subject instance was detected")
            height, width = image.shape[:2]
            center_y, center_x = height / 2, width / 2

            def subject_score(candidate: tuple[str, np.ndarray]) -> float:
                _, mask = candidate
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    return 0.0
                area = float(len(xs)) / mask.size
                distance = np.hypot(xs.mean() - center_x, ys.mean() - center_y)
                normalized_distance = distance / max(1.0, np.hypot(center_x, center_y))
                return area * (1.5 - min(1.0, normalized_distance))

            subject = max(candidates, key=subject_score)[1]
            # Provider masks always mean EXCLUDED pixels: keep only the chosen subject.
            return np.where(subject, 0, 255).astype(np.uint8)

        excluded = np.zeros(image.shape[:2], dtype=np.uint8)
        for class_name, mask in candidates:
            if class_name in DYNAMIC_CLASSES | SKY_CLASSES:
                excluded[mask] = 255
        return excluded

    return segment


def _write_report(run_dir: Path, payload: dict[str, object]) -> Path:
    report_path = run_dir / "segmentation_comparison.json"
    atomic_json(report_path, payload)
    return report_path


def run_optional_segmentation(
    run_dir: Path,
    run_id: str,
    keyframes: list[dict[str, object]],
    model_path: str | None,
    *,
    provider: MaskProvider | None = None,
    reconstruction_target: str = "FULL_SCENE",
    masking_mode: str = "AUTO",
) -> tuple[list[Path], list[dict[str, str]]]:
    """Create review and reconstruction masks, or record an explicit fallback/blocker.

    A provider returns an exclusion mask: non-zero pixels are excluded. Operational masks use
    the inverse convention required by COLMAP/OpenMVS (zero excluded, 255 included).
    """

    selected_count = sum(bool(frame.get("selected", True)) for frame in keyframes)
    base_report: dict[str, object] = {
        "schema_version": "2.0",
        "reconstruction_target": reconstruction_target,
        "masking_mode": masking_mode,
        "mask_semantics": {
            "review_masks": "NONZERO_IS_EXCLUDED",
            "reconstruction_masks": "ZERO_IS_EXCLUDED",
        },
        "dynamic_classes": sorted(DYNAMIC_CLASSES),
        "sky_classes_when_model_supports_them": sorted(SKY_CLASSES),
        "selected_frame_count": selected_count,
        "masks_applied_to": [
            "COLMAP_SPARSE_FEATURE_EXTRACTION",
            "OPENMVS_DENSE_MATCHING",
            "OPENMVS_TEXTURE_GENERATION",
        ],
    }
    if masking_mode == "OFF":
        base_report |= {
            "status": "DISABLED",
            "masking_decision": "UNMASKED_BY_CONFIGURATION",
            "comparison": "The operator explicitly disabled masking.",
        }
        report = _write_report(run_dir, base_report)
        update_masking_decision(run_dir, "UNMASKED_BY_CONFIGURATION", "Masking mode is OFF.")
        return [report], []

    if provider is None:
        configured = model_path or os.environ.get("SIH_SEGMENTATION_MODEL")
        candidate = Path(configured).expanduser() if configured else None
        unavailable_reason = None
        if candidate is None or not candidate.is_file():
            unavailable_reason = "No local segmentation weights were supplied; no model was downloaded automatically."
        else:
            try:
                provider = _ultralytics_provider(candidate, reconstruction_target)
            except (RuntimeError, OSError) as exc:
                unavailable_reason = str(exc)
        if unavailable_reason is not None:
            blocked = masking_mode == "REQUIRED" or reconstruction_target == "PRIMARY_SUBJECT"
            decision = "BLOCKED_REQUIRED_MASK_UNAVAILABLE" if blocked else "UNMASKED_FALLBACK"
            code = (
                "REQUIRED_SEGMENTATION_UNAVAILABLE"
                if blocked
                else "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES"
            )
            base_report |= {
                "status": "BLOCKED" if blocked else "UNAVAILABLE_FALLBACK",
                "provider": "UNAVAILABLE",
                "masking_decision": decision,
                "comparison": unavailable_reason,
                "mean_excluded_fraction": None,
            }
            report = _write_report(run_dir, base_report)
            update_masking_decision(run_dir, decision, unavailable_reason)
            return [report], [{"code": code, "message": unavailable_reason}]

    masks_dir = run_dir / "masks"
    reconstruction_masks = masks_dir / "reconstruction"
    reconstruction_masks.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    fractions: list[float] = []
    try:
        for frame in keyframes:
            if not frame.get("selected", True):
                continue
            image_name = Path(str(frame.get("image_name", ""))).name
            image_path = run_dir / "frames" / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read selected frame: {image_name}")
            excluded = provider(image_path)
            if excluded.ndim != 2:
                raise ValueError("Segmentation provider must return one 2D mask per frame")
            if excluded.shape != image.shape[:2]:
                excluded = cv2.resize(
                    excluded.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            excluded = np.where(excluded > 0, 255, 0).astype(np.uint8)
            included = cv2.bitwise_not(excluded)
            fraction = float(np.count_nonzero(excluded) / excluded.size)
            if fraction >= 0.98:
                raise ValueError(f"Mask for {image_name} excludes at least 98% of the image")
            if reconstruction_target == "PRIMARY_SUBJECT" and fraction <= 0.01:
                raise ValueError(
                    f"Primary-subject mask for {image_name} does not isolate a subject"
                )

            review_path = masks_dir / f"{Path(image_name).stem}_dynamic.png"
            colmap_path = reconstruction_masks / f"{image_name}.png"
            openmvs_path = reconstruction_masks / f"{image_name}.mask.png"
            for path, mask in (
                (review_path, excluded),
                (colmap_path, included),
                (openmvs_path, included),
            ):
                if not cv2.imwrite(str(path), mask):
                    raise OSError(f"Could not write mask: {path.name}")
                artifacts.append(path)
            frame["dynamic_mask_fraction"] = fraction
            frame["mask_url"] = f"/api/runs/{run_id}/artifacts/masks/{review_path.name}"
            frame["mask_semantics"] = "NONZERO_IS_EXCLUDED"
            fractions.append(fraction)
    except (OSError, ValueError, RuntimeError) as exc:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        for frame in keyframes:
            frame.pop("dynamic_mask_fraction", None)
            frame.pop("mask_url", None)
            frame.pop("mask_semantics", None)
        blocked = masking_mode == "REQUIRED" or reconstruction_target == "PRIMARY_SUBJECT"
        decision = "BLOCKED_REQUIRED_MASK_FAILED" if blocked else "UNMASKED_FALLBACK"
        message = f"Optional segmentation failed safely: {exc}"
        base_report |= {
            "status": "BLOCKED" if blocked else "FAILED_FALLBACK",
            "provider": "SEGMENTATION_PROVIDER",
            "masking_decision": decision,
            "comparison": message,
            "mean_excluded_fraction": None,
        }
        report = _write_report(run_dir, base_report)
        update_masking_decision(run_dir, decision, message)
        code = "REQUIRED_SEGMENTATION_FAILED" if blocked else "SEGMENTATION_FAILED_FALLBACK"
        return [report], [{"code": code, "message": message}]

    base_report |= {
        "status": "APPLIED",
        "provider": (
            "INJECTED_TEST_PROVIDER" if model_path is None else "YOLO_SEGMENTATION_LOCAL_WEIGHTS"
        ),
        "masking_decision": "APPLIED",
        "mean_excluded_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "minimum_excluded_fraction": float(np.min(fractions)) if fractions else 0.0,
        "maximum_excluded_fraction": float(np.max(fractions)) if fractions else 0.0,
        "operational_mask_directory": "masks/reconstruction",
        "comparison": (
            "Declared masks are consumed consistently by sparse features, OpenMVS dense "
            "matching, and OpenMVS texturing; original source frames remain unchanged."
        ),
    }
    report = _write_report(run_dir, base_report)
    update_masking_decision(
        run_dir,
        "APPLIED",
        f"Generated complete operational masks for {len(fractions)} selected frames.",
    )
    return [*artifacts, report], []
