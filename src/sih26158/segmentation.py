"""Optional dynamic-object segmentation with a no-download, fail-open contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

MaskProvider = Callable[[Path], np.ndarray]
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def _ultralytics_provider(model_path: Path) -> MaskProvider:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional segmentation dependency group") from exc
    model = YOLO(str(model_path))

    def segment(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read selected frame: {image_path.name}")
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        results = model.predict(source=str(image_path), verbose=False)
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            names = result.names
            for class_id, tensor in zip(
                result.boxes.cls.tolist(), result.masks.data.cpu().numpy(), strict=True
            ):
                if str(names[int(class_id)]).lower() not in DYNAMIC_CLASSES:
                    continue
                resized = cv2.resize(
                    tensor.astype(np.float32),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                combined[resized >= 0.5] = 255
        return combined

    return segment


def run_optional_segmentation(
    run_dir: Path,
    run_id: str,
    keyframes: list[dict[str, object]],
    model_path: str | None,
    *,
    provider: MaskProvider | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Create masks when explicitly configured; otherwise return an honest warning."""
    warnings: list[dict[str, str]] = []
    if provider is None:
        candidate = Path(model_path or "")
        if not model_path or not candidate.is_file():
            return [], [
                {
                    "code": "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES",
                    "message": "Configured local YOLO segmentation weights were not found.",
                }
            ]
        try:
            provider = _ultralytics_provider(candidate)
        except (RuntimeError, OSError) as exc:
            return [], [
                {
                    "code": "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES",
                    "message": str(exc),
                }
            ]

    masks_dir = run_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    artifacts: list[Path] = []
    fractions: list[float] = []
    try:
        for frame in keyframes:
            if not frame.get("selected", True):
                continue
            image_name = Path(str(frame.get("image_name", ""))).name
            image_path = run_dir / "frames" / image_name
            mask = provider(image_path)
            if mask.ndim != 2:
                raise ValueError("Segmentation provider must return one 2D mask per frame")
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            mask_path = masks_dir / f"{Path(image_name).stem}_dynamic.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Could not write mask: {mask_path.name}")
            fraction = float(np.count_nonzero(mask) / mask.size)
            frame["dynamic_mask_fraction"] = fraction
            frame["mask_url"] = f"/api/runs/{run_id}/artifacts/masks/{mask_path.name}"
            artifacts.append(mask_path)
            fractions.append(fraction)
    except (OSError, ValueError, RuntimeError) as exc:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        for frame in keyframes:
            frame.pop("dynamic_mask_fraction", None)
            frame.pop("mask_url", None)
        return [], [
            {
                "code": "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES",
                "message": f"Optional segmentation failed safely: {exc}",
            }
        ]

    report_path = run_dir / "segmentation_comparison.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": "YOLO_SEGMENTATION_LOCAL_WEIGHTS",
                "dynamic_classes": sorted(DYNAMIC_CLASSES),
                "sky_segmentation": False,
                "selected_frame_count": len(fractions),
                "mean_dynamic_mask_fraction": float(np.mean(fractions)) if fractions else 0.0,
                "comparison": "COLMAP may use masks when supported; unmasked frames remain available.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*artifacts, report_path], warnings
