"""Optional scene-aware segmentation with a no-download, fail-honestly contract."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

from .process_control import ProcessCancelledError, ProcessTimeoutError
from .scene_policy import update_masking_decision
from .storage import atomic_json, sha256_file

MaskProvider = Callable[[Path], np.ndarray]
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
SKY_CLASSES = {"sky"}


@dataclass(frozen=True)
class SegmentationSettings:
    """CPU defaults; accelerator use requires the caller's scheduler allocation."""

    device: str = "cpu"
    image_size: int = 640
    confidence: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 100
    max_frames: int = 1000
    timeout_s: float = 300.0
    review_sample_size: int = 6
    mask_dilation_px: int = 0
    mask_erosion_px: int = 0
    excluded_classes: tuple[str, ...] = tuple(sorted(DYNAMIC_CLASSES | SKY_CLASSES))

    def __post_init__(self) -> None:
        if not self.device or self.device == "auto":
            raise ValueError("Select an explicit inference device")
        if not 32 <= self.image_size <= 4096 or self.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 between 32 and 4096")
        if (
            not 0 < self.confidence <= 1
            or not 0 < self.iou_threshold <= 1
            or not 1 <= self.max_detections <= 1000
        ):
            raise ValueError("Invalid confidence or detection limit")
        if (
            not 1 <= self.max_frames <= 1000
            or not math.isfinite(self.timeout_s)
            or not 0 < self.timeout_s <= 3600
        ):
            raise ValueError("Invalid frame limit or inference deadline")
        if not 0 <= self.review_sample_size <= 12:
            raise ValueError("review_sample_size must be between 0 and 12")
        if not 0 <= self.mask_dilation_px <= 64 or not 0 <= self.mask_erosion_px <= 64:
            raise ValueError("Mask morphology radii must be between 0 and 64 pixels")
        normalized = tuple(item.strip().lower() for item in self.excluded_classes)
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("excluded_classes must contain unique non-empty names")


def _versions() -> dict[str, str | None]:
    result = {}
    for name in ("ultralytics", "torch", "opencv-python-headless", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def segmentation_fingerprint_inputs(
    model_path: str | None, settings: SegmentationSettings | None = None
) -> dict[str, object]:
    """Jay's PREPROCESS integration must hash this resolved dependency record."""
    configured = model_path or os.environ.get("SIH_SEGMENTATION_MODEL")
    candidate = Path(configured).expanduser().resolve() if configured else None
    return {
        "contract": "segmentation-2.1",
        "model_source": "ARGUMENT" if model_path else ("ENVIRONMENT" if configured else "NONE"),
        "resolved_model_path": str(candidate) if candidate else None,
        "model_sha256": sha256_file(candidate) if candidate and candidate.is_file() else None,
        "settings": asdict(settings or SegmentationSettings()),
        "versions": _versions(),
        "excluded_classes": sorted((settings or SegmentationSettings()).excluded_classes),
    }


def _ultralytics_provider(
    model_path: Path, reconstruction_target: str, settings: SegmentationSettings
) -> MaskProvider:
    # Disable Ultralytics dependency auto-install before importing its settings.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional segmentation dependency group") from exc
    if model_path.suffix.lower() != ".pt" or not model_path.is_file():
        raise ValueError("Supply an existing local .pt segmentation checkpoint")
    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError("Local checkpoint is not a segmentation model")
    names = model.names.values() if isinstance(model.names, dict) else model.names
    metadata = {
        "actual_device": None,
        "supported_classes": sorted(str(name).lower() for name in names),
    }

    def segment(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"Could not read selected frame: {image_path.name}")
        candidates: list[tuple[str, np.ndarray]] = []
        results = model.predict(
            source=image,
            verbose=False,
            device=settings.device,
            imgsz=settings.image_size,
            conf=settings.confidence,
            iou=settings.iou_threshold,
            max_det=settings.max_detections,
            retina_masks=True,
            save=False,
            stream=False,
        )
        metadata["actual_device"] = str(model.predictor.device)
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            for class_id, tensor in zip(
                result.boxes.cls.tolist(), result.masks.data.cpu().numpy(), strict=True
            ):
                resized = cv2.resize(
                    (tensor >= 0.5).astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
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
            if class_name in settings.excluded_classes:
                excluded[mask] = 255
        return excluded

    segment.execution_metadata = metadata
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
    settings: SegmentationSettings | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    accelerator_authorized: bool = False,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Create review and reconstruction masks, or record an explicit fallback/blocker.

    A provider returns an exclusion mask: non-zero pixels are excluded. Operational masks use
    the inverse convention required by COLMAP/OpenMVS (zero excluded, 255 included).
    """

    started = time.monotonic()
    settings = settings or SegmentationSettings()
    if masking_mode not in {"OFF", "AUTO", "REQUIRED"}:
        raise ValueError("Unknown masking mode")
    if reconstruction_target not in {"FULL_SCENE", "PRIMARY_SUBJECT"}:
        raise ValueError("Unknown reconstruction target")
    if masking_mode == "OFF" and reconstruction_target == "PRIMARY_SUBJECT":
        raise ValueError("PRIMARY_SUBJECT requires masking")
    if settings.device != "cpu" and not accelerator_authorized:
        raise ValueError("Accelerator inference requires an explicit scheduler allocation")

    def check_control() -> None:
        if cancel_requested and cancel_requested():
            raise ProcessCancelledError("Segmentation cancellation requested")
        if time.monotonic() - started >= settings.timeout_s:
            raise ProcessTimeoutError("Segmentation exceeded its cooperative deadline")

    def clear_outputs() -> None:
        for frame in keyframes:
            name = Path(str(frame.get("image_name", ""))).name
            if name:
                for relative in (
                    f"masks/{Path(name).stem}_dynamic.png",
                    f"masks/reconstruction/{name}.png",
                    f"masks/reconstruction/{name}.mask.png",
                ):
                    (run_dir / relative).unlink(missing_ok=True)
            for key in ("dynamic_mask_fraction", "mask_url", "mask_semantics"):
                frame.pop(key, None)
        for name in (
            "segmentation_contact_sheet.jpg",
            "segmentation_review.json",
            "segmentation_comparison.json",
        ):
            (run_dir / name).unlink(missing_ok=True)

    clear_outputs()
    selected_count = sum(bool(frame.get("selected", True)) for frame in keyframes)
    base_report: dict[str, object] = {
        "schema_version": "2.1",
        "reconstruction_target": reconstruction_target,
        "masking_mode": masking_mode,
        "mask_semantics": {
            "review_masks": "NONZERO_IS_EXCLUDED",
            "reconstruction_masks": "ZERO_IS_EXCLUDED",
        },
        "execution": {
            "requested_provider": "INJECTED_PROVIDER"
            if provider
            else "YOLO_SEGMENTATION_LOCAL_WEIGHTS",
            "executed_provider": None,
            "requested_device": settings.device,
            "actual_device": None,
            "parameters": asdict(settings),
            "versions": _versions(),
            "cache_classification": "FRESH_CALL",
            "timeout_enforcement": "COOPERATIVE_BETWEEN_CALLS; process isolation required for hard deadlines",
        },
        "limitations": [
            "Class-based exclusion does not prove object motion; parked vehicles may be excluded.",
            "Animals are not excluded; sky is excluded only if the checkpoint supports sky.",
            "Mask fractions and this report are not reconstruction A/B or accuracy evidence.",
        ],
        "dynamic_classes": sorted(DYNAMIC_CLASSES),
        "sky_classes_when_model_supports_them": sorted(SKY_CLASSES),
        "selected_frame_count": selected_count,
        "downstream_consumption_verified": False,
        "masks_applied_to": [
            "COLMAP_SPARSE_FEATURE_EXTRACTION",
            "OPENMVS_DENSE_MATCHING",
            "OPENMVS_TEXTURE_GENERATION",
        ],
    }
    if masking_mode == "OFF":
        base_report |= {
            "status": "DISABLED",
            "runtime_s": time.monotonic() - started,
            "masking_decision": "UNMASKED_BY_CONFIGURATION",
            "comparison": "The operator explicitly disabled masking.",
        }
        report = _write_report(run_dir, base_report)
        update_masking_decision(run_dir, "UNMASKED_BY_CONFIGURATION", "Masking mode is OFF.")
        return [report], []

    execution = base_report["execution"]
    provider_name = "INJECTED_PROVIDER" if provider else "YOLO_SEGMENTATION_LOCAL_WEIGHTS"
    if provider is None:
        resolved = segmentation_fingerprint_inputs(model_path, settings)
        execution.update(resolved)
        candidate = (
            Path(resolved["resolved_model_path"]) if resolved["resolved_model_path"] else None
        )
        unavailable_reason = None
        if candidate is None or not candidate.is_file():
            unavailable_reason = "No local segmentation weights were supplied; no model was downloaded automatically."
        else:
            try:
                check_control()
                provider = _ultralytics_provider(candidate, reconstruction_target, settings)
                check_control()
            except (ProcessCancelledError, ProcessTimeoutError):
                raise
            except Exception as exc:  # noqa: BLE001 - optional third-party checkpoint boundary
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
                "runtime_s": time.monotonic() - started,
                "masking_decision": decision,
                "comparison": unavailable_reason,
                "mean_excluded_fraction": None,
            }
            report = _write_report(run_dir, base_report)
            update_masking_decision(run_dir, decision, unavailable_reason)
            return [report], [{"code": code, "message": unavailable_reason}]

    else:
        execution.update(
            {"model_source": "INJECTED", "resolved_model_path": None, "model_sha256": None}
        )
    masks_dir = run_dir / "masks"
    reconstruction_masks = masks_dir / "reconstruction"
    reconstruction_masks.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    fractions: list[float] = []
    review_rows: list[np.ndarray] = []
    review_samples: list[dict[str, object]] = []
    try:
        if not 0 < selected_count <= settings.max_frames:
            raise ValueError("Selected frame count is empty or exceeds the inference limit")
        names = [str(f.get("image_name", "")) for f in keyframes if f.get("selected", True)]
        if any(not n or Path(n).name != n for n in names):
            raise ValueError("Selected image names must be local basenames")
        if len({Path(n).stem for n in names}) != len(names):
            raise ValueError("Selected image stems must be unique for review masks")
        for frame in keyframes:
            if not frame.get("selected", True):
                continue
            image_name = Path(str(frame.get("image_name", ""))).name
            image_path = run_dir / "frames" / image_name
            check_control()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise ValueError(f"Could not read selected frame: {image_name}")
            execution["executed_provider"] = provider_name
            excluded = np.asarray(provider(image_path))
            execution.update(getattr(provider, "execution_metadata", {}))
            check_control()
            if excluded.ndim != 2 or excluded.size == 0 or not np.isfinite(excluded).all():
                raise ValueError("Segmentation provider must return one 2D mask per frame")
            excluded = np.where(excluded != 0, 255, 0).astype(np.uint8)
            if excluded.shape != image.shape[:2]:
                excluded = cv2.resize(
                    excluded.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            excluded = np.where(excluded > 0, 255, 0).astype(np.uint8)
            if settings.mask_dilation_px:
                radius = settings.mask_dilation_px
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                )
                excluded = cv2.dilate(excluded, kernel)
            if settings.mask_erosion_px:
                radius = settings.mask_erosion_px
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                )
                excluded = cv2.erode(excluded, kernel)
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
            if len(review_rows) < settings.review_sample_size:
                # Resize all three panels identically; masks stay binary during resizing.
                size = (320, max(1, round(image.shape[0] * 320 / image.shape[1])))
                small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(excluded, size, interpolation=cv2.INTER_NEAREST)
                overlay = small.copy()
                overlay[mask != 0] = (
                    0.55 * small[mask != 0] + 0.45 * np.array([0, 0, 255])
                ).astype(np.uint8)
                panels = np.hstack((small, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay))
                header = np.zeros((30, panels.shape[1], 3), np.uint8)
                cv2.putText(
                    header,
                    f"{image_name}: source | excluded mask | overlay",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                review_rows.append(np.vstack((header, panels)))
                review_samples.append(
                    {
                        "image_name": image_name,
                        "image_sha256": sha256_file(image_path),
                        "excluded_fraction": fraction,
                        "mask_sha256": sha256_file(review_path),
                        "review_status": "UNREVIEWED",
                        "missed_objects": None,
                        "over_masked_static_surfaces": None,
                        "reviewer": None,
                    }
                )
        if review_rows:
            sheet = run_dir / "segmentation_contact_sheet.jpg"
            if not cv2.imwrite(str(sheet), np.vstack(review_rows)):
                raise OSError("Could not write segmentation contact sheet")
            artifacts.append(sheet)
            review = run_dir / "segmentation_review.json"
            atomic_json(
                review,
                {
                    "schema_version": "1.0",
                    "status": "AWAITING_MANUAL_REVIEW",
                    "samples": review_samples,
                },
            )
            artifacts.append(review)
    except Exception as exc:  # Cleanup is required even for unexpected provider failures.
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        clear_outputs()
        if isinstance(exc, (ProcessCancelledError, ProcessTimeoutError)):
            raise
        blocked = masking_mode == "REQUIRED" or reconstruction_target == "PRIMARY_SUBJECT"
        decision = "BLOCKED_REQUIRED_MASK_FAILED" if blocked else "UNMASKED_FALLBACK"
        message = f"Optional segmentation failed safely: {exc}"
        base_report |= {
            "status": "BLOCKED" if blocked else "FAILED_FALLBACK",
            "provider": provider_name,
            "runtime_s": time.monotonic() - started,
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
        "provider": provider_name,
        "runtime_s": time.monotonic() - started,
        "manual_review_status": "AWAITING_MANUAL_REVIEW" if review_rows else "NOT_REQUESTED",
        "masking_decision": "APPLIED",
        "mean_excluded_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "minimum_excluded_fraction": float(np.min(fractions)) if fractions else 0.0,
        "maximum_excluded_fraction": float(np.max(fractions)) if fractions else 0.0,
        "operational_mask_directory": "masks/reconstruction",
        "comparison": (
            "Masks have been generated for the existing sparse/OpenMVS consumers. "
            "This call does not verify downstream reconstruction or an A/B comparison."
        ),
    }
    report = _write_report(run_dir, base_report)
    update_masking_decision(
        run_dir,
        "APPLIED",
        f"Generated complete operational masks for {len(fractions)} selected frames.",
    )
    return [*artifacts, report], []
