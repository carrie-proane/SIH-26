from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import RunConfig
from .preflight import _telemetry_check, _video_check
from .storage import atomic_json, sha256_file

Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


class DatasetAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def portable_relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if not value.strip() or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "asset path must be a non-empty portable path below the manifest directory"
            )
        return candidate.as_posix()


class CameraCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_model: str
    parameters: str | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    coordinate_reference: Literal["SOURCE_VIDEO", "PROCESSED_FRAMES"]
    source: str
    stated_accuracy: str | None = None


class CaptureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_make: str | None = None
    device_model: str | None = None
    recording_mode: str | None = None
    calibration: CameraCalibration | None = None


class TelemetryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_basis: Literal["VIDEO_START_RELATIVE", "UTC_ABSOLUTE", "REBASING_REQUIRED", "UNKNOWN"]
    altitude_reference: Literal["ELLIPSOIDAL", "ORTHOMETRIC", "RELATIVE_TO_LAUNCH", "UNKNOWN"]


class CoordinateTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_frame: str
    target_frame: str
    output_units: Literal["m"] = "m"
    matrix_4x4: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    description: str
    fitted_using_checkpoint_ids: list[str] = Field(default_factory=list)

    @field_validator("matrix_4x4")
    @classmethod
    def finite_matrix(cls, value: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        if not all(math.isfinite(item) for row in value for item in row):
            raise ValueError("coordinate transform must contain only finite values")
        if tuple(value[3]) != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("coordinate transform must be an affine 4x4 matrix")
        a, b, c = value[0][:3]
        d, e, f = value[1][:3]
        g, h, i = value[2][:3]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(determinant) <= 1e-15:
            raise ValueError("coordinate transform must have an invertible 3x3 linear component")
        return value


class ReferenceCoordinateSystem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: str
    units: Literal["m", "cm", "mm", "degrees"]
    axis_convention: str
    vertical_datum: str | None = None
    origin_wgs84: tuple[float, float, float] | None = None
    reference_to_reconstruction_transform: CoordinateTransform | None = None


class RelativeDistanceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    distance_m: float = Field(gt=0)
    acquisition_method: str
    stated_accuracy_m: float | None = Field(default=None, gt=0)
    role: Literal["SCALE_CONTROL", "HELD_OUT_EVALUATION"]
    evidence: str | None = None


class PositionalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    coordinates: tuple[float, float, float]
    coordinate_frame: str
    units: Literal["m", "cm", "mm", "degrees"]
    altitude_reference: str
    acquisition_method: str
    stated_accuracy_m: float | None = Field(default=None, gt=0)
    role: Literal["SCALE_CONTROL", "HELD_OUT_EVALUATION"]

    @field_validator("coordinates")
    @classmethod
    def finite_coordinates(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("checkpoint coordinates must be finite")
        return value


class ReferenceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_system: ReferenceCoordinateSystem | None = None
    relative_distances: list[RelativeDistanceReference] = Field(default_factory=list)
    positional_checkpoints: list[PositionalCheckpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_reference_ids(self) -> ReferenceData:
        identifiers = [item.id for item in self.relative_distances]
        identifiers.extend(item.id for item in self.positional_checkpoints)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("reference IDs must be unique across the dataset")
        if self.positional_checkpoints and self.coordinate_system is None:
            raise ValueError("reference coordinate_system is required for positional checkpoints")
        if self.coordinate_system is not None:
            mismatched_frames = [
                item.id
                for item in self.positional_checkpoints
                if item.coordinate_frame != self.coordinate_system.frame
            ]
            mismatched_units = [
                item.id
                for item in self.positional_checkpoints
                if item.units != self.coordinate_system.units
            ]
            if mismatched_frames:
                raise ValueError(
                    f"checkpoint frames differ from reference coordinate system: {mismatched_frames}"
                )
            if mismatched_units:
                raise ValueError(
                    f"checkpoint units differ from reference coordinate system: {mismatched_units}"
                )
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    dataset_role: Literal["DEVELOPMENT", "VALIDATION", "HELD_OUT"]
    description: str = ""
    synthetic_example: bool = False
    video: DatasetAsset
    telemetry: DatasetAsset
    benchmark_config: DatasetAsset
    capture: CaptureMetadata
    telemetry_metadata: TelemetryMetadata
    references: ReferenceData = Field(default_factory=ReferenceData)


class ReconstructedCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    reconstructed_coordinates: tuple[float, float, float]

    @field_validator("reconstructed_coordinates")
    @classmethod
    def finite_coordinates(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("reconstructed checkpoint coordinates must be finite")
        return value


class CheckpointCorrespondenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    reconstruction_coordinate_frame: str
    units: Literal["m"] = "m"
    altitude_reference: str
    checkpoints: list[ReconstructedCheckpoint]

    @model_validator(mode="after")
    def unique_checkpoint_ids(self) -> CheckpointCorrespondenceSet:
        identifiers = [item.checkpoint_id for item in self.checkpoints]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("checkpoint correspondences must use unique checkpoint IDs")
        return self


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path)
    return DatasetManifest.model_validate_json(source.read_text(encoding="utf-8"))


def resolve_dataset_asset(manifest_path: str | Path, asset: DatasetAsset) -> Path:
    root = Path(manifest_path).resolve().parent
    resolved = (root / asset.path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Dataset asset escapes manifest directory: {asset.path}")
    return resolved


def _check(check_id: str, status: str, summary: str, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "summary": summary, "details": details}


def _asset_check(
    manifest_path: Path, role: str, asset: DatasetAsset
) -> tuple[dict[str, Any], Path]:
    path = resolve_dataset_asset(manifest_path, asset)
    if not path.is_file():
        return (
            _check(
                f"asset.{role}",
                "BLOCKED",
                f"Declared {role} asset does not exist",
                relative_path=asset.path,
                expected_sha256=asset.sha256,
            ),
            path,
        )
    actual = sha256_file(path)
    status = "PASS" if actual == asset.sha256 else "BLOCKED"
    return (
        _check(
            f"asset.{role}",
            status,
            f"Declared {role} hash matches"
            if status == "PASS"
            else f"Declared {role} hash does not match",
            relative_path=asset.path,
            expected_sha256=asset.sha256,
            actual_sha256=actual,
            size_bytes=path.stat().st_size,
        ),
        path,
    )


def prepare_dataset(
    manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    runner: Runner = subprocess.run,
    which: Which = shutil.which,
) -> dict[str, Any]:
    """Validate a real-data bundle without probing or starting reconstruction tools."""

    source = Path(manifest_path).resolve()
    try:
        manifest = load_dataset_manifest(source)
    except (OSError, ValidationError, ValueError) as exc:
        try:
            manifest_hash = sha256_file(source) if source.is_file() else None
        except OSError:
            manifest_hash = None
        report = {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_id": None,
            "dataset_role": None,
            "synthetic_example": None,
            "manifest_path": str(source),
            "manifest_sha256": manifest_hash,
            "status": "BLOCKED",
            "ready_for_explicit_execution": False,
            "accuracy_claim_status": "UNVERIFIED_INVALID_MANIFEST",
            "reconstruction_started": False,
            "reconstruction_tools_checked": False,
            "blockers": ["manifest.schema"],
            "warnings": [],
            "checks": [
                _check(
                    "manifest.schema",
                    "BLOCKED",
                    "Dataset manifest is missing, unreadable, or violates schema 1.0",
                    error=str(exc),
                    action=(
                        "Start from datasets/templates/real-dataset.manifest.json and declare "
                        "all required paths, hashes, units, frames, and acquisition metadata."
                    ),
                )
            ],
            "declared_assets": {},
            "limitations": ["No reconstruction was started."],
        }
        if output is not None:
            atomic_json(Path(output), report)
        return report
    checks: list[dict[str, Any]] = []
    assets: dict[str, Path] = {}
    for role, asset in (
        ("video", manifest.video),
        ("telemetry", manifest.telemetry),
        ("benchmark_config", manifest.benchmark_config),
    ):
        check, path = _asset_check(source, role, asset)
        checks.append(check)
        assets[role] = path

    ffprobe = which("ffprobe")
    video_check, video_duration = _video_check(
        assets["video"] if assets["video"].is_file() else None,
        ffprobe,
        runner=runner,
    )
    if ffprobe is None and assets["video"].is_file():
        video_check["details"]["action"] = (
            "Install ffprobe/FFmpeg; reconstruction tools are not required."
        )
    telemetry_check, telemetry_duration = _telemetry_check(
        assets["telemetry"] if assets["telemetry"].is_file() else None
    )
    checks.extend((video_check, telemetry_check))

    if video_duration is not None and telemetry_duration is not None:
        difference = abs(video_duration - telemetry_duration)
        tolerance = max(2.0, video_duration * 0.2)
        checks.append(
            _check(
                "input.duration_coverage",
                "PASS" if difference <= tolerance else "WARN",
                "Telemetry plausibly covers the video duration"
                if difference <= tolerance
                else "Telemetry/video duration mismatch requires synchronization review",
                video_duration_s=video_duration,
                telemetry_duration_s=telemetry_duration,
                absolute_difference_s=difference,
                tolerance_s=tolerance,
            )
        )

    config: RunConfig | None = None
    if assets["benchmark_config"].is_file():
        try:
            config = RunConfig.model_validate_json(
                assets["benchmark_config"].read_text(encoding="utf-8")
            )
            incompatible = config.execution_mode != "COLMAP"
            checks.append(
                _check(
                    "benchmark.configuration",
                    "BLOCKED" if incompatible else "PASS",
                    "Benchmark configuration is compatible with real execution"
                    if not incompatible
                    else "Real datasets cannot use SYNTHETIC_DEMO execution",
                    profile=config.profile,
                    execution_mode=config.execution_mode,
                    enable_dense_reconstruction=config.enable_dense_reconstruction,
                )
            )
        except (OSError, ValidationError, ValueError) as exc:
            checks.append(
                _check(
                    "benchmark.configuration",
                    "BLOCKED",
                    "Benchmark configuration is invalid",
                    error=str(exc),
                )
            )

    calibration = manifest.capture.calibration
    missing_capture_fields = [
        name
        for name in ("device_make", "device_model", "recording_mode")
        if not getattr(manifest.capture, name)
    ]
    if missing_capture_fields:
        checks.append(
            _check(
                "capture.metadata",
                "WARN",
                "Capture device/mode metadata is incomplete",
                missing_fields=missing_capture_fields,
                implication="Reproducibility and calibration applicability require manual review.",
            )
        )
    else:
        checks.append(
            _check("capture.metadata", "PASS", "Capture device and recording mode are declared")
        )
    if calibration is None:
        checks.append(
            _check(
                "capture.calibration",
                "WARN",
                "No independent camera calibration was supplied; intrinsics remain uncertain",
            )
        )
    elif config is not None:
        details = video_check.get("details", {})
        source_size_matches = calibration.coordinate_reference != "SOURCE_VIDEO" or (
            calibration.image_width == details.get("width")
            and calibration.image_height == details.get("height")
        )
        model_matches = calibration.camera_model == config.camera_model
        reference_matches = (
            calibration.parameters is None
            or config.camera_params is None
            or calibration.coordinate_reference == config.camera_params_reference
        )
        params_match = (
            calibration.parameters is None
            or config.camera_params is None
            or calibration.parameters.strip() == config.camera_params.strip()
        )
        compatible = source_size_matches and model_matches and reference_matches and params_match
        checks.append(
            _check(
                "capture.calibration",
                "PASS" if compatible else "BLOCKED",
                "Calibration and benchmark camera settings are compatible"
                if compatible
                else "Calibration conflicts with video metadata or benchmark camera settings",
                calibration=calibration.model_dump(mode="json"),
                benchmark_camera_model=config.camera_model,
                benchmark_camera_params_reference=config.camera_params_reference,
                source_video_dimensions=[details.get("width"), details.get("height")],
            )
        )
        if calibration.parameters is not None and config.camera_params is None:
            checks.append(
                _check(
                    "capture.calibration_usage",
                    "WARN",
                    "A calibration is declared but its parameters are not configured for execution",
                    implication="The run may estimate intrinsics instead of using the supplied calibration.",
                )
            )

    if config is not None and video_duration is not None:
        representative = 595 <= video_duration <= 605
        profile_compatible = (
            config.profile == "accurate" if representative else config.profile != "accurate"
        )
        checks.append(
            _check(
                "benchmark.workload_profile",
                "PASS" if profile_compatible else "WARN",
                "Benchmark profile matches the declared workload phase"
                if profile_compatible
                else (
                    "A representative ten-minute workload should use the approved accurate profile"
                    if representative
                    else "An accurate profile on a short input cannot validate the ten-minute target"
                ),
                duration_s=video_duration,
                profile=config.profile,
                representative_ten_minute_input=representative,
            )
        )

    if manifest.telemetry_metadata.time_basis in {"UNKNOWN", "REBASING_REQUIRED"}:
        checks.append(
            _check(
                "telemetry.time_basis",
                "WARN",
                "Telemetry time basis requires explicit review before alignment",
                declared=manifest.telemetry_metadata.time_basis,
            )
        )
    else:
        checks.append(
            _check(
                "telemetry.time_basis",
                "PASS",
                "Telemetry time basis is declared",
                declared=manifest.telemetry_metadata.time_basis,
            )
        )
    if manifest.telemetry_metadata.altitude_reference == "UNKNOWN":
        checks.append(
            _check(
                "telemetry.altitude_reference",
                "WARN",
                "Telemetry vertical datum is unknown; vertical/geolocated accuracy is unavailable",
            )
        )

    heldout_distance = sum(
        item.role == "HELD_OUT_EVALUATION" for item in manifest.references.relative_distances
    )
    heldout_positions = sum(
        item.role == "HELD_OUT_EVALUATION" for item in manifest.references.positional_checkpoints
    )
    if not heldout_distance and not heldout_positions:
        checks.append(
            _check(
                "references.independent_evaluation",
                "WARN",
                "No held-out reference is supplied; reconstruction may run but accuracy remains unverified",
                held_out_relative_distances=heldout_distance,
                held_out_positional_checkpoints=heldout_positions,
            )
        )
    else:
        checks.append(
            _check(
                "references.independent_evaluation",
                "PASS",
                "Held-out references are declared for post-reconstruction evaluation",
                held_out_relative_distances=heldout_distance,
                held_out_positional_checkpoints=heldout_positions,
            )
        )
    heldout_references = [
        item
        for item in [
            *manifest.references.relative_distances,
            *manifest.references.positional_checkpoints,
        ]
        if item.role == "HELD_OUT_EVALUATION"
    ]
    missing_reference_accuracy = [
        item.id for item in heldout_references if item.stated_accuracy_m is None
    ]
    if missing_reference_accuracy:
        checks.append(
            _check(
                "references.stated_accuracy",
                "WARN",
                "Some held-out references have no stated acquisition accuracy",
                checkpoint_or_measurement_ids=missing_reference_accuracy,
                implication=(
                    "Residual statistics remain computable, but reference uncertainty cannot be "
                    "separated from reconstruction error."
                ),
            )
        )

    coordinate_system = manifest.references.coordinate_system
    if coordinate_system is not None and coordinate_system.units == "degrees":
        checks.append(
            _check(
                "references.coordinate_frame",
                "WARN",
                "Angular reference coordinates require an explicit metric transform before positional evaluation",
                frame=coordinate_system.frame,
                units=coordinate_system.units,
            )
        )
    if (
        coordinate_system is not None
        and "ENU" in coordinate_system.frame.upper()
        and coordinate_system.origin_wgs84 is None
    ):
        checks.append(
            _check(
                "references.local_origin",
                "WARN",
                "The local ENU reference frame has no declared WGS84 origin",
                implication="Absolute geolocation and frame reproduction require the local origin.",
            )
        )

    blockers = [item["id"] for item in checks if item["status"] == "BLOCKED"]
    warnings = [item["id"] for item in checks if item["status"] == "WARN"]
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_id": manifest.dataset_id,
        "dataset_role": manifest.dataset_role,
        "synthetic_example": manifest.synthetic_example,
        "manifest_path": str(source),
        "manifest_sha256": sha256_file(source),
        "status": "BLOCKED" if blockers else ("READY_WITH_WARNINGS" if warnings else "READY"),
        "ready_for_explicit_execution": not blockers and not manifest.synthetic_example,
        "accuracy_claim_status": (
            "REFERENCE_AVAILABLE"
            if heldout_distance or heldout_positions
            else "UNVERIFIED_NO_HELD_OUT_REFERENCE"
        ),
        "reconstruction_started": False,
        "reconstruction_tools_checked": False,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "declared_assets": {
            role: {
                "relative_path": getattr(
                    manifest, role if role != "benchmark_config" else "benchmark_config"
                ).path,
                "sha256": getattr(
                    manifest, role if role != "benchmark_config" else "benchmark_config"
                ).sha256,
            }
            for role in ("video", "telemetry", "benchmark_config")
        },
        "limitations": [
            "Preparation validates declared inputs; it does not run reconstruction or establish accuracy.",
            "Missing held-out references do not block reconstruction, but do block an accuracy-pass claim.",
            "Large-scene runtime and memory behavior require execution on the target server.",
        ],
    }
    if output is not None:
        atomic_json(Path(output), report)
    return report


def load_correspondences(path: str | Path) -> CheckpointCorrespondenceSet:
    return CheckpointCorrespondenceSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
