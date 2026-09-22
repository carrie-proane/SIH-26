from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    PREPROCESSING = "PREPROCESSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ConfidenceLabel(StrEnum):
    OBSERVED_HIGH = "OBSERVED_HIGH"
    OBSERVED_MEDIUM = "OBSERVED_MEDIUM"
    OBSERVED_LOW = "OBSERVED_LOW"
    AI_ASSISTED_NOT_MEASURABLE = "AI_ASSISTED_NOT_MEASURABLE"
    UNSEEN = "UNSEEN"


class ProvenanceOrigin(StrEnum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"


class OffsetSource(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CALIBRATED = "calibrated"
    NOT_APPLICABLE = "not_applicable"


class InputAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["video", "telemetry", "config", "intrinsics", "ground_truth"]
    original_name: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str | None = None
    origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN


class ProjectManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    name: str
    description: str = ""
    data_classification: Literal["PUBLIC_DEMO", "INTERNAL", "SENSITIVE", "RESTRICTED"] = (
        "PUBLIC_DEMO"
    )
    config_version: str = "1.0"
    created_at: str = Field(default_factory=utc_now)
    immutable: bool = True
    assets: list[InputAsset]
    warnings: list[dict[str, str]] = Field(default_factory=list)
    source_provenance: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN
    video_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN
    telemetry_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str = "1.0"
    profile: Literal["smoke", "preview", "balanced", "accurate", "diagnostic"] = "preview"
    matcher: Literal["SIFT", "SUPERPOINT_LIGHTGLUE"] = "SIFT"
    execution_mode: Literal["COLMAP", "SYNTHETIC_DEMO"] = "COLMAP"
    camera_model: str = "SIMPLE_RADIAL"
    camera_model_policy: Literal["AUTO", "FIXED"] = "AUTO"
    camera_params: str | None = None
    camera_params_reference: Literal["PROCESSED_FRAMES", "SOURCE_VIDEO"] = "PROCESSED_FRAMES"
    refine_intrinsics: bool = True
    max_reconstruction_retries: int = Field(default=1, ge=0, le=1)
    sequential_overlap: int = Field(default=10, ge=1, le=50)
    use_gpu: bool = False
    known_distance_m: float | None = Field(default=None, gt=0)
    measured_distance_m: float | None = Field(default=None, gt=0)
    local_origin: tuple[float, float, float] | None = None
    preprocessing_run: str | None = None
    telemetry_offset_s: float | None = Field(default=None, ge=-5.0, le=5.0)
    telemetry_offset_source: Literal["manual", "calibrated"] | None = None
    force_include_frame_indices: list[int] = Field(default_factory=list)
    force_exclude_frame_indices: list[int] = Field(default_factory=list)
    frame_min_laplacian_variance: float = Field(default=40.0, ge=0)
    frame_min_exposure_score: float = Field(default=0.18, ge=0, le=1)
    frame_relative_sharpness_floor: float = Field(default=0.60, ge=0, le=1)
    frame_min_feature_count: int = Field(default=4, ge=0, le=5000)
    frame_min_feature_grid_coverage: float = Field(default=0.05, ge=0, le=1)
    frame_max_parallax_fraction: float = Field(default=0.30, gt=0, le=1)
    matching_strategy: Literal["AUTO", "SEQUENTIAL", "EXHAUSTIVE"] = "AUTO"
    vocab_tree_path: str | None = None
    enable_segmentation: bool = False
    segmentation_model_path: str | None = None
    segmentation_model_name: str | None = None
    segmentation_model_version: str | None = None
    segmentation_device: Literal["cpu", "cuda", "mps"] = "cpu"
    segmentation_allow_cpu_fallback: bool = False
    segmentation_confidence: float = Field(default=0.25, gt=0, le=1)
    segmentation_iou_threshold: float = Field(default=0.7, gt=0, le=1)
    segmentation_image_size: int = Field(default=640, ge=32, le=4096)
    segmentation_mask_dilation_px: int = Field(default=0, ge=0, le=64)
    segmentation_mask_erosion_px: int = Field(default=0, ge=0, le=64)
    segmentation_excluded_classes: list[str] = Field(
        default_factory=lambda: ["bicycle", "bus", "car", "motorcycle", "person", "sky", "truck"]
    )
    segmentation_timeout_s: float = Field(default=300, gt=0, le=3600)
    reconstruction_target: Literal["FULL_SCENE", "PRIMARY_SUBJECT"] = "FULL_SCENE"
    masking_mode: Literal["OFF", "AUTO", "REQUIRED"] = "OFF"
    enable_dense_reconstruction: bool = False
    dense_provider: Literal["auto", "colmap", "openmvs"] = "auto"
    sparse_timeout_s: float = Field(default=7200, ge=30, le=86400)
    dense_timeout_s: float = Field(default=21600, ge=30, le=172800)
    command_heartbeat_s: float = Field(default=10, ge=1, le=300)
    max_candidate_frames: int = Field(default=240, ge=3, le=5000)
    max_selected_frames: int = Field(default=100, ge=3, le=1000)
    processing_max_image_dimension: int | None = Field(default=None, ge=640, le=16384)
    worker_threads: int = Field(default=0, ge=0, le=256)
    coverage_interval_s: float = Field(default=60.0, gt=0, le=600)
    enable_coverage_analysis: bool = True
    enable_completion: bool = False
    completion_timeout_s: float = Field(default=60, gt=0, le=3600)
    dataset_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    dataset_role: Literal["DEVELOPMENT", "VALIDATION", "HELD_OUT"] | None = None
    dataset_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_benchmark_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("measured_distance_m")
    @classmethod
    def require_reference_for_measurement(cls, value: float | None, info: Any) -> float | None:
        if value is not None and info.data.get("known_distance_m") is None:
            raise ValueError("known_distance_m is required when measured_distance_m is provided")
        return value

    @model_validator(mode="after")
    def validate_manual_offset_source(self) -> RunConfig:
        if self.camera_params is not None and not self.camera_params.strip():
            raise ValueError("camera_params cannot be blank")
        if self.camera_params is not None or not self.refine_intrinsics:
            self.camera_model_policy = "FIXED"
        if self.telemetry_offset_source is not None and self.telemetry_offset_s is None:
            raise ValueError("telemetry_offset_s is required when telemetry_offset_source is set")
        if self.telemetry_offset_s is not None and self.telemetry_offset_source is None:
            self.telemetry_offset_source = "manual"
        for name in ("force_include_frame_indices", "force_exclude_frame_indices"):
            values = getattr(self, name)
            if any(value < 0 for value in values):
                raise ValueError(f"{name} cannot contain negative frame indices")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicate frame indices")
        if set(self.force_include_frame_indices) & set(self.force_exclude_frame_indices):
            raise ValueError("A frame cannot be both force-included and force-excluded")
        if self.reconstruction_target == "PRIMARY_SUBJECT":
            self.enable_segmentation = True
            if self.masking_mode == "OFF":
                raise ValueError("PRIMARY_SUBJECT reconstruction requires AUTO or REQUIRED masking")
        if self.enable_segmentation and self.masking_mode == "OFF":
            self.masking_mode = "AUTO"
        if self.segmentation_image_size % 32:
            raise ValueError("segmentation_image_size must be a multiple of 32")
        normalized_classes = [item.strip().lower() for item in self.segmentation_excluded_classes]
        if any(not item for item in normalized_classes) or len(normalized_classes) != len(
            set(normalized_classes)
        ):
            raise ValueError("segmentation_excluded_classes must be unique non-empty names")
        self.segmentation_excluded_classes = normalized_classes
        if self.max_selected_frames > self.max_candidate_frames:
            raise ValueError("max_selected_frames cannot exceed max_candidate_frames")
        return self


class ArtifactEntry(BaseModel):
    name: str
    relative_path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str | None = None
    created_at: str = Field(default_factory=utc_now)


class StageEvent(BaseModel):
    stage: RunStatus
    status: Literal["STARTED", "COMPLETED", "FAILED", "CANCELLED"]
    timestamp: str = Field(default_factory=utc_now)
    progress: int = Field(ge=0, le=100)
    message: str


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    stage: RunStatus = RunStatus.QUEUED
    status: RunStatus = RunStatus.QUEUED
    progress: int = 0
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    config_version: str = "1.0"
    config: RunConfig
    environment: dict[str, str | None] = Field(default_factory=dict)
    failure_reason: str | None = None
    events: list[StageEvent] = Field(default_factory=list)
    artifacts: list[ArtifactEntry] = Field(default_factory=list)
    synthetic_fixture: bool = False
    source_provenance: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN
    video_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN
    telemetry_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN
    telemetry_offset_s: float = 0.0
    offset_source: OffsetSource = OffsetSource.NOT_APPLICABLE
    rmse_before_m: float | None = Field(default=None, ge=0)
    rmse_after_m: float | None = Field(default=None, ge=0)
    matched_camera_count: int = Field(default=0, ge=0)
    inlier_count: int = Field(default=0, ge=0)
    checkpoint_stage: str | None = None
    last_heartbeat_at: str | None = None
    recovery_count: int = Field(default=0, ge=0)
    cancel_requested_at: str | None = None
    cancelled_at: str | None = None
    capability_profile_path: str | None = None
    effective_sparse_gpu: bool = False
    selected_dense_provider: Literal["colmap", "openmvs"] | None = None
    processing_started_at: str | None = None
    processing_completed_at: str | None = None
    stage_timings_s: dict[str, float] = Field(default_factory=dict)
    derived_from_run_id: str | None = None
    requested_matcher: Literal["SIFT", "SUPERPOINT_LIGHTGLUE"] | None = None
    executed_matcher: Literal["SIFT"] | None = None


class StageCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "INGEST",
        "PREPROCESS",
        "SEGMENTATION",
        "SPARSE",
        "DENSE",
        "COVERAGE",
        "COMPLETION",
        "REPORT",
    ]
    completed_at: str = Field(default_factory=utc_now)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[dict[str, str]] = Field(default_factory=list)
    input_fingerprint: str | None = None
    configuration_fingerprint: str | None = None
    environment_fingerprint: str | None = None
    started_at: str | None = None
    duration_s: float | None = Field(default=None, ge=0)


class RunCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    active_stage: str | None = None
    updated_at: str = Field(default_factory=utc_now)
    completed: dict[str, StageCheckpoint] = Field(default_factory=dict)
    warnings: list[dict[str, str]] = Field(default_factory=list)
    active_stage_started_at: str | None = None


class MeasurementEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinates: tuple[float, float, float]
    point_id: int | None = Field(default=None, ge=0)
    face_id: int | None = Field(default=None, ge=0)

    @field_validator("coordinates")
    @classmethod
    def finite_coordinates(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        import math

        if not all(math.isfinite(component) for component in value):
            raise ValueError("measurement endpoint coordinates must be finite")
        return value


class MeasurementCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geometry_artifact_path: str
    geometry_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start: MeasurementEndpoint
    end: MeasurementEndpoint
    coordinate_frame: str = "LOCAL_ENU_METRES"
    units: Literal["m"] = "m"
    measurement_kind: Literal["DISTANCE_3D", "HORIZONTAL", "VERTICAL", "RELATIVE_DIMENSION"] = (
        "DISTANCE_3D"
    )
    reference_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,120}$")
    reference_value_m: float | None = Field(default=None, gt=0)
    reference_method: str | None = None
    reference_evidence: str | None = None
    reference_role: Literal["NONE", "SCALE_CONTROL", "HELD_OUT_EVALUATION"] = "NONE"

    @model_validator(mode="after")
    def validate_reference(self) -> MeasurementCreate:
        if self.reference_role != "NONE" and self.reference_value_m is None:
            raise ValueError("reference_value_m is required for a reference measurement")
        if self.reference_value_m is not None and self.reference_role == "NONE":
            raise ValueError("reference_role must classify a supplied reference value")
        return self


class MeasurementRecord(MeasurementCreate):
    measurement_id: str
    run_id: str
    created_at: str = Field(default_factory=utc_now)
    backend_distance_m: float = Field(ge=0)
    geometry_provenance: Literal["OBSERVED", "INFERRED", "UNKNOWN"]
    measurement_eligible: bool
    eligibility_reason: str
    legacy_manual_reconstructed_value: bool = False


class PointConfidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: int = Field(ge=0)
    supporting_views: int = Field(ge=0)
    track_length: int = Field(ge=0)
    reprojection_error: float = Field(ge=0)
    triangulation_angle: float = Field(ge=0, le=180)
    confidence_class: ConfidenceLabel


class PointConfidenceArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    point_order: Literal["PLY_VERTEX_ORDER"] = "PLY_VERTEX_ORDER"
    points: list[PointConfidenceRecord]

    @model_validator(mode="after")
    def require_contiguous_vertex_ids(self) -> PointConfidenceArtifact:
        point_ids = [point.point_id for point in self.points]
        if sorted(point_ids) != list(range(len(point_ids))):
            raise ValueError(
                "point_id values must be unique and contiguous PLY vertex indices starting at zero"
            )
        return self


class MatcherMetrics(BaseModel):
    matcher: str
    eligible_frames: int = Field(gt=0)
    registered_frames: int = Field(ge=0)
    median_reprojection_error_px: float = Field(ge=0)
    p95_reprojection_error_px: float = Field(ge=0)
    runtime_s: float = Field(ge=0)

    @property
    def registration_rate(self) -> float:
        return self.registered_frames / self.eligible_frames
