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
    enable_segmentation: bool = False
    segmentation_model_path: str | None = None
    reconstruction_target: Literal["FULL_SCENE", "PRIMARY_SUBJECT"] = "FULL_SCENE"
    masking_mode: Literal["OFF", "AUTO", "REQUIRED"] = "OFF"
    enable_dense_reconstruction: bool = False
    dense_provider: Literal["auto", "colmap", "openmvs"] = "auto"
    enable_surface_completion: bool = False
    surface_completion_provider: Literal["external"] = "external"
    surface_completion_model_path: str | None = None
    surface_completion_samples: int = Field(default=3, ge=1, le=8)

    @field_validator("measured_distance_m")
    @classmethod
    def require_reference_for_measurement(cls, value: float | None, info: Any) -> float | None:
        if value is not None and info.data.get("known_distance_m") is None:
            raise ValueError("known_distance_m is required when measured_distance_m is provided")
        return value

    @model_validator(mode="after")
    def validate_manual_offset_source(self) -> RunConfig:
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
    status: Literal["STARTED", "COMPLETED", "FAILED"]
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
