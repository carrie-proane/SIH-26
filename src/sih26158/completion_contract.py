"""Proposed completion v1 contract; deliberately independent of run/viewer enums.

Integration and publication require Jay's schema/lifecycle review. These models do not
register artifacts or authorize writes to completed runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .dataset import DatasetAsset


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MeshCoordinates(ContractModel):
    source_frame: Literal["LOCAL_ENU_METRES"] = "LOCAL_ENU_METRES"
    units: Literal["metre"] = "metre"
    axis_convention: tuple[Literal["east"], Literal["north"], Literal["up"]] = (
        "east",
        "north",
        "up",
    )
    alignment_verified: Literal[True]
    altitude_reference: str = Field(min_length=1)
    # Must identify the transform/scale evidence; no inferred alignment from a filename.
    alignment_evidence: DatasetAsset


class MeshSource(ContractModel):
    run_id: str = Field(min_length=1)
    artifact: DatasetAsset
    coordinates: MeshCoordinates
    geometry_source: Literal["OBSERVED_RECONSTRUCTION"] = "OBSERVED_RECONSTRUCTION"


class GapReview(ContractModel):
    boundary_id: str
    decision: Literal["CONFIRMED_SMALL_GAP", "STRUCTURAL_OPENING", "UNKNOWN"]
    reviewer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_images: tuple[DatasetAsset, ...] = ()


class CompletionParameters(ContractModel):
    max_area_m2: float = Field(default=0.25, gt=0, le=1)
    max_diameter_m: float = Field(default=1, gt=0, le=2)
    max_edge_m: float = Field(default=0.5, gt=0, le=1)
    max_plane_distance_m: float = Field(default=0.01, gt=0, le=0.05)
    max_normal_angle_deg: float = Field(default=10, gt=0, le=20)
    min_support_faces: int = Field(default=6, ge=3)
    min_support_area_ratio: float = Field(default=2, ge=1)
    max_boundary_vertices: int = Field(default=64, ge=3, le=256)
    max_regions: int = Field(default=64, ge=1, le=256)
    max_source_faces: int = Field(default=200_000, ge=1, le=1_000_000)
    max_source_bytes: int = Field(default=50_000_000, ge=1, le=250_000_000)


class CompletionReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    contract_status: Literal["PROPOSED_NOT_INTEGRATED"] = "PROPOSED_NOT_INTEGRATED"
    status: Literal["NOT_RUN", "UNAVAILABLE", "REJECTED", "COMPLETED", "FAILED"]
    reason: str
    method: Literal["BOUNDED_PLANAR_GAP"] = "BOUNDED_PLANAR_GAP"
    source: MeshSource | None
    parameters: CompletionParameters
    runtime_s: float = Field(ge=0)
    generated_vertex_count: int = Field(default=0, ge=0)
    generated_face_count: int = Field(default=0, ge=0)
    accepted_regions: list[dict] = Field(default_factory=list)
    rejected_regions: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    measurement_eligible: Literal[False] = False


class CompletionProvenance(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    source: MeshSource
    output: DatasetAsset
    geometry_source: Literal["AI_INFERRED"] = "AI_INFERRED"
    confidence_label: Literal["AI_ASSISTED_NOT_MEASURABLE"] = "AI_ASSISTED_NOT_MEASURABLE"
    method: Literal["BOUNDED_PLANAR_GAP"] = "BOUNDED_PLANAR_GAP"
    measurable: Literal[False] = False
    face_ids: list[int]
    regions: list[dict]
    assumptions: list[str]
    score_semantics: Literal["RULE_BASED_NOT_CALIBRATED_PROBABILITY"] = (
        "RULE_BASED_NOT_CALIBRATED_PROBABILITY"
    )
