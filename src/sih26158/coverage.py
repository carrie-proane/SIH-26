"""Depth-consistent surface support and additional-frame proposals, separate from time coverage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .completion import MeshUnavailable, load_observed_mesh
from .completion_contract import ContractModel, MeshSource
from .exports import ExportUnavailable


@dataclass(frozen=True)
class DepthView:
    """Rectified camera-Z metric depth in the declared ENU frame, never relative monocular depth."""

    frame_id: str
    video_sha256: str
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    depth_m: np.ndarray
    depth_sha256: str
    selected: bool = False
    depth_semantics: str = "CAMERA_Z_METRES"


def depth_array_sha256(array: np.ndarray) -> str:
    """Canonical numeric hash includes shape and little-endian float64 samples."""
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(str(canonical.shape).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


class CoverageReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    contract_status: Literal["PROPOSED_NOT_INTEGRATED"] = "PROPOSED_NOT_INTEGRATED"
    status: Literal["UNAVAILABLE", "COMPLETED"]
    domain: str = "CENTROIDS_OF_DECLARED_OBSERVED_MESH_FACES"
    method: str = "RECTIFIED_CAMERA_Z_DEPTH_CONSISTENCY"
    source: MeshSource | None
    available_support_statistics: dict = Field(default_factory=dict)
    views: list[dict] = Field(default_factory=list)
    additional_frame_proposals: list[dict] = Field(default_factory=list)
    observed_surface_area_m2: None = None
    reference_surface_area_m2: None = None
    observed_surface_completeness_percent: None = None
    denominator: None = None
    unavailable_reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Frustum membership alone is not visibility; nearer valid depth marks occlusion.",
            "Centroid samples do not establish visibility of whole triangles or unseen surfaces.",
            "Depth consistency inherits the source reconstruction and depth uncertainty.",
            "Inferred faces are excluded by the observed-source contract.",
            "Additional frames are proposals; execution reuses existing extraction and selection.",
        ]
    )


def _validate_view(view: DepthView, video_sha256: str) -> None:
    if view.video_sha256 != video_sha256:
        raise ValueError("Candidate frames must come from the same raw video")
    if view.depth_semantics != "CAMERA_Z_METRES":
        raise ValueError("Only rectified camera-Z metric depth is supported")
    pose, intrinsics, depth = map(np.asarray, (view.world_to_camera, view.intrinsics, view.depth_m))
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid world-to-camera pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-6
    ):
        raise ValueError("Camera pose must use a proper rigid rotation")
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or not np.allclose(intrinsics[2], [0, 0, 1])
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
    ):
        raise ValueError("Invalid rectified camera intrinsics")
    if depth.ndim != 2 or not depth.size or depth.size > 16_777_216:
        raise ValueError("Invalid or oversized depth image")
    if depth_array_sha256(depth) != view.depth_sha256:
        raise ValueError("Depth checksum mismatch")


def analyze_surface_support(
    source_root: Path,
    source: MeshSource | None,
    views: tuple[DepthView, ...],
    *,
    video_sha256: str,
    depth_tolerance_m: float = 0.03,
    max_additional_frames: int = 10,
) -> CoverageReport:
    if not np.isfinite(depth_tolerance_m) or not 0 < depth_tolerance_m <= 0.25:
        raise ValueError("Invalid depth tolerance")
    if not 0 <= max_additional_frames <= 100 or len(views) > 1000:
        raise ValueError("Frame work limits exceeded")
    if len(video_sha256) != 64 or any(c not in "0123456789abcdef" for c in video_sha256):
        raise ValueError("A raw-video SHA-256 is required")
    if len({view.frame_id for view in views}) != len(views):
        raise ValueError("Frame IDs must be unique")
    unavailable = [
        "No independent visible-surface reference domain or area denominator is declared."
    ]
    if source is None or not views:
        return CoverageReport(
            status="UNAVAILABLE",
            source=source,
            unavailable_reasons=unavailable
            + ["Declared observed mesh and metric depth with camera poses are required."],
        )
    try:
        mesh = load_observed_mesh(source_root, source)
    except (MeshUnavailable, ExportUnavailable, OSError) as exc:
        return CoverageReport(
            status="UNAVAILABLE", source=source, unavailable_reasons=unavailable + [str(exc)]
        )
    centroids = np.asarray(mesh.triangles_center)
    support_sets = {}
    details = []
    supported = set()
    for view in views:
        _validate_view(view, video_sha256)
        camera = centroids @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3]
        z = camera[:, 2]
        projected = camera @ view.intrinsics.T
        pixels = projected[:, :2] / np.where(z > 0, z, 1)[:, None]
        height, width = view.depth_m.shape
        # Bounds before conversion avoid integer overflow on far-away/behind-camera samples.
        valid = (
            (z > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width - 0.5)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height - 0.5)
        )
        ids = np.flatnonzero(valid)
        uv = np.rint(pixels[ids]).astype(int)
        depth = view.depth_m[uv[:, 1], uv[:, 0]]
        measured = np.isfinite(depth) & (depth > 0)
        consistent = measured & (np.abs(depth - z[ids]) <= depth_tolerance_m)
        occluded = measured & (depth < z[ids] - depth_tolerance_m)
        support_sets[view.frame_id] = set(ids[consistent].tolist())
        if view.selected:
            supported.update(support_sets[view.frame_id])
        details.append(
            {
                "frame_id": view.frame_id,
                "selected": view.selected,
                "video_sha256": view.video_sha256,
                "depth_sha256": view.depth_sha256,
                "world_to_camera": view.world_to_camera.tolist(),
                "intrinsics": view.intrinsics.tolist(),
                "frustum_sample_count": len(ids),
                "supported_sample_count": int(consistent.sum()),
                "occluded_sample_count": int(occluded.sum()),
                "unknown_depth_sample_count": int((~measured).sum()),
                "inconsistent_free_space_sample_count": int(
                    (measured & ~consistent & ~occluded).sum()
                ),
            }
        )
    # Greedy marginal support avoids selecting redundant frames solely for a high total score.
    candidates = {view.frame_id for view in views if not view.selected}
    recovered = set(supported)
    proposals = []
    for _ in range(max_additional_frames):
        if not candidates:
            break
        best = min(candidates, key=lambda key: (-len(support_sets[key] - recovered), key))
        new = support_sets[best] - recovered
        if not new:
            break
        proposals.append(
            {
                "frame_id": best,
                "additional_supported_face_ids": sorted(new),
                "additional_supported_sample_count": len(new),
            }
        )
        recovered.update(new)
        candidates.remove(best)
    return CoverageReport(
        status="COMPLETED",
        source=source,
        views=details,
        additional_frame_proposals=proposals,
        available_support_statistics={
            "evaluated_face_centroids": len(centroids),
            "selected_view_supported_centroids": len(supported),
            "proposed_additional_supported_centroids": len(recovered - supported),
            "depth_tolerance_m": depth_tolerance_m,
            "max_additional_frames": max_additional_frames,
        },
        unavailable_reasons=unavailable,
    )
