"""Conservative bounded planar infill. No pipeline, export or viewer registration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

from .completion_contract import (
    CompletionParameters,
    CompletionProvenance,
    CompletionReport,
    GapReview,
    MeshSource,
)
from .dataset import DatasetAsset, resolve_dataset_asset
from .exports import _load_ply
from .process_control import ProcessCancelledError, ProcessTimeoutError
from .storage import atomic_json, sha256_file


class MeshUnavailable(ValueError):
    pass


def verified_asset(root: Path, asset: DatasetAsset) -> Path:
    path = resolve_dataset_asset(root / "manifest.json", asset)
    if not path.is_file() or sha256_file(path) != asset.sha256:
        raise MeshUnavailable(f"Missing or checksum-mismatched declared asset: {asset.path}")
    return path


def load_observed_mesh(
    root: Path, source: MeshSource, parameters: CompletionParameters | None = None
) -> trimesh.Trimesh:
    limits = parameters or CompletionParameters()
    path = verified_asset(root, source.artifact)
    verified_asset(root, source.coordinates.alignment_evidence)
    if path.suffix.lower() != ".ply" or path.stat().st_size > limits.max_source_bytes:
        raise MeshUnavailable("A bounded, declared PLY mesh is required")
    mesh = _load_ply(path)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise MeshUnavailable("A surface mesh is required; sparse points cannot support completion")
    if len(mesh.faces) > limits.max_source_faces:
        raise MeshUnavailable("Source exceeds the configured face limit")
    if not np.isfinite(mesh.area_faces).all() or np.any(mesh.area_faces <= 1e-12):
        raise MeshUnavailable("Degenerate source faces are unsupported")
    if len(np.unique(np.sort(mesh.faces, axis=1), axis=0)) != len(mesh.faces):
        raise MeshUnavailable("Duplicate source faces are unsupported")
    return mesh


def boundary_id(vertices: list[int]) -> str:
    # Independent of loop start and winding; source hash in the contract binds the IDs.
    digest = hashlib.sha256(json.dumps(sorted(vertices)).encode()).hexdigest()[:16]
    return f"boundary-{digest}"


def boundary_loops(mesh: trimesh.Trimesh) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    owners: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(mesh.faces):
        for a, b in zip(face, np.roll(face, -1), strict=True):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b), face_id))
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    edge_faces = {}
    for key, entries in owners.items():
        if len(entries) > 2:
            raise MeshUnavailable("Non-manifold edges are unsupported")
        if len(entries) == 2:
            if entries[0][:2] != entries[1][1::-1]:
                raise MeshUnavailable("Inconsistent source face winding")
            continue
        a, b, face_id = entries[0]
        if a in outgoing or b in incoming:
            raise MeshUnavailable("Disconnected or touching boundary loops are unsupported")
        outgoing[a], incoming[b] = b, a
        edge_faces[key] = face_id
    if set(outgoing) != set(incoming):
        raise MeshUnavailable("Open boundary chains are unsupported")
    remaining = set(outgoing)
    loops = []
    while remaining:
        start = min(remaining)
        loop, current = [], start
        while current in remaining:
            loop.append(current)
            remaining.remove(current)
            current = outgoing[current]
        if current != start or len(loop) < 3:
            raise MeshUnavailable("Boundary is not a simple closed loop")
        loops.append(loop)
    return loops, edge_faces


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    signed = _cross2(polygon, np.roll(polygon, -1, axis=0)).sum()
    # Keep memory linear in source size, rather than allocating points x loop vertices.
    inside = np.ones(len(points), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        inside &= _cross2(b - a, points - a) * np.sign(signed) > 1e-10
        if not inside.any():
            break
    return inside


def assess_gap(
    mesh: trimesh.Trimesh,
    loop: list[int],
    edge_faces: dict[tuple[int, int], int],
    parameters: CompletionParameters,
) -> dict:
    """Geometric candidate assessment only; source-image review is a separate gate."""
    result = {"boundary_id": boundary_id(loop), "source_vertex_ids": loop, "reasons": []}
    reasons = result["reasons"]
    if len(loop) > parameters.max_boundary_vertices:
        reasons.append("BOUNDARY_VERTEX_LIMIT")
        return result
    points = np.asarray(mesh.vertices)[loop]
    support_ids = sorted(
        {edge_faces[tuple(sorted((a, b)))] for a, b in zip(loop, loop[1:] + loop[:1], strict=True)}
    )
    support_vertices = np.unique(mesh.faces[support_ids])
    support = mesh.vertices[support_vertices]
    center = support.mean(axis=0)
    _, singular, axes = np.linalg.svd(support - center, full_matrices=False)
    if len(singular) < 3 or singular[1] <= 1e-10:
        reasons.append("UNSUPPORTED_PLANE")
        return result
    normal = axes[2]
    projected = (points - center) @ axes[:2].T
    edges = np.roll(projected, -1, axis=0) - projected
    turns = _cross2(edges, np.roll(edges, -1, axis=0))
    signed_area = _cross2(projected, np.roll(projected, -1, axis=0)).sum() / 2
    area = abs(float(signed_area))
    # Every nonadjacent vertex must lie strictly inside each edge's half-plane.
    # This rejects self-crossing polygons as well as concave/collinear loops.
    halfplanes = _cross2(
        edges[:, None, :], projected[None, :, :] - projected[:, None, :]
    ) * np.sign(signed_area)
    for i in range(len(loop)):
        halfplanes[i, i] = halfplanes[i, (i + 1) % len(loop)] = 1
    convex = bool(np.all(turns * np.sign(signed_area) > 1e-10) and np.all(halfplanes > 1e-10))
    distance = float(np.max(np.abs((support - center) @ normal)))
    diameter = float(np.linalg.norm(points[:, None] - points[None, :], axis=2).max())
    edge_max = float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).max())
    normals = np.asarray(mesh.face_normals)[support_ids]
    angle = float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    support_area = float(mesh.area_faces[support_ids].sum())
    result.update(
        area_m2=area,
        diameter_m=diameter,
        max_edge_m=edge_max,
        plane_max_distance_m=distance,
        normal_spread_deg=angle,
        support_face_ids=support_ids,
        support_face_count=len(support_ids),
        support_area_m2=support_area,
    )
    gates = [
        (convex and area > 1e-10, "NONCONVEX_OR_DEGENERATE_BOUNDARY"),
        (area <= parameters.max_area_m2, "AREA_LIMIT"),
        (diameter <= parameters.max_diameter_m, "DIAMETER_LIMIT"),
        (edge_max <= parameters.max_edge_m, "EDGE_LIMIT"),
        (distance <= parameters.max_plane_distance_m, "NONPLANAR_SUPPORT"),
        (angle <= parameters.max_normal_angle_deg, "SHARP_CORNER_OR_NORMAL_DISAGREEMENT"),
        (len(support_ids) >= parameters.min_support_faces, "INSUFFICIENT_SUPPORT_FACES"),
        (support_area >= area * parameters.min_support_area_ratio, "INSUFFICIENT_SUPPORT_AREA"),
    ]
    reasons.extend(reason for passed, reason in gates if not passed)
    if convex:
        # An outside scene boundary surrounds existing faces; a hole surrounds empty space.
        centroids = mesh.triangles_center
        near_plane = np.abs((centroids - center) @ normal) <= parameters.max_plane_distance_m
        if np.any(_inside_convex((centroids[near_plane] - center) @ axes[:2].T, projected)):
            reasons.append("EXISTING_SURFACE_INSIDE_BOUNDARY")
        other_vertices = np.setdiff1d(np.arange(len(mesh.vertices)), loop)
        nearby = mesh.vertices[other_vertices]
        nearby = nearby[np.abs((nearby - center) @ normal) <= parameters.max_plane_distance_m]
        if np.any(_inside_convex((nearby - center) @ axes[:2].T, projected)):
            reasons.append("NESTED_OR_DISCONNECTED_SURFACE")
    result["rule_based_score"] = max(0.0, 1.0 - distance / parameters.max_plane_distance_m)
    return result


def complete_planar_gaps(
    source_root: Path,
    output_dir: Path,
    source: MeshSource | None,
    *,
    parameters: CompletionParameters | None = None,
    reviews: tuple[GapReview, ...] = (),
    cancel_requested: Callable[[], bool] | None = None,
    timeout_s: float = 60.0,
) -> CompletionReport:
    """Write a fresh, external derived bundle; never modify or publish a source run."""
    started = time.monotonic()
    if not np.isfinite(timeout_s) or not 0 < timeout_s <= 3600:
        raise ValueError("Invalid completion timeout")

    def check_control() -> None:
        if cancel_requested and cancel_requested():
            raise ProcessCancelledError("Completion cancellation requested")
        if time.monotonic() - started >= timeout_s:
            raise ProcessTimeoutError("Completion exceeded its cooperative deadline")

    check_control()
    parameters = parameters or CompletionParameters()
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("Completion output must be outside the immutable source run")
    output_dir.mkdir(parents=True, exist_ok=False)
    completion_dir = output_dir / "completion"
    completion_dir.mkdir()
    accepted, rejected, generated_vertices, generated_faces = [], [], [], []
    status, reason = "UNAVAILABLE", "No declared metric surface mesh supplied"
    try:
        if source is not None:
            mesh = load_observed_mesh(source_root, source, parameters)
            check_control()
            loops, edge_faces = boundary_loops(mesh)
            if len(loops) > parameters.max_regions:
                raise MeshUnavailable("Boundary region limit exceeded")
            review_map = {item.boundary_id: item for item in reviews}
            if len(review_map) != len(reviews):
                raise ValueError("Duplicate boundary reviews")
            unknown = set(review_map) - {boundary_id(loop) for loop in loops}
            if unknown:
                raise ValueError("Reviews reference boundaries absent from the source mesh")
            for loop in loops:
                check_control()
                region = assess_gap(mesh, loop, edge_faces, parameters)
                review = review_map.get(region["boundary_id"])
                if review is None or review.decision != "CONFIRMED_SMALL_GAP":
                    region["reasons"].append(
                        "STRUCTURAL_OPENING"
                        if review and review.decision == "STRUCTURAL_OPENING"
                        else "SOURCE_IMAGE_REVIEW_REQUIRED"
                    )
                elif not review.source_images:
                    region["reasons"].append("SOURCE_IMAGE_EVIDENCE_REQUIRED")
                else:
                    for asset in review.source_images:
                        verified_asset(source_root, asset)
                if review is not None:
                    region["review"] = review.model_dump(mode="json")
                if region["reasons"]:
                    rejected.append(region)
                    continue
                # Boundary vertices retain original coordinates, without a second ENU/GLB transform.
                # Reverse boundary winding so shared edges oppose the observed faces.
                ordered = list(reversed(loop))
                offset, face_start = len(generated_vertices), len(generated_faces)
                generated_vertices.extend(mesh.vertices[ordered].tolist())
                generated_faces.extend(
                    [[offset, offset + i, offset + i + 1] for i in range(1, len(loop) - 1)]
                )
                region["face_ids"] = list(range(face_start, len(generated_faces)))
                region["output_vertex_source_ids"] = ordered
                accepted.append(region)
            verified_asset(source_root, source.artifact)
            verified_asset(source_root, source.coordinates.alignment_evidence)
            check_control()
            if accepted:
                patch = trimesh.Trimesh(
                    vertices=generated_vertices, faces=generated_faces, process=False
                )
                target = completion_dir / "generated_mesh.ply"
                target.write_bytes(
                    trimesh.exchange.ply.export_ply(
                        patch, encoding="binary_little_endian", vertex_normal=False
                    )
                )
                reopened = _load_ply(target)
                if not np.array_equal(reopened.faces, patch.faces) or not np.array_equal(
                    reopened.vertices, patch.vertices
                ):
                    raise ValueError(
                        "Generated PLY cannot preserve coordinate precision or face identity"
                    )
                provenance = CompletionProvenance(
                    source=source,
                    output=DatasetAsset(
                        path="completion/generated_mesh.ply", sha256=sha256_file(target)
                    ),
                    face_ids=list(range(len(generated_faces))),
                    regions=accepted,
                    assumptions=[
                        "Reviewed small gaps are assumed to continue the adjacent plane.",
                        "Operator review is an assertion, not automated proof of visibility.",
                        "No copied texture; generated faces never count as observed coverage.",
                    ],
                )
                atomic_json(completion_dir / "provenance.json", provenance.model_dump(mode="json"))
                status, reason = "COMPLETED", "Generated only reviewed, bounded planar patches"
            else:
                status, reason = (
                    "REJECTED",
                    "No boundary passed geometry and source-image review gates",
                )
    except (ProcessCancelledError, ProcessTimeoutError):
        raise
    except MeshUnavailable as exc:
        status, reason = "UNAVAILABLE", str(exc)
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        status, reason = "FAILED", str(exc)
    if status != "COMPLETED":
        for name in ("generated_mesh.ply", "provenance.json"):
            (completion_dir / name).unlink(missing_ok=True)
        accepted, generated_vertices, generated_faces = [], [], []
    report = CompletionReport(
        status=status,
        reason=reason,
        source=source,
        parameters=parameters,
        runtime_s=time.monotonic() - started,
        generated_vertex_count=len(generated_vertices),
        generated_face_count=len(generated_faces),
        accepted_regions=accepted,
        rejected_regions=rejected,
        warnings=[
            "Experimental fixture-tested algorithm; real gap accuracy is unvalidated.",
            "Symmetry, terrain interpolation and learned depth fusion are deferred.",
        ],
    )
    atomic_json(completion_dir / "report.json", report.model_dump(mode="json"))
    return report
