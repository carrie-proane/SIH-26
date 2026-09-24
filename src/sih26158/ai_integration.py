"""Checkpoint-facing contracts for segmentation, topology coverage, and completion.

The helpers in this module bind the standalone algorithms to declared run artifacts.  They do
not weaken the observed/inferred provenance boundary: observed geometry is never rewritten and
every completion output is published under a hash-keyed derived path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from .completion import assess_gap, boundary_loops, complete_planar_gaps, load_observed_mesh
from .completion_contract import CompletionParameters, GapReview, MeshCoordinates, MeshSource
from .coverage import DepthView, analyze_surface_support
from .dataset import DatasetAsset
from .exports import ExportUnavailable, _load_ply
from .models import RunConfig, RunRecord
from .segmentation import SegmentationSettings, segmentation_fingerprint_inputs
from .storage import atomic_json, sha256_file

AI_STAGE_CONTRACT_VERSION = "1.0"
MAX_METRIC_DEPTH_ARTIFACT_BYTES = 134_217_728


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def segmentation_settings(config: RunConfig, *, device: str | None = None) -> SegmentationSettings:
    return SegmentationSettings(
        device=device or config.segmentation_device,
        image_size=config.segmentation_image_size,
        confidence=config.segmentation_confidence,
        iou_threshold=config.segmentation_iou_threshold,
        max_frames=config.max_selected_frames,
        timeout_s=config.segmentation_timeout_s,
        mask_dilation_px=config.segmentation_mask_dilation_px,
        mask_erosion_px=config.segmentation_mask_erosion_px,
        excluded_classes=tuple(config.segmentation_excluded_classes),
    )


def resolve_segmentation_device(config: RunConfig) -> tuple[str | None, list[dict[str, str]]]:
    """Resolve only explicitly available devices; CPU fallback is an opt-in policy."""

    requested = config.segmentation_device
    if requested == "cpu":
        return "cpu", []
    available = False
    cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_explicitly_hidden = requested == "cuda" and cuda_visibility is not None and (
        not cuda_visibility.strip() or cuda_visibility.strip() == "-1"
    )
    try:
        import torch

        available = not cuda_explicitly_hidden and (
            bool(torch.cuda.is_available())
            if requested == "cuda"
            else bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )
    except ImportError:
        available = False
    if available:
        return requested, []
    if config.segmentation_allow_cpu_fallback:
        return "cpu", [
            {
                "code": "SEGMENTATION_CPU_FALLBACK",
                "message": f"Requested {requested} was unavailable; explicit CPU fallback was used.",
            }
        ]
    return None, [
        {
            "code": "SEGMENTATION_DEVICE_UNAVAILABLE",
            "message": f"Requested segmentation device {requested} is unavailable and CPU fallback is disabled.",
        }
    ]


def selected_frame_evidence(run_dir: Path) -> dict[str, object]:
    keyframes_path = run_dir / "keyframes.json"
    payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    selected: list[dict[str, object]] = []
    for order, frame in enumerate(frames):
        if not isinstance(frame, dict) or not frame.get("selected", True):
            continue
        name = Path(str(frame.get("image_name", ""))).name
        path = run_dir / "frames" / name
        selected.append(
            {
                "order": order,
                "frame_index": frame.get("frame_index"),
                "image_name": name,
                "sha256": sha256_file(path) if name and path.is_file() else None,
                "artifact_path": f"frames/{name}" if name and path.is_file() else None,
                "artifact_url": f"/api/runs/{{run_id}}/artifacts/frames/{name}"
                if name and path.is_file()
                else None,
            }
        )
    return {
        "selected_frames": selected,
        "selected_frame_set_sha256": canonical_hash(selected),
    }


def segmentation_fingerprint(
    record: RunRecord, run_dir: Path, *, device: str | None = None
) -> dict:
    settings = segmentation_settings(record.config, device=device)
    return {
        "contract": AI_STAGE_CONTRACT_VERSION,
        "frames": selected_frame_evidence(run_dir),
        "model_name": record.config.segmentation_model_name,
        "model_version": record.config.segmentation_model_version,
        "provider": segmentation_fingerprint_inputs(
            record.config.segmentation_model_path, settings
        ),
        "allow_cpu_fallback": record.config.segmentation_allow_cpu_fallback,
    }


def write_segmentation_status(
    record: RunRecord,
    run_dir: Path,
    comparison: dict[str, object] | None,
    warnings: list[dict[str, str]],
    *,
    requested_device: str,
    executed_device: str | None,
) -> Path:
    comparison = comparison or {}
    raw_status = str(comparison.get("status", "DISABLED"))
    status = {
        "APPLIED": "completed",
        "DISABLED": "not_run",
        "UNAVAILABLE_FALLBACK": "unavailable",
        "FAILED_FALLBACK": "failed",
        "BLOCKED": "failed",
    }.get(raw_status, "failed")
    selected_count = int(comparison.get("selected_frame_count", 0) or 0)
    mask_paths = sorted((run_dir / "masks" / "reconstruction").glob("*.png"))
    mask_hashes = [
        {"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)}
        for path in mask_paths
        if path.is_file()
    ]
    fingerprint = segmentation_fingerprint(
        record, run_dir, device=executed_device or requested_device
    )
    execution = comparison.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    failure_reason = None
    if status in {"failed", "unavailable"}:
        failure_reason = str(comparison.get("comparison") or "Segmentation did not execute")
    payload = {
        "schema_version": "1.0",
        "contract_version": AI_STAGE_CONTRACT_VERSION,
        "status": status,
        "model": record.config.segmentation_model_name,
        "model_version": record.config.segmentation_model_version,
        "weights_sha256": execution.get("model_sha256"),
        "requested_device": requested_device,
        "device": executed_device or "unavailable",
        "frame_count": selected_count,
        "masked_frame_count": len(mask_paths) // 2 if status == "completed" else 0,
        "mask_directory": "masks/reconstruction" if status == "completed" else None,
        "artifact_sha256": canonical_hash(mask_hashes) if mask_hashes else None,
        "warnings": warnings,
        "failure_reason": failure_reason,
        "fallback_to_unmasked": status in {"failed", "unavailable"}
        and record.config.masking_mode == "AUTO",
        "downstream_consumption": "PENDING_SPARSE_RECONSTRUCTION"
        if status == "completed"
        else "NOT_APPLIED",
        "fingerprint": fingerprint,
    }
    path = run_dir / "segmentation_status.json"
    atomic_json(path, payload)
    return path


def _declared(record: RunRecord) -> dict[str, object]:
    return {item.relative_path: item for item in record.artifacts}


def observed_mesh_source(record: RunRecord, run_dir: Path) -> tuple[MeshSource | None, str]:
    declared = _declared(record)
    candidates = [
        "dense/textured/model.ply",
        "dense/textured/textured.ply",
        "dense/meshed-openmvs-refined.ply",
        "dense/meshed-poisson-simplified.ply",
        "dense/meshed-openmvs.ply",
        "dense/meshed-poisson.ply",
    ]
    candidates.extend(
        sorted(
            path
            for path in declared
            if path.startswith("dense/")
            and path.endswith(".ply")
            and "completion" not in path.lower()
            and "generated" not in path.lower()
        )
    )
    transform = declared.get("local_transform.json")
    if transform is None:
        return None, "A declared local_transform.json is required for metric completion"
    transform_path = run_dir / "local_transform.json"
    try:
        transform_payload = json.loads(transform_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "The declared local transform is unreadable"
    for relative in dict.fromkeys(candidates):
        artifact = declared.get(relative)
        path = run_dir / relative
        if artifact is None or not path.is_file() or sha256_file(path) != artifact.sha256:
            continue
        try:
            geometry = _load_ply(path)
        except (ExportUnavailable, OSError, ValueError):
            continue
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        source = MeshSource(
            run_id=record.run_id,
            artifact=DatasetAsset(path=relative, sha256=artifact.sha256),
            coordinates=MeshCoordinates(
                alignment_verified=True,
                altitude_reference=str(transform_payload.get("altitude_reference", "UNKNOWN")),
                alignment_evidence=DatasetAsset(
                    path="local_transform.json", sha256=transform.sha256
                ),
            ),
        )
        return source, "Declared observed dense mesh is available"
    return (
        None,
        "No declared observed dense mesh with faces is available; sparse points are not triangulated",
    )


def _declared_metric_depth_views(
    record: RunRecord, run_dir: Path, video_sha256: str
) -> tuple[DepthView, ...]:
    contract_relative = "coverage_depth_views.json"
    declared = _declared(record)
    contract_artifact = declared.get(contract_relative)
    if contract_artifact is None:
        return ()
    contract_path = run_dir / contract_relative
    if not contract_path.is_file() or sha256_file(contract_path) != contract_artifact.sha256:
        raise ValueError("Declared metric-depth contract checksum mismatch")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or payload.get("video_sha256") != video_sha256:
        raise ValueError("Metric-depth contract is not bound to this run's raw video hash")
    rows = payload.get("views")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Metric-depth contract contains no views")
    views: list[DepthView] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Metric-depth view must be an object")
        relative = str(row.get("depth_artifact_path", ""))
        artifact = declared.get(relative)
        path = (run_dir / relative).resolve()
        if (
            artifact is None
            or not path.is_relative_to(run_dir.resolve())
            or not path.is_file()
            or path.stat().st_size > MAX_METRIC_DEPTH_ARTIFACT_BYTES
            or sha256_file(path) != artifact.sha256
            or row.get("depth_artifact_sha256") != artifact.sha256
        ):
            raise ValueError(
                "Metric-depth view is not a bounded, hash-matched declared artifact"
            )
        depth = np.load(path, allow_pickle=False)
        views.append(
            DepthView(
                frame_id=str(row.get("frame_id", "")),
                video_sha256=str(row.get("video_sha256", "")),
                world_to_camera=np.asarray(row.get("world_to_camera"), dtype=float),
                intrinsics=np.asarray(row.get("intrinsics"), dtype=float),
                depth_m=depth,
                depth_sha256=str(row.get("depth_array_sha256", "")),
                selected=bool(row.get("selected", False)),
                depth_semantics=str(row.get("depth_semantics", "")),
            )
        )
    return tuple(views)


def write_coverage_report(
    record: RunRecord, run_dir: Path, *, video_sha256: str | None = None
) -> Path:
    source, reason = observed_mesh_source(record, run_dir)
    frame_evidence = selected_frame_evidence(run_dir)
    declared_artifacts = _declared(record)
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "contract_status": "INTEGRATED",
        "status": "UNAVAILABLE",
        "method": "OBSERVED_MESH_TOPOLOGY",
        "method_version": AI_STAGE_CONTRACT_VERSION,
        "source_geometry_sha256": source.artifact.sha256 if source else None,
        "source_frame_set_sha256": frame_evidence["selected_frame_set_sha256"],
        "review_evidence_options": [
            {
                **item,
                "artifact_url": str(item["artifact_url"]).format(run_id=record.run_id),
            }
            for item in frame_evidence["selected_frames"]
            if item.get("sha256")
            and item.get("artifact_path")
            and (declared_item := declared_artifacts.get(str(item["artifact_path"]))) is not None
            and declared_item.sha256 == item["sha256"]
        ],
        "observed_region_statistics": {},
        "candidate_missing_region_statistics": {},
        "disconnected_components": None,
        "confidence": None,
        "sufficient_for_completion": False,
        "observed_surface_completeness_percent": None,
        "reference_surface_area_m2": None,
        "denominator": None,
        "warnings": [],
        "metric_depth_status": "NOT_DECLARED",
        "limitations": [
            "Topology can identify bounded holes but cannot prove that unseen surfaces exist.",
            "No defensible visible-scene reference denominator is available.",
            "Inferred geometry is excluded from observed coverage and official measurement.",
        ],
        "failure_reason": None if source else reason,
    }
    if not record.config.enable_coverage_analysis:
        payload["status"] = "NOT_RUN"
        payload["failure_reason"] = "Coverage analysis was disabled by configuration"
        path = run_dir / "coverage_report.json"
        atomic_json(path, payload)
        return path
    if source is not None:
        try:
            mesh = load_observed_mesh(run_dir, source)
            loops, edge_faces = boundary_loops(mesh)
            parameters = CompletionParameters()
            regions = []
            for loop in loops:
                region = assess_gap(mesh, loop, edge_faces, parameters)
                region["boundary_coordinates_enu_m"] = np.asarray(
                    mesh.vertices[np.asarray(loop, dtype=int)]
                ).tolist()
                region["eligible_for_operator_review"] = not bool(region.get("reasons"))
                regions.append(region)
            eligible = [item for item in regions if not item.get("reasons")]
            components = trimesh.graph.connected_components(
                mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
            )
            payload.update(
                status="COMPLETED",
                observed_region_statistics={
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "surface_area_m2": float(mesh.area),
                    "bounds": np.asarray(mesh.bounds).tolist(),
                },
                candidate_missing_region_statistics={
                    "boundary_count": len(regions),
                    "geometrically_eligible_bounded_gap_count": len(eligible),
                    "regions": regions,
                },
                disconnected_components=len(components),
                confidence={
                    "label": "RULE_BASED_NOT_CALIBRATED_PROBABILITY",
                    "value": None,
                },
                sufficient_for_completion=bool(eligible),
            )
            if video_sha256 and "coverage_depth_views.json" in _declared(record):
                try:
                    depth_views = _declared_metric_depth_views(record, run_dir, video_sha256)
                    depth_support = analyze_surface_support(
                        run_dir, source, depth_views, video_sha256=video_sha256
                    )
                    payload["method"] = (
                        "OBSERVED_MESH_TOPOLOGY_AND_RECTIFIED_CAMERA_Z_DEPTH_CONSISTENCY"
                    )
                    payload["metric_depth_status"] = depth_support.status
                    payload["depth_support"] = depth_support.model_dump(mode="json")
                except (OSError, ValueError, TypeError) as exc:
                    payload["metric_depth_status"] = "REJECTED"
                    payload["warnings"].append(
                        {"code": "METRIC_DEPTH_REJECTED", "message": str(exc)}
                    )
        except (ExportUnavailable, OSError, ValueError, RuntimeError, IndexError) as exc:
            payload["failure_reason"] = str(exc)
            payload["warnings"] = [{"code": "COVERAGE_ANALYSIS_UNAVAILABLE", "message": str(exc)}]
    path = run_dir / "coverage_report.json"
    atomic_json(path, payload)
    return path


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["BOUNDED_PLANAR_GAP", "SYMMETRY"] = "BOUNDED_PLANAR_GAP"
    source_geometry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parameters: CompletionParameters = Field(default_factory=CompletionParameters)
    reviews: tuple[GapReview, ...] = ()
    symmetry_plane: tuple[float, float, float, float] | None = None
    symmetry_confidence: float | None = Field(default=None, ge=0, le=1)
    minimum_symmetry_support: float | None = Field(default=None, ge=0, le=1)


def completion_fingerprint(source: MeshSource | None, request: CompletionRequest) -> str:
    return canonical_hash(
        {
            "contract": AI_STAGE_CONTRACT_VERSION,
            "source": source.model_dump(mode="json") if source else None,
            "request": request.model_dump(mode="json"),
            "numpy": np.__version__,
            "trimesh": trimesh.__version__,
        }
    )


def _completion_status(
    record: RunRecord,
    *,
    status: str,
    method: str | None,
    source_hash: str | None,
    fingerprint: str,
    failure_reason: str | None,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "contract_status": "INTEGRATED",
        "status": status,
        "method": method,
        "method_version": AI_STAGE_CONTRACT_VERSION if method else None,
        "source_geometry_sha256": source_hash,
        "fingerprint": fingerprint,
        "artifact_url": None,
        "artifact_sha256": None,
        "inferred_artifact_url": None,
        "inferred_artifact_sha256": None,
        "face_provenance_url": None,
        "inferred_regions_present": False,
        "observed_vertex_count": 0,
        "inferred_vertex_count": 0,
        "observed_face_count": 0,
        "inferred_face_count": 0,
        "confidence": None,
        "eligible_for_measurement": False,
        "warnings": warnings or [],
        "failure_reason": failure_reason,
        "run_id": record.run_id,
    }


def _publish_completion_status(
    record: RunRecord, run_dir: Path, payload: dict[str, object]
) -> tuple[dict[str, object], list[Path]]:
    """Version every decision while exposing one declared current-status pointer."""

    current = run_dir / "completion_status.json"
    previous: dict[str, object] = {}
    if current.is_file():
        try:
            parsed = json.loads(current.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                previous = parsed
        except (OSError, json.JSONDecodeError):
            pass
    if previous.get("status") == "completed" and previous.get("fingerprint") != payload.get(
        "fingerprint"
    ):
        payload["previous_completed_attempt"] = {
            key: previous.get(key)
            for key in (
                "fingerprint",
                "artifact_url",
                "artifact_sha256",
                "inferred_artifact_url",
                "inferred_artifact_sha256",
                "face_provenance_url",
            )
        }
    fingerprint = str(payload["fingerprint"])
    attempt_dir = run_dir / "completion" / "requests" / fingerprint
    attempt_dir.mkdir(parents=True, exist_ok=True)
    decision_hash = canonical_hash(payload)[:16]
    versioned = attempt_dir / f"status-{decision_hash}.json"
    versioned_url = (
        f"/api/runs/{record.run_id}/artifacts/"
        f"{versioned.relative_to(run_dir).as_posix()}"
    )
    payload["versioned_status_url"] = versioned_url
    atomic_json(versioned, payload)
    atomic_json(current, payload)
    return payload, [current, versioned]


def write_completion_not_run(record: RunRecord, run_dir: Path, reason: str) -> Path:
    source, _ = observed_mesh_source(record, run_dir)
    fingerprint = completion_fingerprint(source, CompletionRequest())
    payload = _completion_status(
        record,
        status="not_run",
        method=None,
        source_hash=source.artifact.sha256 if source else None,
        fingerprint=fingerprint,
        failure_reason=reason,
    )
    _, paths = _publish_completion_status(record, run_dir, payload)
    return paths[0]


def run_completion(
    record: RunRecord,
    run_dir: Path,
    request: CompletionRequest,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], list[Path], bool]:
    source, source_reason = observed_mesh_source(record, run_dir)
    fingerprint = completion_fingerprint(source, request)
    status_path = run_dir / "completion_status.json"
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("fingerprint") == fingerprint and existing.get("status") in {
            "completed",
            "partial",
            "refused",
            "unavailable",
        }:
            artifact_hash = existing.get("artifact_sha256")
            artifact_url = existing.get("artifact_url")
            relative = str(artifact_url or "").split("/artifacts/", 1)[-1]
            if not artifact_hash or (
                relative
                and (run_dir / relative).is_file()
                and sha256_file(run_dir / relative) == artifact_hash
            ):
                paths = [status_path]
                for key in ("artifact_url", "inferred_artifact_url", "face_provenance_url"):
                    value = existing.get(key)
                    if isinstance(value, str) and "/artifacts/" in value:
                        candidate = run_dir / value.split("/artifacts/", 1)[1]
                        if candidate.is_file():
                            paths.append(candidate)
                return existing, paths, True
    if source is None:
        payload = _completion_status(
            record,
            status="unavailable",
            method=request.method,
            source_hash=None,
            fingerprint=fingerprint,
            failure_reason=source_reason,
        )
        payload, status_paths = _publish_completion_status(record, run_dir, payload)
        return payload, status_paths, False
    if request.source_geometry_sha256 and request.source_geometry_sha256 != source.artifact.sha256:
        payload = _completion_status(
            record,
            status="refused",
            method=request.method,
            source_hash=source.artifact.sha256,
            fingerprint=fingerprint,
            failure_reason="Requested source geometry hash does not match the declared observed mesh",
        )
        payload, status_paths = _publish_completion_status(record, run_dir, payload)
        return payload, status_paths, False
    coverage_path = run_dir / "coverage_report.json"
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        coverage = {}
    declared = _declared(record)
    allowed_evidence = {
        (str(item.get("artifact_path")), str(item.get("sha256")))
        for item in coverage.get("review_evidence_options", [])
        if isinstance(item, dict)
        and (artifact := declared.get(str(item.get("artifact_path")))) is not None
        and artifact.sha256 == item.get("sha256")
    }
    for review in request.reviews:
        if review.decision == "CONFIRMED_SMALL_GAP" and not review.source_images:
            payload = _completion_status(
                record,
                status="refused",
                method=request.method,
                source_hash=source.artifact.sha256,
                fingerprint=fingerprint,
                failure_reason=(
                    f"Confirmed gap {review.boundary_id} requires declared selected-frame evidence"
                ),
            )
            payload, status_paths = _publish_completion_status(record, run_dir, payload)
            return payload, status_paths, False
        for image in review.source_images:
            if (image.path, image.sha256) not in allowed_evidence:
                payload = _completion_status(
                    record,
                    status="refused",
                    method=request.method,
                    source_hash=source.artifact.sha256,
                    fingerprint=fingerprint,
                    failure_reason=(
                        f"Review evidence for {review.boundary_id} is not a hash-matched "
                        "declared selected frame"
                    ),
                )
                payload, status_paths = _publish_completion_status(record, run_dir, payload)
                return payload, status_paths, False
    if request.method == "SYMMETRY":
        reason = "Symmetry completion is not implemented and no inferred geometry was produced"
        if request.symmetry_plane is None or (request.symmetry_confidence or 0) < 0.8:
            reason = "Symmetry completion refused: a high-confidence declared plane and support are required"
        payload = _completion_status(
            record,
            status="refused",
            method=request.method,
            source_hash=source.artifact.sha256,
            fingerprint=fingerprint,
            failure_reason=reason,
        )
        payload, status_paths = _publish_completion_status(record, run_dir, payload)
        return payload, status_paths, False

    with tempfile.TemporaryDirectory(prefix=f"sih-completion-{record.run_id}-") as temporary:
        bundle = Path(temporary) / "bundle"
        engine = complete_planar_gaps(
            run_dir,
            bundle,
            source,
            parameters=request.parameters,
            reviews=request.reviews,
            cancel_requested=cancel_requested,
            timeout_s=record.config.completion_timeout_s,
        )
        if engine.status != "COMPLETED":
            mapped = "refused" if engine.status == "REJECTED" else "failed"
            payload = _completion_status(
                record,
                status=mapped,
                method=request.method,
                source_hash=source.artifact.sha256,
                fingerprint=fingerprint,
                failure_reason=engine.reason,
                warnings=engine.warnings,
            )
            payload["candidate_regions"] = engine.rejected_regions
            payload, status_paths = _publish_completion_status(record, run_dir, payload)
            return payload, status_paths, False

        observed = load_observed_mesh(run_dir, source, request.parameters)
        inferred = _load_ply(bundle / "completion" / "generated_mesh.ply")
        if not isinstance(inferred, trimesh.Trimesh):
            raise TypeError("Completion output did not reopen as a mesh")
        observed_copy = observed.copy()
        observed_copy.visual.vertex_colors = np.tile(
            np.array([[190, 205, 200, 255]], dtype=np.uint8), (len(observed_copy.vertices), 1)
        )
        inferred.visual.vertex_colors = np.tile(
            np.array([[168, 85, 247, 255]], dtype=np.uint8), (len(inferred.vertices), 1)
        )
        combined = trimesh.util.concatenate((observed_copy, inferred))
        publish_root = run_dir / "completion"
        publish_root.mkdir(exist_ok=True)
        final = publish_root / fingerprint[:16]
        if not final.exists():
            staging = publish_root / f".{fingerprint[:16]}-{uuid.uuid4().hex}.tmp"
            staging.mkdir()
            inferred_path = staging / "inferred_geometry.ply"
            completed_path = staging / "completed_geometry.ply"
            inferred_path.write_bytes(
                trimesh.exchange.ply.export_ply(
                    inferred, encoding="binary_little_endian", vertex_normal=False
                )
            )
            completed_path.write_bytes(
                trimesh.exchange.ply.export_ply(
                    combined, encoding="binary_little_endian", vertex_normal=False
                )
            )
            shutil.copy2(bundle / "completion" / "provenance.json", staging / "provenance.json")
            shutil.copy2(bundle / "completion" / "report.json", staging / "engine_report.json")
            face_provenance = {
                "schema_version": "1.0",
                "source_geometry": source.artifact.model_dump(mode="json"),
                "completed_geometry_sha256": sha256_file(completed_path),
                "observed_face_ids": list(range(len(observed.faces))),
                "inferred_face_ids": list(range(len(observed.faces), len(combined.faces))),
                "measurement_eligible": False,
                "confidence_label": "AI_ASSISTED_NOT_MEASURABLE",
            }
            atomic_json(staging / "face_provenance.json", face_provenance)
            reopened = _load_ply(completed_path)
            if not isinstance(reopened, trimesh.Trimesh) or len(reopened.faces) != len(
                combined.faces
            ):
                raise ValueError("Completed geometry failed independent reopen validation")
            if not np.array_equal(
                np.asarray(reopened.vertices)[: len(observed.vertices)], observed.vertices
            ):
                raise ValueError("Completed geometry changed observed vertex coordinates")
            os.replace(staging, final)
        completed_path = final / "completed_geometry.ply"
        inferred_path = final / "inferred_geometry.ply"
        base = f"/api/runs/{record.run_id}/artifacts/{final.relative_to(run_dir).as_posix()}"
        payload = _completion_status(
            record,
            status="completed",
            method=request.method,
            source_hash=source.artifact.sha256,
            fingerprint=fingerprint,
            failure_reason=None,
            warnings=engine.warnings,
        )
        payload.update(
            artifact_url=f"{base}/completed_geometry.ply",
            artifact_sha256=sha256_file(completed_path),
            inferred_artifact_url=f"{base}/inferred_geometry.ply",
            inferred_artifact_sha256=sha256_file(inferred_path),
            face_provenance_url=f"{base}/face_provenance.json",
            inferred_regions_present=True,
            observed_vertex_count=len(observed.vertices),
            inferred_vertex_count=len(inferred.vertices),
            observed_face_count=len(observed.faces),
            inferred_face_count=len(inferred.faces),
            confidence={
                "label": "RULE_BASED_NOT_CALIBRATED_PROBABILITY",
                "value": None,
            },
        )
        payload, status_paths = _publish_completion_status(record, run_dir, payload)
        paths = [*status_paths, *sorted(path for path in final.iterdir() if path.is_file())]
        return payload, paths, False
