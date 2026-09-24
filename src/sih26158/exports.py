from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import struct
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .confidence import validate_point_confidence_for_ply
from .models import ArtifactEntry, ProvenanceOrigin, RunRecord, utc_now
from .storage import atomic_json, sha256_file

EXPORTER_NAME = "sih26158.geometry_exports"
EXPORTER_VERSION = "1.0"
EXPORT_CONTRACT_VERSION = "2.0"
LAS_TARGET_SCALE_M = 0.0001
MAX_EXPORT_SOURCE_BYTES = 500_000_000
MAX_EXPORT_VERTICES = 5_000_000
MAX_EXPORT_FACES = 10_000_000

# Local evidence is right-handed ENU: X east, Y north, Z up. glTF is
# conventionally consumed as X right, Y up, -Z forward. This proper rotation
# maps ENU -> glTF as (east, up, -north) and preserves lengths and handedness.
ENU_TO_GLTF = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
GLTF_TO_ENU = np.linalg.inv(ENU_TO_GLTF)


class ExportUnavailable(ValueError):
    """A declared source cannot be exported without weakening its contract."""


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _artifact_url(record: RunRecord, relative_path: str) -> str:
    return f"/api/runs/{record.run_id}/artifacts/{relative_path}"


def _declared(record: RunRecord) -> dict[str, ArtifactEntry]:
    return {item.relative_path: item for item in record.artifacts}


def _verified_artifact(record: RunRecord, run_dir: Path, relative_path: str) -> Path:
    artifact = _declared(record).get(relative_path)
    if artifact is None:
        raise ExportUnavailable(f"Source artifact is not declared: {relative_path}")
    path = (run_dir / relative_path).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
        raise ExportUnavailable(f"Declared source artifact is missing: {relative_path}")
    if sha256_file(path) != artifact.sha256:
        raise ExportUnavailable(f"Declared source artifact checksum changed: {relative_path}")
    return path


def _ply_texture_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("rb") as stream:
        for raw in stream:
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ExportUnavailable("PLY header is not ASCII") from exc
            if line.startswith("comment TextureFile "):
                name = line.removeprefix("comment TextureFile ").strip()
                if Path(name).name != name or name in {"", ".", ".."}:
                    raise ExportUnavailable("PLY texture reference must be a local basename")
                names.append(name)
            if line == "end_header":
                return names
    raise ExportUnavailable("PLY is missing end_header")


def _source_provenance(relative_path: str) -> str:
    lowered = relative_path.lower()
    if any(token in lowered for token in ("inferred", "completion", "generated")):
        return "INFERRED"
    if relative_path in {"sparse/sparse_local.ply", "sparse/sparse.ply"}:
        return "OBSERVED"
    if relative_path.startswith("dense/"):
        return "DERIVED_OBSERVED_VISUAL"
    return "UNKNOWN"


def _source_measurement_eligible(record: RunRecord, relative_path: str) -> bool:
    return bool(
        relative_path == "sparse/sparse_local.ply"
        and record.source_provenance == ProvenanceOrigin.REAL
        and record.config.execution_mode != "SYNTHETIC_DEMO"
    )


def _coordinate_contract(record: RunRecord, run_dir: Path) -> dict[str, Any]:
    declared = _declared(record)
    transform: dict[str, Any] = {}
    transform_hash: str | None = None
    if "local_transform.json" in declared:
        path = _verified_artifact(record, run_dir, "local_transform.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                transform = payload
        except (OSError, json.JSONDecodeError):
            transform = {}
        transform_hash = declared["local_transform.json"].sha256
    altitude_reference = transform.get("altitude_reference", "UNKNOWN_OR_UNDECLARED")
    return {
        "source_frame": "LOCAL_ENU_METRES",
        "units": "metre",
        "axis_convention": {"x": "east", "y": "north", "z": "up"},
        "origin_wgs84": transform.get("origin_wgs84"),
        "epsg": None,
        "epsg_reason": "Local ENU is not assigned a fabricated projected CRS/EPSG code.",
        "altitude_reference": altitude_reference,
        "vertical_datum_status": (
            "DECLARED_NON_SURVEY_OR_UNKNOWN"
            if str(altitude_reference).lower()
            in {"unknown", "unknown_or_undeclared", "relative", "relative_to_launch"}
            else "DECLARED_BY_SOURCE_METADATA_NOT_INDEPENDENTLY_VERIFIED"
        ),
        "local_transform_artifact": (
            {"path": "local_transform.json", "sha256": transform_hash}
            if transform_hash
            else None
        ),
        "local_transform_metadata": (
            {
                name: transform.get(name)
                for name in (
                    "coordinate_frame",
                    "scale",
                    "rotation",
                    "translation_m",
                )
                if name in transform
            }
            if transform_hash
            else None
        ),
        "scale_and_geolocation_statement": (
            "Coordinates retain the run's local metric alignment. Export does not improve or "
            "independently validate scale, absolute geolocation, or vertical datum."
        ),
    }


def _raw_vertex_properties(geometry: object) -> tuple[dict[str, np.ndarray], list[str]]:
    metadata = getattr(geometry, "metadata", {})
    raw = metadata.get("_ply_raw", {}) if isinstance(metadata, dict) else {}
    vertex = raw.get("vertex", {}) if isinstance(raw, dict) else {}
    data = vertex.get("data", {}) if isinstance(vertex, dict) else {}
    if not isinstance(data, dict):
        return {}, []
    values = {name: np.asarray(value).reshape(-1) for name, value in data.items()}
    las_supported = {
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "classification",
    }
    return values, sorted(set(values) - las_supported)


def _unsupported_mesh_attributes(geometry: object) -> dict[str, list[str]]:
    metadata = getattr(geometry, "metadata", {})
    raw = metadata.get("_ply_raw", {}) if isinstance(metadata, dict) else {}
    supported = {
        "vertex": {
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "texture_u",
            "texture_v",
            "u",
            "v",
            "s",
            "t",
        },
        # OpenMVS textured PLY stores per-corner UV values and atlas indices
        # on faces; trimesh expands these to exportable per-vertex UVs.
        "face": {"vertex_indices", "texcoord", "texnumber"},
    }
    unsupported: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return unsupported
    for element, body in raw.items():
        if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
            continue
        names = sorted(set(body["data"]) - supported.get(str(element), set()))
        if names:
            unsupported[str(element)] = names
    return unsupported


def _load_ply(path: Path) -> trimesh.Trimesh | trimesh.points.PointCloud:
    try:
        geometry = trimesh.load(path, process=False, maintain_order=True)
    except Exception as exc:  # format-specific exceptions vary by trimesh release
        raise ExportUnavailable(f"PLY could not be parsed: {exc}") from exc
    if isinstance(geometry, trimesh.Scene):
        if len(geometry.geometry) == 0:
            raise ExportUnavailable("Geometry contains no vertices")
        if len(geometry.geometry) != 1:
            raise ExportUnavailable("PLY unexpectedly contains multiple geometries")
        geometry = next(iter(geometry.geometry.values()))
    if not isinstance(geometry, (trimesh.Trimesh, trimesh.points.PointCloud)):
        raise ExportUnavailable("PLY did not reopen as a supported mesh or point cloud")
    vertices = np.asarray(geometry.vertices, dtype=float)
    if len(vertices) == 0:
        raise ExportUnavailable("Geometry contains no vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ExportUnavailable("Geometry contains malformed or non-finite coordinates")
    if isinstance(geometry, trimesh.Trimesh):
        faces = np.asarray(geometry.faces)
        if faces.size and (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ExportUnavailable("Mesh contains malformed or out-of-range faces")
    return geometry


def _validate_ply(path: Path) -> dict[str, Any]:
    geometry = _load_ply(path)
    vertices = np.asarray(geometry.vertices, dtype=float)
    face_count = len(geometry.faces) if isinstance(geometry, trimesh.Trimesh) else 0
    return {
        "reader": "trimesh_ply_loader",
        "geometry_kind": "MESH" if face_count else "POINT_CLOUD",
        "vertex_count": len(vertices),
        "face_count": face_count,
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "finite_coordinates": True,
    }


def _texture_dependencies(
    record: RunRecord, run_dir: Path, source_relative: str, source_path: Path
) -> list[tuple[str, Path, ArtifactEntry]]:
    names = _ply_texture_names(source_path)
    if len(names) > 1:
        raise ExportUnavailable(
            "Multi-atlas textured PLY export is not yet supported; no atlas was silently dropped."
        )
    dependencies: list[tuple[str, Path, ArtifactEntry]] = []
    declared = _declared(record)
    for name in names:
        relative = str(Path(source_relative).parent / name)
        artifact = declared.get(relative)
        if artifact is None:
            raise ExportUnavailable(f"Mesh references undeclared texture artifact: {relative}")
        path = _verified_artifact(record, run_dir, relative)
        dependencies.append((relative, path, artifact))
    return dependencies


def _mesh_context(
    record: RunRecord, run_dir: Path, relative_path: str
) -> tuple[trimesh.Trimesh, Path, list[tuple[str, Path, ArtifactEntry]]]:
    source = _verified_artifact(record, run_dir, relative_path)
    if source.stat().st_size > MAX_EXPORT_SOURCE_BYTES:
        raise ExportUnavailable("Mesh exceeds the configured export source-byte limit")
    textures = _texture_dependencies(record, run_dir, relative_path, source)
    geometry = _load_ply(source)
    if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
        raise ExportUnavailable(
            "Source is a point cloud without declared faces; sparse points are not triangulated."
        )
    if len(geometry.vertices) > MAX_EXPORT_VERTICES or len(geometry.faces) > MAX_EXPORT_FACES:
        raise ExportUnavailable(
            "Mesh exceeds configured vertex/face export limits; use a validated bounded input"
        )
    if textures:
        uv = getattr(geometry.visual, "uv", None)
        if uv is None or np.asarray(uv).shape != (len(geometry.vertices), 2):
            raise ExportUnavailable(
                "Textured mesh has no supported per-vertex UV coordinates; texture was not dropped."
            )
        material = getattr(geometry.visual, "material", None)
        image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        if image is None:
            raise ExportUnavailable("Declared texture could not be decoded by the mesh reader")
    return geometry, source, textures


def _package_outputs(package: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(package)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "export_manifest.json"
    ]


def _manifest_reusable(package: Path, cache_key: str) -> dict[str, Any] | None:
    manifest_path = package / "export_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("cache_key") != cache_key:
        return None
    source = manifest.get("source_artifact", {})
    dependencies = manifest.get("source_dependencies", [])
    if not isinstance(source, dict) or not isinstance(dependencies, list):
        return None
    dependency_hashes = {
        str(item.get("path")): str(item.get("sha256"))
        for item in dependencies
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    }
    manifest_cache_material = {
        "exporter": manifest.get("exporter"),
        "exporter_version": manifest.get("exporter_version"),
        "format": manifest.get("format"),
        "source_artifact_path": source.get("path"),
        "source_artifact_sha256": source.get("sha256"),
        "source_dependencies": dependency_hashes,
        "options": manifest.get("options"),
        "coordinate_contract": manifest.get("coordinate_contract"),
        "geometry_provenance": manifest.get("geometry_provenance"),
        "measurement_eligible": manifest.get("measurement_eligible"),
    }
    if _json_hash(manifest_cache_material) != cache_key:
        return None
    generated = manifest.get("generated_files", [])
    if not isinstance(generated, list):
        return None
    root = package.resolve()
    for output in generated:
        if not isinstance(output, dict):
            return None
        relative = str(output.get("path", ""))
        path = (package / relative).resolve()
        if (
            not relative
            or not path.is_relative_to(root)
            or not path.is_file()
            or sha256_file(path) != output.get("sha256")
        ):
            return None
    return manifest


def _publish_package(
    *,
    record: RunRecord,
    run_dir: Path,
    format_name: str,
    variant: str,
    source_relative: str,
    source_hash: str,
    dependency_hashes: dict[str, str],
    options: dict[str, Any],
    coordinate_contract: dict[str, Any],
    geometry_provenance: str,
    measurement_eligible: bool,
    build: Callable[[Path], dict[str, Any]],
    validate: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    cache_material = {
        "exporter": EXPORTER_NAME,
        "exporter_version": EXPORTER_VERSION,
        "format": format_name,
        "source_artifact_path": source_relative,
        "source_artifact_sha256": source_hash,
        "source_dependencies": dependency_hashes,
        "options": options,
        # Origin, vertical-reference, and transform metadata are part of the
        # product contract even when the geometry bytes themselves are
        # unchanged. A package with stale coordinate metadata is not reusable.
        "coordinate_contract": coordinate_contract,
        "geometry_provenance": geometry_provenance,
        "measurement_eligible": measurement_eligible,
    }
    cache_key = _json_hash(cache_material)
    variant_slug = re.sub(r"[^a-z0-9_-]+", "-", variant)
    base = run_dir / "exports" / format_name.lower() / variant_slug
    base.mkdir(parents=True, exist_ok=True)
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith(".staging-"):
            continue
        manifest = _manifest_reusable(candidate, cache_key)
        if manifest is None:
            continue
        try:
            validation = validate(candidate)
        except ExportUnavailable:
            continue
        return _report_export(record, run_dir, candidate, manifest, validation, reused=True)

    staging = base / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        build_metadata = build(staging)
        validation = validate(staging)
        if sha256_file(run_dir / source_relative) != source_hash:
            raise ExportUnavailable("Source artifact changed during export")
        generated = _package_outputs(staging)
        manifest = {
            "schema_version": "1.0",
            "export_contract_version": EXPORT_CONTRACT_VERSION,
            "generated_at": utc_now(),
            "exporter": EXPORTER_NAME,
            "exporter_version": EXPORTER_VERSION,
            "format": format_name,
            "variant": variant,
            "cache_key": cache_key,
            "source_artifact": {"path": source_relative, "sha256": source_hash},
            "source_dependencies": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(dependency_hashes.items())
            ],
            "options": options,
            "coordinate_contract": coordinate_contract,
            "geometry_provenance": geometry_provenance,
            "measurement_eligible": measurement_eligible,
            "measurement_api_supported": False,
            "measurement_statement": (
                "Export preserves the source geometry classification but is not accepted by the "
                "current measurement API; export success never promotes visual/inferred geometry."
            ),
            "build_metadata": build_metadata,
            "validation": validation,
            "generated_files": generated,
        }
        atomic_json(staging / "export_manifest.json", manifest)
        destination = base / cache_key[:16]
        if destination.exists():
            destination = base / f"{cache_key[:16]}-repair-{uuid.uuid4().hex[:8]}"
        os.replace(staging, destination)
        return _report_export(record, run_dir, destination, manifest, validation, reused=False)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _report_export(
    record: RunRecord,
    run_dir: Path,
    package: Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    files = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(run_dir))
        files.append(
            {
                "relative_path": relative,
                "url": _artifact_url(record, relative),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
            }
        )
    return {
        "status": "AVAILABLE_VALIDATED",
        "format": manifest["format"],
        "variant": manifest["variant"],
        "source_artifact": manifest["source_artifact"],
        "source_dependencies": manifest["source_dependencies"],
        "geometry_provenance": manifest["geometry_provenance"],
        "measurement_eligible": manifest["measurement_eligible"],
        "measurement_api_supported": False,
        "coordinate_contract": manifest["coordinate_contract"],
        "options": manifest["options"],
        "validation": validation,
        "reused": reused,
        "files": files,
        "manifest_url": next(
            item["url"] for item in files if item["relative_path"].endswith("export_manifest.json")
        ),
    }


def _write_obj(
    package: Path,
    mesh: trimesh.Trimesh,
    textures: list[tuple[str, Path, ArtifactEntry]],
) -> dict[str, Any]:
    obj_path = package / "model.obj"
    mtl_path = package / "material.mtl"
    texture_relative: str | None = None
    if textures:
        texture_dir = package / "textures"
        texture_dir.mkdir()
        _, source, _ = textures[0]
        destination = texture_dir / source.name
        shutil.copyfile(source, destination)
        texture_relative = f"textures/{destination.name}"

    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    uv = getattr(mesh.visual, "uv", None)
    uv_array = np.asarray(uv, dtype=float) if uv is not None else None
    colors_value = getattr(mesh.visual, "vertex_colors", None)
    colors = np.asarray(colors_value)[:, :3] if colors_value is not None else None
    # Stream the textual payload instead of materializing one string per
    # vertex/face for the complete mesh. The geometry reader remains the main
    # memory cost, but output size no longer adds another scene-sized list.
    with obj_path.open("w", encoding="utf-8") as stream:
        stream.write("# SIH26158 geometry export\n")
        stream.write("# coordinates: LOCAL_ENU_METRES; X=east Y=north Z=up\n")
        stream.write("mtllib material.mtl\nusemtl material_0\n")
        for index, vertex in enumerate(vertices):
            suffix = ""
            if colors is not None and len(colors) == len(vertices):
                rgb = colors[index].astype(float) / 255.0
                suffix = f" {rgb[0]:.8g} {rgb[1]:.8g} {rgb[2]:.8g}"
            stream.write(
                f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}{suffix}\n"
            )
        if uv_array is not None:
            for value in uv_array:
                stream.write(f"vt {value[0]:.17g} {value[1]:.17g}\n")
        for normal in normals:
            stream.write(f"vn {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}\n")
        for face in np.asarray(mesh.faces, dtype=np.int64):
            tokens = []
            for vertex_index in face:
                value = int(vertex_index) + 1
                if uv_array is not None:
                    tokens.append(f"{value}/{value}/{value}")
                else:
                    tokens.append(f"{value}//{value}")
            stream.write("f " + " ".join(tokens) + "\n")
    mtl = ["newmtl material_0", "Ka 0 0 0", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1"]
    if texture_relative:
        mtl.append(f"map_Kd {texture_relative}")
    mtl_path.write_text("\n".join(mtl) + "\n", encoding="utf-8")
    return {
        "material_mode": "PHOTOGRAPHIC_TEXTURE" if texture_relative else "UNTEXTURED",
        "texture_path": texture_relative,
        "vertex_color_extension": colors is not None,
    }


def _validate_obj(package: Path, expected_texture: bool) -> dict[str, Any]:
    path = package / "model.obj"
    try:
        mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    except Exception as exc:
        raise ExportUnavailable(f"OBJ reopen failed: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ExportUnavailable("OBJ reopen did not produce a non-empty mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not np.isfinite(vertices).all():
        raise ExportUnavailable("OBJ reopen produced non-finite coordinates")
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    if expected_texture and image is None:
        raise ExportUnavailable("OBJ reopen did not preserve the photographic texture")
    return {
        "reader": "trimesh_obj_loader",
        "vertex_count": len(vertices),
        "face_count": len(mesh.faces),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "finite_coordinates": True,
        "texture_preserved": image is not None,
        "coordinate_tolerance_m": 1e-9,
    }


def _write_glb(package: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    converted = mesh.copy()
    converted.apply_transform(ENU_TO_GLTF)
    payload = trimesh.exchange.gltf.export_glb(
        trimesh.Scene(converted),
        include_normals=True,
    )
    (package / "model.glb").write_bytes(payload)
    return {
        "material_mode": (
            "PHOTOGRAPHIC_TEXTURE" if getattr(mesh.visual, "kind", None) == "texture" else "UNTEXTURED"
        ),
        "axis_transform_enu_to_gltf": ENU_TO_GLTF.tolist(),
        "axis_transform_gltf_to_enu": GLTF_TO_ENU.tolist(),
    }


def _glb_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ExportUnavailable("GLB has an invalid binary header")
    version, total_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or total_length != len(raw):
        raise ExportUnavailable("GLB version/length header is invalid")
    json_length, chunk_type = struct.unpack_from("<II", raw, 12)
    if chunk_type != 0x4E4F534A or 20 + json_length > len(raw):
        raise ExportUnavailable("GLB is missing its JSON chunk")
    try:
        payload = json.loads(raw[20 : 20 + json_length].decode("utf-8").rstrip(" \t\r\n\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportUnavailable("GLB JSON chunk is invalid") from exc
    if not isinstance(payload, dict):
        raise ExportUnavailable("GLB JSON root is invalid")
    return payload


def _validate_glb(package: Path, expected_texture: bool) -> dict[str, Any]:
    path = package / "model.glb"
    payload = _glb_json(path)
    primitives = [
        primitive
        for mesh in payload.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    if not primitives:
        raise ExportUnavailable("GLB contains no mesh primitives")
    accessors = payload.get("accessors", [])
    vertex_count = 0
    face_count = 0
    bounds: list[list[float]] | None = None
    for primitive in primitives:
        position_index = primitive.get("attributes", {}).get("POSITION")
        if not isinstance(position_index, int) or position_index >= len(accessors):
            raise ExportUnavailable("GLB primitive has no valid POSITION accessor")
        accessor = accessors[position_index]
        vertex_count += int(accessor.get("count", 0))
        minimum, maximum = accessor.get("min"), accessor.get("max")
        if not (
            isinstance(minimum, list)
            and isinstance(maximum, list)
            and len(minimum) == len(maximum) == 3
            and all(math.isfinite(float(value)) for value in [*minimum, *maximum])
        ):
            raise ExportUnavailable("GLB POSITION bounds are missing or non-finite")
        if bounds is None:
            bounds = [[float(value) for value in minimum], [float(value) for value in maximum]]
        index_accessor = primitive.get("indices")
        if isinstance(index_accessor, int) and index_accessor < len(accessors):
            face_count += int(accessors[index_accessor].get("count", 0)) // 3
    texture_preserved = bool(payload.get("textures") and payload.get("images"))
    if expected_texture and not texture_preserved:
        raise ExportUnavailable("GLB did not embed the declared photographic texture")
    return {
        "reader": "independent_glb2_chunk_and_accessor_parser",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds_gltf": bounds,
        "finite_coordinates": True,
        "texture_preserved": texture_preserved,
        "coordinate_tolerance_m": 1e-6,
    }


def _vertex_colors(geometry: object, count: int) -> np.ndarray | None:
    visual = getattr(geometry, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        colors = getattr(geometry, "colors", None)
    if colors is None:
        return None
    values = np.asarray(colors)
    if values.ndim != 2 or len(values) != count or values.shape[1] < 3:
        return None
    return np.clip(values[:, :3], 0, 255).astype(np.uint8)


def _las_scale_offset(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    scale = np.maximum(LAS_TARGET_SCALE_M, extent / (2_147_483_647 * 1.9))
    offset = (minimum + maximum) / 2.0
    return scale, offset


def _write_las(
    package: Path,
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    provenance_sidecar: dict[str, Any],
) -> dict[str, Any]:
    points = np.asarray(geometry.vertices, dtype=np.float64)
    if len(points) > 0xFFFFFFFF:
        raise ExportUnavailable("LAS 1.2 exporter cannot represent more than 2^32-1 points")
    colors = _vertex_colors(geometry, len(points))
    raw_properties, unsupported = _raw_vertex_properties(geometry)
    classifications = raw_properties.get("classification")
    if classifications is None:
        classification_values = np.zeros(len(points), dtype=np.uint8)
    else:
        if len(classifications) != len(points) or not np.isfinite(classifications).all():
            raise ExportUnavailable("PLY classification property is malformed")
        if not np.equal(classifications, np.floor(classifications)).all():
            raise ExportUnavailable("PLY classification values must be integers")
        if np.any(classifications < 0) or np.any(classifications > 31):
            raise ExportUnavailable(
                "LAS 1.2 point format 2 supports classification values 0-31; source exceeds range."
            )
        classification_values = classifications.astype(np.uint8)

    provenance_sidecar = provenance_sidecar | {
        "las_classification": {
            "source_property_present": classifications is not None,
            "preserved_in_standard_dimension": classifications is not None,
            "default_when_absent": "0 (created, never classified)",
        },
        "photographic_rgb": {
            "source_present": colors is not None,
            "encoding": "LAS uint16 RGB; uint8 source values multiplied by 257",
        },
        "unsupported_source_vertex_attributes": unsupported,
        "unsupported_attribute_policy": (
            "Names are disclosed here and values remain hash-bound in the unchanged source PLY; "
            "they are not silently relabelled as LAS standard dimensions."
        ),
    }
    atomic_json(package / "provenance.json", provenance_sidecar)
    sidecar_hash = sha256_file(package / "provenance.json")
    vlr_payload = json.dumps(
        {
            "schema": "SIH26158_EXPORT_PROVENANCE_V1",
            "provenance_sidecar": "provenance.json",
            "provenance_sidecar_sha256": sidecar_hash,
            "source_artifact": provenance_sidecar["source_artifact"],
            "geometry_provenance": provenance_sidecar["geometry_provenance"],
            "coordinate_frame": "LOCAL_ENU_METRES",
            "epsg": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(vlr_payload) > 65535:
        raise ExportUnavailable("LAS provenance VLR exceeds the LAS 1.2 size limit")

    scale, offset = _las_scale_offset(points)
    integer_minimum = np.rint((points.min(axis=0) - offset) / scale).astype(np.int64)
    integer_maximum = np.rint((points.max(axis=0) - offset) / scale).astype(np.int64)
    if np.any(integer_minimum < -2_147_483_648) or np.any(
        integer_maximum > 2_147_483_647
    ):
        raise ExportUnavailable("LAS coordinate quantization exceeds signed 32-bit storage")
    minimum = integer_minimum * scale + offset
    maximum = integer_maximum * scale + offset
    point_data_offset = 227 + 54 + len(vlr_payload)
    header = bytearray(227)
    header[:4] = b"LASF"
    struct.pack_into("<H", header, 4, 0)
    struct.pack_into("<H", header, 6, 0)
    header[24:26] = bytes((1, 2))
    header[26:58] = b"SIH26158 LOCAL ENU".ljust(32, b"\x00")
    header[58:90] = f"{EXPORTER_NAME} {EXPORTER_VERSION}".encode("ascii")[:32].ljust(
        32, b"\x00"
    )
    now = datetime.now(UTC)
    struct.pack_into("<H", header, 90, int(now.strftime("%j")))
    struct.pack_into("<H", header, 92, now.year)
    struct.pack_into("<H", header, 94, 227)
    struct.pack_into("<I", header, 96, point_data_offset)
    struct.pack_into("<I", header, 100, 1)
    struct.pack_into("<B", header, 104, 2)
    struct.pack_into("<H", header, 105, 26)
    struct.pack_into("<I", header, 107, len(points))
    struct.pack_into("<5I", header, 111, len(points), 0, 0, 0, 0)
    struct.pack_into("<3d", header, 131, *scale)
    struct.pack_into("<3d", header, 155, *offset)
    struct.pack_into(
        "<6d",
        header,
        179,
        maximum[0],
        minimum[0],
        maximum[1],
        minimum[1],
        maximum[2],
        minimum[2],
    )
    vlr_header = struct.pack(
        "<H16sHH32s",
        0,
        b"SIH26158".ljust(16, b"\x00"),
        1,
        len(vlr_payload),
        b"Geometry provenance JSON".ljust(32, b"\x00"),
    )
    output = package / "cloud.las"
    with output.open("wb") as stream:
        stream.write(header)
        stream.write(vlr_header)
        stream.write(vlr_payload)
        for index, point in enumerate(points):
            coordinate = tuple(
                round((float(point[axis]) - offset[axis]) / scale[axis])
                for axis in range(3)
            )
            rgb = colors[index].astype(np.uint16) * 257 if colors is not None else np.zeros(3, dtype=np.uint16)
            stream.write(
                struct.pack(
                    "<iiiHBBbBH3H",
                    int(coordinate[0]),
                    int(coordinate[1]),
                    int(coordinate[2]),
                    0,
                    0,
                    int(classification_values[index]),
                    0,
                    0,
                    0,
                    int(rgb[0]),
                    int(rgb[1]),
                    int(rgb[2]),
                )
            )
    return {
        "las_version": "1.2",
        "point_data_format": 2,
        "coordinate_scale_m": scale.tolist(),
        "coordinate_offset_m": offset.tolist(),
        "photographic_rgb_preserved": colors is not None,
        "classification_preserved": classifications is not None,
        "provenance_vlr": {
            "user_id": "SIH26158",
            "record_id": 1,
            "sidecar_sha256": sidecar_hash,
        },
    }


def _read_las_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(227)
        if len(header) != 227 or header[:4] != b"LASF":
            raise ExportUnavailable("LAS reopen found an invalid header")
        version = f"{header[24]}.{header[25]}"
        point_offset = struct.unpack_from("<I", header, 96)[0]
        vlr_count = struct.unpack_from("<I", header, 100)[0]
        point_format = header[104] & 0x3F
        record_length = struct.unpack_from("<H", header, 105)[0]
        count = struct.unpack_from("<I", header, 107)[0]
        scale = np.array(struct.unpack_from("<3d", header, 131))
        offset = np.array(struct.unpack_from("<3d", header, 155))
        if version != "1.2" or point_format != 2 or record_length != 26 or count == 0:
            raise ExportUnavailable("LAS reopen found an unsupported or empty point layout")
        vlrs = []
        for _ in range(vlr_count):
            raw = stream.read(54)
            if len(raw) != 54:
                raise ExportUnavailable("LAS VLR header is truncated")
            _, user_id, record_id, length, description = struct.unpack("<H16sHH32s", raw)
            payload = stream.read(length)
            if len(payload) != length:
                raise ExportUnavailable("LAS VLR payload is truncated")
            vlrs.append(
                {
                    "user_id": user_id.rstrip(b"\x00").decode("ascii"),
                    "record_id": record_id,
                    "description": description.rstrip(b"\x00").decode("ascii"),
                    "payload": payload,
                }
            )
        stream.seek(point_offset)
        minimum = np.full(3, np.inf)
        maximum = np.full(3, -np.inf)
        color_present = False
        classifications: set[int] = set()
        for _ in range(count):
            raw = stream.read(record_length)
            if len(raw) != record_length:
                raise ExportUnavailable("LAS point records are truncated")
            values = struct.unpack("<iiiHBBbBH3H", raw)
            point = np.array(values[:3], dtype=float) * scale + offset
            if not np.isfinite(point).all():
                raise ExportUnavailable("LAS reopen produced non-finite coordinates")
            minimum = np.minimum(minimum, point)
            maximum = np.maximum(maximum, point)
            classifications.add(int(values[5]))
            color_present = color_present or any(values[-3:])
    provenance = next(
        (item for item in vlrs if item["user_id"] == "SIH26158" and item["record_id"] == 1),
        None,
    )
    if provenance is None:
        raise ExportUnavailable("LAS provenance VLR is missing")
    try:
        provenance_payload = json.loads(provenance["payload"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportUnavailable("LAS provenance VLR is invalid") from exc
    return {
        "reader": "independent_las_1_2_point_format_2_parser",
        "version": version,
        "point_format": point_format,
        "point_count": count,
        "bounds": [minimum.tolist(), maximum.tolist()],
        "coordinate_scale_m": scale.tolist(),
        "coordinate_offset_m": offset.tolist(),
        "coordinate_tolerance_m": float(scale.max() / 2),
        "finite_coordinates": True,
        "photographic_rgb_present": color_present,
        "classification_values": sorted(classifications),
        "provenance_vlr": provenance_payload,
        "epsg_assigned": False,
    }


def _mesh_candidates(record: RunRecord, *, include_inferred: bool = False) -> list[str]:
    declared = _declared(record)
    priority = [
        "dense/textured/model.ply",
        "dense/textured/textured.ply",
        "dense/meshed-openmvs-refined.ply",
        "dense/meshed-poisson-simplified.ply",
        "dense/meshed-openmvs.ply",
        "dense/meshed-poisson.ply",
    ]
    candidates = [path for path in priority if path in declared]
    candidates.extend(
        path
        for path in sorted(declared)
        if path.lower().endswith(".ply")
        and (
            path.startswith("dense/textured/")
            or Path(path).name.lower().startswith("meshed-")
        )
        and path not in candidates
    )
    inferred = [
        path
        for path in sorted(declared)
        if Path(path).name.lower() == "inferred_geometry.ply"
    ]
    return list(dict.fromkeys([*candidates, *inferred] if include_inferred else candidates))


def _point_candidates(record: RunRecord, *, include_inferred: bool = False) -> list[str]:
    declared = _declared(record)
    candidates = [
        path
        for path in ("sparse/sparse_local.ply", "dense/fused.ply")
        if path in declared
    ]
    inferred = [
        path
        for path in sorted(declared)
        if Path(path).name.lower() == "inferred_geometry.ply"
        and path not in candidates
    ]
    if include_inferred and inferred:
        candidates.append(inferred[0])
    return candidates


def _format_summary(exports: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    if exports:
        return {
            "status": "AVAILABLE_VALIDATED",
            "exports": exports,
            "failures": failures,
            "download_files": [file for item in exports for file in item["files"]],
        }
    return {
        "status": "UNAVAILABLE" if not failures else "INVALID_OR_UNSUPPORTED",
        "reason": (
            "No declared compatible source geometry exists."
            if not failures
            else "; ".join(item["reason"] for item in failures)
        ),
        "exports": [],
        "failures": failures,
        "download_files": [],
    }


def _export_mesh_formats(
    record: RunRecord,
    run_dir: Path,
    coordinate_contract: dict[str, Any],
    *,
    include_inferred: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obj_exports: list[dict[str, Any]] = []
    glb_exports: list[dict[str, Any]] = []
    obj_failures: list[dict[str, str]] = []
    glb_failures: list[dict[str, str]] = []
    candidates = _mesh_candidates(record, include_inferred=include_inferred)
    for source_relative in candidates:
        try:
            mesh, source_path, textures = _mesh_context(record, run_dir, source_relative)
            source_hash = _declared(record)[source_relative].sha256
            dependency_hashes = {relative: artifact.sha256 for relative, _, artifact in textures}
            provenance = _source_provenance(source_relative)
            eligible = _source_measurement_eligible(record, source_relative)
            variant = (
                "completed"
                if Path(source_relative).name == "completed_geometry.ply"
                else "inferred"
                if provenance == "INFERRED"
                else "visual-mesh"
            )
            completion_metadata: dict[str, Any] | None = None
            if provenance == "INFERRED" and "completion_status.json" in _declared(record):
                completion_path = _verified_artifact(record, run_dir, "completion_status.json")
                completion_metadata = json.loads(completion_path.read_text(encoding="utf-8"))
                inferred_url = completion_metadata.get("inferred_artifact_url")
                expected_relative = (
                    str(inferred_url).split("/artifacts/", 1)[1]
                    if isinstance(inferred_url, str) and "/artifacts/" in inferred_url
                    else None
                )
                if completion_metadata.get("status") != "completed" or expected_relative != source_relative:
                    raise ExportUnavailable(
                        "Inferred geometry is not selected by the current completed attempt; "
                        "choose or rerun that attempt before exporting it."
                    )
                dependency_hashes["completion_status.json"] = _declared(record)[
                    "completion_status.json"
                ].sha256
            unsupported_attributes = _unsupported_mesh_attributes(mesh)
            common_options = {
                "source_coordinate_frame": "LOCAL_ENU_METRES",
                "units": "metre",
                "face_policy": "triangulate_existing_polygon_faces_only",
                "sparse_point_triangulation": False,
                "photographic_texture_required": bool(textures),
                "unsupported_source_attributes": unsupported_attributes,
                "unsupported_attribute_policy": (
                    "Property names are disclosed in this hash-bound manifest and values remain "
                    "in the unchanged source PLY; they are not silently relabelled."
                ),
                "completion_metadata": completion_metadata,
            }
            obj_exports.append(
                _publish_package(
                    record=record,
                    run_dir=run_dir,
                    format_name="OBJ",
                    variant=variant,
                    source_relative=source_relative,
                    source_hash=source_hash,
                    dependency_hashes=dependency_hashes,
                    options=common_options
                    | {
                        "output_axis_convention": "X_EAST_Y_NORTH_Z_UP",
                        "axis_transform": "IDENTITY",
                    },
                    coordinate_contract=coordinate_contract
                    | {
                        "output_frame": "LOCAL_ENU_METRES",
                        "axis_transform_source_to_output": np.eye(4).tolist(),
                        "axis_transform_output_to_source": np.eye(4).tolist(),
                    },
                    geometry_provenance=provenance,
                    measurement_eligible=eligible,
                    build=lambda package, current=mesh, current_textures=textures: _write_obj(
                        package, current, current_textures
                    ),
                    validate=lambda package, expected=bool(textures): _validate_obj(
                        package, expected
                    ),
                )
            )
            glb_exports.append(
                _publish_package(
                    record=record,
                    run_dir=run_dir,
                    format_name="GLB",
                    variant=variant,
                    source_relative=source_relative,
                    source_hash=source_hash,
                    dependency_hashes=dependency_hashes,
                    options=common_options
                    | {
                        "output_axis_convention": "X_EAST_Y_UP_Z_SOUTH",
                        "axis_transform_enu_to_gltf": ENU_TO_GLTF.tolist(),
                    },
                    coordinate_contract=coordinate_contract
                    | {
                        "output_frame": "GLTF_VIEWER_METRES",
                        "output_axis_convention": {
                            "x": "east",
                            "y": "up",
                            "z": "south (negative north)",
                        },
                        "axis_transform_source_to_output": ENU_TO_GLTF.tolist(),
                        "axis_transform_output_to_source": GLTF_TO_ENU.tolist(),
                    },
                    geometry_provenance=provenance,
                    measurement_eligible=eligible,
                    build=lambda package, current=mesh: _write_glb(package, current),
                    validate=lambda package, expected=bool(textures): _validate_glb(
                        package, expected
                    ),
                )
            )
            if sha256_file(source_path) != source_hash:
                raise ExportUnavailable("Source mesh changed during export validation")
        except (ExportUnavailable, OSError, ValueError) as exc:
            failure = {"source_artifact_path": source_relative, "reason": str(exc)}
            if not any(item["source_artifact"]["path"] == source_relative for item in obj_exports):
                obj_failures.append(failure)
            if not any(item["source_artifact"]["path"] == source_relative for item in glb_exports):
                glb_failures.append(failure)
    if not candidates:
        reason = (
            "No declared mesh with faces is available. Point clouds remain available as PLY/LAS; "
            "the exporter will not triangulate sparse points into an observed surface."
        )
        obj_failures.append({"source_artifact_path": "", "reason": reason})
        glb_failures.append({"source_artifact_path": "", "reason": reason})
    return _format_summary(obj_exports, obj_failures), _format_summary(glb_exports, glb_failures)


def _export_las(
    record: RunRecord,
    run_dir: Path,
    coordinate_contract: dict[str, Any],
    *,
    include_inferred: bool = False,
) -> dict[str, Any]:
    exports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source_relative in _point_candidates(record, include_inferred=include_inferred):
        try:
            source_path = _verified_artifact(record, run_dir, source_relative)
            geometry = _load_ply(source_path)
            source_hash = _declared(record)[source_relative].sha256
            provenance = _source_provenance(source_relative)
            eligible = _source_measurement_eligible(record, source_relative)
            dependency_hashes: dict[str, str] = {}
            confidence: dict[str, str] | None = None
            if (
                source_relative == "sparse/sparse_local.ply"
                and "point_confidence.json" in _declared(record)
            ):
                confidence_path = _verified_artifact(record, run_dir, "point_confidence.json")
                try:
                    validate_point_confidence_for_ply(confidence_path, source_path)
                except ValueError as exc:
                    raise ExportUnavailable(
                        f"Point-confidence sidecar exists but is invalid: {exc}"
                    ) from exc
                confidence = {
                    "path": "point_confidence.json",
                    "sha256": _declared(record)["point_confidence.json"].sha256,
                }
                dependency_hashes["point_confidence.json"] = confidence["sha256"]
            variant = (
                "observed-sparse"
                if source_relative == "sparse/sparse_local.ply"
                else "inferred"
                if provenance == "INFERRED"
                else "dense-visual"
            )
            options = {
                "las_version": "1.2",
                "point_data_format": 2,
                "target_coordinate_scale_m": LAS_TARGET_SCALE_M,
                "output_axis_convention": "X_EAST_Y_NORTH_Z_UP",
                "axis_transform": "IDENTITY",
                "epsg": None,
            }
            provenance_sidecar = {
                "schema_version": "1.0",
                "source_artifact": {"path": source_relative, "sha256": source_hash},
                "geometry_provenance": provenance,
                "measurement_eligible": eligible,
                "coordinate_contract": coordinate_contract,
                "point_confidence_artifact": confidence,
            }
            exports.append(
                _publish_package(
                    record=record,
                    run_dir=run_dir,
                    format_name="LAS",
                    variant=variant,
                    source_relative=source_relative,
                    source_hash=source_hash,
                    dependency_hashes=dependency_hashes,
                    options=options,
                    coordinate_contract=coordinate_contract
                    | {
                        "output_frame": "LOCAL_ENU_METRES",
                        "axis_transform_source_to_output": np.eye(4).tolist(),
                        "axis_transform_output_to_source": np.eye(4).tolist(),
                    },
                    geometry_provenance=provenance,
                    measurement_eligible=eligible,
                    build=lambda package, current=geometry, sidecar=provenance_sidecar: _write_las(
                        package, current, sidecar
                    ),
                    validate=lambda package: _read_las_summary(package / "cloud.las"),
                )
            )
            if sha256_file(source_path) != source_hash:
                raise ExportUnavailable("Source point cloud changed during export validation")
        except (ExportUnavailable, OSError, ValueError) as exc:
            failures.append({"source_artifact_path": source_relative, "reason": str(exc)})
    return _format_summary(exports, failures)


def build_export_readiness(
    record: RunRecord, run_dir: Path, *, include_inferred: bool = False
) -> dict[str, Any]:
    """Generate and validate reusable export packages from already-declared geometry."""

    declared = _declared(record)
    formats: dict[str, dict[str, Any]] = {}
    ply_relative = next(
        (
            name
            for name in ("sparse/sparse_local.ply", "sparse/sparse.ply")
            if name in declared
        ),
        None,
    )
    if ply_relative:
        try:
            validation = _validate_ply(_verified_artifact(record, run_dir, ply_relative))
            formats["PLY"] = {
                "status": "AVAILABLE_VALIDATED",
                "artifact_path": ply_relative,
                "artifact_url": _artifact_url(record, ply_relative),
                "artifact_sha256": declared[ply_relative].sha256,
                "coordinate_frame": (
                    "LOCAL_ENU_METRES" if ply_relative.endswith("sparse_local.ply") else "SFM_LOCAL"
                ),
                "geometry_provenance": _source_provenance(ply_relative),
                "measurement_eligible": _source_measurement_eligible(record, ply_relative),
                "validation": validation,
            }
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            formats["PLY"] = {"status": "INVALID", "reason": str(exc)}
    else:
        formats["PLY"] = {"status": "UNAVAILABLE", "reason": "No declared point cloud."}

    coordinate_contract = _coordinate_contract(record, run_dir)
    formats["OBJ"], formats["GLB_GLTF"] = _export_mesh_formats(
        record, run_dir, coordinate_contract, include_inferred=include_inferred
    )
    formats["LAS"] = _export_las(
        record, run_dir, coordinate_contract, include_inferred=include_inferred
    )
    formats["GEOTIFF"] = {
        "status": "UNAVAILABLE",
        "reason": (
            "Out of scope: no defensible georeferenced DSM or orthomosaic raster product exists "
            "for this run; a point cloud renamed as GeoTIFF would be invalid."
        ),
    }
    formats["FBX"] = {
        "status": "UNAVAILABLE",
        "reason": "Out of scope: no FBX export and independent reopen validation is implemented.",
    }
    generated_files = [
        file
        for name in ("OBJ", "LAS", "GLB_GLTF")
        for file in formats[name].get("download_files", [])
    ]
    return {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "run_id": record.run_id,
        "exporter": {"name": EXPORTER_NAME, "version": EXPORTER_VERSION},
        "source_requirement": "SIH26158 desired-output table",
        "listed_formats_mandatory": "ORGANIZER_CLARIFICATION_REQUIRED",
        "coordinate_contract": coordinate_contract,
        "formats": formats,
        "available_validated_formats": [
            name for name, value in formats.items() if value["status"] == "AVAILABLE_VALIDATED"
        ],
        "generated_files": generated_files,
        "partial_success": any(
            value["status"] == "AVAILABLE_VALIDATED" for value in formats.values()
        )
        and any(value["status"] != "AVAILABLE_VALIDATED" for value in formats.values()),
        "measurement_eligibility_statement": (
            "Export never changes geometry provenance or measurement eligibility. The current "
            "measurement API remains bound to declared PLY evidence geometry."
        ),
        "geometry_selection": "OBSERVED_AND_COMPLETED" if include_inferred else "OBSERVED_ONLY",
        "completed_geometry_requires_explicit_selection": True,
        "completed_export_contract": (
            {
                "representation": "SEPARATE_OBSERVED_AND_INFERRED_PACKAGES",
                "reason": (
                    "Separate packages preserve provenance without relying on fragile face-index "
                    "ranges after independent format readers reorder geometry."
                ),
                "observed_geometry_provenance": "DERIVED_OBSERVED_VISUAL",
                "inferred_geometry_provenance": "INFERRED",
                "combined_completed_geometry_exported": False,
            }
            if include_inferred
            else None
        ),
    }


def export_artifact_paths(report: dict[str, Any], run_dir: Path) -> list[Path]:
    """Resolve generated report entries back to run-local files for declaration."""

    paths: list[Path] = []
    root = run_dir.resolve()
    for item in report.get("generated_files", []):
        relative = str(item.get("relative_path", ""))
        candidate = (run_dir / relative).resolve()
        if not relative or not candidate.is_relative_to(root) or not candidate.is_file():
            raise ExportUnavailable(f"Generated export path is invalid: {relative!r}")
        if sha256_file(candidate) != item.get("sha256"):
            raise ExportUnavailable(f"Generated export checksum changed: {relative}")
        paths.append(candidate)
    return paths


def write_export_readiness(path: Path, report: dict[str, Any]) -> None:
    atomic_json(path, report)
