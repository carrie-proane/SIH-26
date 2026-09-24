from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from sih26158.app import create_app
from sih26158.exports import (
    GLTF_TO_ENU,
    build_export_readiness,
    export_artifact_paths,
    write_export_readiness,
)
from sih26158.models import ProvenanceOrigin, RunConfig, RunStatus
from sih26158.storage import ProjectStore, atomic_json, sha256_file

POINTS_PLY = """ply
format ascii 1.0
element vertex 4
property double x
property double y
property double z
property uchar red
property uchar green
property uchar blue
property uchar classification
property float custom_quality
end_header
0 0 0 10 20 30 2 0.9
2 0 0 40 50 60 3 0.8
2 3 1 70 80 90 4 0.7
0 3 1 100 110 120 5 0.6
"""

TEXTURED_MESH_PLY = """ply
format ascii 1.0
comment TextureFile atlas.png
element vertex 4
property double x
property double y
property double z
property uchar classification
property float source_confidence
element face 2
property list uchar int vertex_indices
property list uchar float texcoord
property int texnumber
end_header
0 0 0 7 0.9
2 0 0 7 0.8
2 3 1 8 0.7
0 3 1 8 0.6
3 0 1 2 6 0 0 1 0 1 1 0
3 0 2 3 6 0 0 1 1 0 1 0
"""

UNTEXTURED_MESH_PLY = """ply
format ascii 1.0
element vertex 4
property double x
property double y
property double z
property uchar red
property uchar green
property uchar blue
element face 2
property list uchar int vertex_indices
end_header
0 0 0 200 10 20
2 0 0 200 10 20
2 3 1 200 10 20
0 3 1 200 10 20
3 0 1 2
3 0 2 3
"""


def _record_with_geometry(
    tmp_path: Path,
    *,
    mesh: str | None = TEXTURED_MESH_PLY,
    mesh_relative: str = "dense/textured/model.ply",
    points: str = POINTS_PLY,
):
    store = ProjectStore(tmp_path / "projects")
    video = tmp_path / "capture.mp4"
    telemetry = tmp_path / "capture.csv"
    video.write_bytes(b"real-video-placeholder")
    telemetry.write_text("timestamp_s,lat,lon,alt_m\n0,18.5,73.8,10\n", encoding="utf-8")
    project = store.create_project(
        name="export fixture",
        description="",
        video_name=video.name,
        video=video,
        telemetry_name=telemetry.name,
        telemetry=telemetry,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
    )
    record = store.create_run(project.project_id, RunConfig())
    run_dir = store.run_dir(project.project_id, record.run_id)
    sparse = run_dir / "sparse" / "sparse_local.ply"
    sparse.write_text(points, encoding="ascii")
    transform = run_dir / "local_transform.json"
    atomic_json(
        transform,
        {
            "coordinate_frame": "LOCAL_ENU_METRES",
            "origin_wgs84": {"lat": 18.5, "lon": 73.8, "alt_m": 10.0},
            "altitude_reference": "relative_to_launch",
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation_m": [0, 0, 0],
        },
    )
    confidence = run_dir / "point_confidence.json"
    atomic_json(
        confidence,
        {
            "schema_version": "1.0",
            "point_order": "PLY_VERTEX_ORDER",
            "points": [
                {
                    "point_id": index,
                    "supporting_views": 4,
                    "track_length": 4,
                    "reprojection_error": 0.5,
                    "triangulation_angle": 8.0,
                    "confidence_class": "OBSERVED_HIGH",
                }
                for index in range(4)
            ],
        },
    )
    paths = [sparse, transform, confidence]
    if mesh is not None:
        mesh_path = run_dir / mesh_relative
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_text(mesh, encoding="ascii")
        paths.append(mesh_path)
        if "TextureFile atlas.png" in mesh:
            texture = mesh_path.parent / "atlas.png"
            image = Image.new("RGB", (4, 4), (12, 34, 56))
            image.putpixel((0, 0), (200, 100, 50))
            image.save(texture)
            paths.append(texture)
    store.register_artifacts(record, paths)
    return store, store.get_run(record.run_id), run_dir


def _file(export: dict[str, object], suffix: str) -> Path:
    files = export["files"]
    assert isinstance(files, list)
    relative = next(
        item["relative_path"]
        for item in files
        if isinstance(item, dict) and str(item["relative_path"]).endswith(suffix)
    )
    return Path(str(relative))


def _read_las_independently(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw[:4] == b"LASF"
    assert tuple(raw[24:26]) == (1, 2)
    point_offset = struct.unpack_from("<I", raw, 96)[0]
    point_format = raw[104] & 0x3F
    record_length = struct.unpack_from("<H", raw, 105)[0]
    count = struct.unpack_from("<I", raw, 107)[0]
    scale = np.array(struct.unpack_from("<3d", raw, 131))
    offset = np.array(struct.unpack_from("<3d", raw, 155))
    vlr_length = struct.unpack_from("<H", raw, 227 + 20)[0]
    vlr_payload = json.loads(raw[227 + 54 : 227 + 54 + vlr_length].decode("utf-8"))
    points = []
    classifications = []
    colors = []
    for index in range(count):
        start = point_offset + index * record_length
        values = struct.unpack_from("<iiiHBBbBH3H", raw, start)
        points.append(np.array(values[:3], dtype=float) * scale + offset)
        classifications.append(values[5])
        colors.append(values[-3:])
    return {
        "point_format": point_format,
        "points": np.asarray(points),
        "classifications": classifications,
        "colors": colors,
        "vlr": vlr_payload,
    }


def test_textured_obj_glb_and_las_reopen_with_coordinates_and_provenance(
    tmp_path: Path,
) -> None:
    store, record, run_dir = _record_with_geometry(tmp_path)
    source_hashes = {
        item.relative_path: sha256_file(run_dir / item.relative_path)
        for item in record.artifacts
    }

    report = build_export_readiness(record, run_dir)

    assert report["available_validated_formats"] == ["PLY", "OBJ", "GLB_GLTF", "LAS"]
    obj_export = report["formats"]["OBJ"]["exports"][0]
    glb_export = report["formats"]["GLB_GLTF"]["exports"][0]
    las_export = next(
        item
        for item in report["formats"]["LAS"]["exports"]
        if item["variant"] == "observed-sparse"
    )
    assert obj_export["geometry_provenance"] == "DERIVED_OBSERVED_VISUAL"
    assert obj_export["measurement_eligible"] is False
    assert obj_export["options"]["unsupported_source_attributes"]["vertex"] == [
        "classification",
        "source_confidence",
    ]
    assert glb_export["measurement_eligible"] is False
    assert las_export["geometry_provenance"] == "OBSERVED"
    assert las_export["measurement_eligible"] is True

    obj_path = run_dir / _file(obj_export, "model.obj")
    obj = trimesh.load(obj_path, force="mesh", process=False, maintain_order=True)
    assert len(obj.vertices) == 4
    assert len(obj.faces) == 2
    assert np.allclose(obj.bounds, [[0, 0, 0], [2, 3, 1]], atol=1e-9)
    assert getattr(obj.visual.material, "image", None) is not None
    assert "map_Kd textures/atlas.png" in (obj_path.parent / "material.mtl").read_text()

    glb_path = run_dir / _file(glb_export, "model.glb")
    scene = trimesh.load(glb_path, force="scene", process=False)
    glb_mesh = next(iter(scene.geometry.values()))
    recovered = glb_mesh.copy()
    recovered.apply_transform(GLTF_TO_ENU)
    assert len(recovered.vertices) == 4
    assert len(recovered.faces) == 2
    assert np.allclose(recovered.bounds, [[0, 0, 0], [2, 3, 1]], atol=1e-6)
    assert np.allclose(np.ptp(recovered.vertices, axis=0), [2, 3, 1], atol=1e-6)
    assert getattr(glb_mesh.visual, "kind", None) == "texture"

    las_path = run_dir / _file(las_export, "cloud.las")
    las = _read_las_independently(las_path)
    assert las["point_format"] == 2
    assert np.allclose(np.asarray(las["points"]), [[0, 0, 0], [2, 0, 0], [2, 3, 1], [0, 3, 1]], atol=5e-5)
    assert las["classifications"] == [2, 3, 4, 5]
    assert las["colors"][0] == (10 * 257, 20 * 257, 30 * 257)
    assert las["vlr"]["epsg"] is None
    assert las["vlr"]["geometry_provenance"] == "OBSERVED"
    provenance_path = las_path.parent / "provenance.json"
    assert las["vlr"]["provenance_sidecar_sha256"] == sha256_file(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    assert provenance["coordinate_contract"]["origin_wgs84"]["lat"] == 18.5
    assert provenance["coordinate_contract"]["epsg"] is None
    assert provenance["coordinate_contract"]["local_transform_metadata"]["scale"] == 1.0
    assert provenance["unsupported_source_vertex_attributes"] == ["custom_quality"]
    assert provenance["point_confidence_artifact"]["sha256"] == source_hashes["point_confidence.json"]

    for relative, digest in source_hashes.items():
        assert sha256_file(run_dir / relative) == digest

    generated = export_artifact_paths(report, run_dir)
    write_export_readiness(run_dir / "export_readiness.json", report)
    store.register_artifacts(record, [run_dir / "export_readiness.json", *generated])
    persisted = store.get_run(record.run_id)
    for path in generated:
        relative = str(path.relative_to(run_dir))
        assert store.resolve_declared_artifact(persisted.run_id, relative) == path.resolve()


def test_export_packages_are_reused_for_matching_source_hash_and_options(tmp_path: Path) -> None:
    _, record, run_dir = _record_with_geometry(tmp_path)

    first = build_export_readiness(record, run_dir)
    second = build_export_readiness(record, run_dir)

    for name in ("OBJ", "GLB_GLTF", "LAS"):
        assert all(item["reused"] is False for item in first["formats"][name]["exports"])
        assert all(item["reused"] is True for item in second["formats"][name]["exports"])
        assert first["formats"][name]["download_files"] == second["formats"][name]["download_files"]


def test_export_cache_invalidates_when_coordinate_metadata_changes(tmp_path: Path) -> None:
    store, record, run_dir = _record_with_geometry(tmp_path)
    first = build_export_readiness(record, run_dir)
    first_manifest = first["formats"]["GLB_GLTF"]["exports"][0]["manifest_url"]
    transform = run_dir / "local_transform.json"
    payload = json.loads(transform.read_text(encoding="utf-8"))
    payload["origin_wgs84"]["lat"] = 19.0
    atomic_json(transform, payload)
    store.register_artifacts(record, [transform])

    second = build_export_readiness(store.get_run(record.run_id), run_dir)

    glb = second["formats"]["GLB_GLTF"]["exports"][0]
    assert glb["reused"] is False
    assert glb["manifest_url"] != first_manifest
    assert glb["coordinate_contract"]["origin_wgs84"]["lat"] == 19.0


def test_untextured_mesh_is_exported_explicitly_without_fabricated_material(
    tmp_path: Path,
) -> None:
    _, record, run_dir = _record_with_geometry(tmp_path, mesh=UNTEXTURED_MESH_PLY)

    report = build_export_readiness(record, run_dir)

    obj = report["formats"]["OBJ"]["exports"][0]
    glb = report["formats"]["GLB_GLTF"]["exports"][0]
    assert obj["options"]["photographic_texture_required"] is False
    assert obj["validation"]["texture_preserved"] is False
    assert glb["validation"]["texture_preserved"] is False


def test_point_only_run_exports_las_but_not_a_fabricated_mesh(tmp_path: Path) -> None:
    _, record, run_dir = _record_with_geometry(tmp_path, mesh=None)

    report = build_export_readiness(record, run_dir)

    assert report["formats"]["LAS"]["status"] == "AVAILABLE_VALIDATED"
    assert report["formats"]["OBJ"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert report["formats"]["GLB_GLTF"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert "will not triangulate sparse points" in report["formats"]["OBJ"]["reason"]


def test_undeclared_geometry_is_not_discovered_from_the_run_directory(tmp_path: Path) -> None:
    _, record, run_dir = _record_with_geometry(tmp_path)
    record.artifacts = [
        artifact for artifact in record.artifacts if not artifact.relative_path.endswith(".ply")
    ]

    report = build_export_readiness(record, run_dir)

    assert report["formats"]["PLY"]["status"] == "UNAVAILABLE"
    assert report["formats"]["LAS"]["status"] == "UNAVAILABLE"
    assert report["formats"]["OBJ"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert report["generated_files"] == []


def test_inferred_mesh_remains_inferred_and_measurement_disabled(tmp_path: Path) -> None:
    _, record, run_dir = _record_with_geometry(
        tmp_path,
        mesh=UNTEXTURED_MESH_PLY,
        mesh_relative="completion/fixture/inferred_geometry.ply",
    )

    observed_only = build_export_readiness(record, run_dir)
    assert observed_only["geometry_selection"] == "OBSERVED_ONLY"
    assert observed_only["formats"]["OBJ"]["exports"] == []

    report = build_export_readiness(record, run_dir, include_inferred=True)
    assert report["geometry_selection"] == "OBSERVED_AND_COMPLETED"
    assert report["completed_export_contract"]["representation"] == (
        "SEPARATE_OBSERVED_AND_INFERRED_PACKAGES"
    )
    assert report["completed_export_contract"]["combined_completed_geometry_exported"] is False

    for name in ("OBJ", "GLB_GLTF"):
        exported = report["formats"][name]["exports"][0]
        assert exported["geometry_provenance"] == "INFERRED"
        assert exported["measurement_eligible"] is False
        assert exported["measurement_api_supported"] is False


def test_completed_selection_reopens_separate_observed_and_inferred_meshes(
    tmp_path: Path,
) -> None:
    store, record, run_dir = _record_with_geometry(tmp_path, mesh=UNTEXTURED_MESH_PLY)
    inferred = run_dir / "completion" / "fixture" / "inferred_geometry.ply"
    inferred.parent.mkdir(parents=True)
    inferred.write_text(UNTEXTURED_MESH_PLY, encoding="ascii")
    status = run_dir / "completion_status.json"
    atomic_json(
        status,
        {
            "status": "completed",
            "method": "BOUNDED_PLANAR_GAP",
            "source_geometry_sha256": next(
                item.sha256
                for item in record.artifacts
                if item.relative_path == "dense/textured/model.ply"
            ),
            "inferred_artifact_url": (
                f"/api/runs/{record.run_id}/artifacts/completion/fixture/inferred_geometry.ply"
            ),
            "inferred_face_count": 2,
            "eligible_for_measurement": False,
        },
    )
    store.register_artifacts(record, [inferred, status])
    record = store.get_run(record.run_id)

    report = build_export_readiness(record, run_dir, include_inferred=True)
    for name, suffix in (("OBJ", "model.obj"), ("GLB_GLTF", "model.glb")):
        exports = report["formats"][name]["exports"]
        assert {item["geometry_provenance"] for item in exports} == {
            "DERIVED_OBSERVED_VISUAL",
            "INFERRED",
        }
        for item in exports:
            reopened = trimesh.load(run_dir / _file(item, suffix), process=False)
            if isinstance(reopened, trimesh.Scene):
                assert len(reopened.geometry) == 1
                reopened = next(iter(reopened.geometry.values()))
            assert len(reopened.vertices) == 4
            assert len(reopened.faces) == 2
            assert item["measurement_eligible"] is False


def test_non_finite_geometry_is_rejected_without_publishing_exports(tmp_path: Path) -> None:
    invalid = POINTS_PLY.replace("2 3 1 70 80 90 4 0.7", "nan 3 1 70 80 90 4 0.7")
    _, record, run_dir = _record_with_geometry(tmp_path, mesh=None, points=invalid)
    source = run_dir / "sparse/sparse_local.ply"
    original_hash = sha256_file(source)

    report = build_export_readiness(record, run_dir)

    assert report["formats"]["PLY"]["status"] == "INVALID"
    assert report["formats"]["LAS"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert "non-finite" in report["formats"]["LAS"]["reason"]
    assert report["generated_files"] == []
    assert sha256_file(source) == original_hash


def test_mesh_export_scene_limit_fails_before_publication(tmp_path: Path, monkeypatch) -> None:
    from sih26158 import exports as export_module

    _, record, run_dir = _record_with_geometry(tmp_path, mesh=UNTEXTURED_MESH_PLY)
    monkeypatch.setattr(export_module, "MAX_EXPORT_VERTICES", 3)
    report = build_export_readiness(record, run_dir)
    assert report["formats"]["OBJ"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert "vertex/face export limits" in report["formats"]["OBJ"]["reason"]
    assert not (run_dir / "exports" / "obj").exists()


@pytest.mark.parametrize(
    ("points", "reason_fragment"),
    [
        (
            "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n"
            + "property float z\nend_header\n",
            "no vertices",
        ),
        ("this is not a ply file\n", "could not be parsed"),
    ],
)
def test_empty_and_malformed_geometry_are_actionable_and_not_published(
    tmp_path: Path,
    points: str,
    reason_fragment: str,
) -> None:
    _, record, run_dir = _record_with_geometry(tmp_path, mesh=None, points=points)

    report = build_export_readiness(record, run_dir)

    assert report["formats"]["PLY"]["status"] == "INVALID"
    assert report["formats"]["LAS"]["status"] == "INVALID_OR_UNSUPPORTED"
    assert reason_fragment.lower() in report["formats"]["LAS"]["reason"].lower()
    assert report["generated_files"] == []


def test_exports_endpoint_builds_without_reconstruction_and_serves_declared_urls(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        video = tmp_path / "api.mp4"
        telemetry = tmp_path / "api.csv"
        video.write_bytes(b"fixture")
        telemetry.write_text("timestamp_s,lat,lon,alt_m\n0,1,2,3\n", encoding="utf-8")
        project = app.state.store.create_project(
            name="api export",
            description="",
            video_name=video.name,
            video=video,
            telemetry_name=telemetry.name,
            telemetry=telemetry,
            video_origin=ProvenanceOrigin.REAL,
            telemetry_origin=ProvenanceOrigin.REAL,
        )
        record = app.state.store.create_run(project.project_id, RunConfig())
        run_dir = app.state.store.run_dir(project.project_id, record.run_id)
        sparse = run_dir / "sparse/sparse_local.ply"
        sparse.write_text(POINTS_PLY, encoding="ascii")
        app.state.store.register_artifacts(record, [sparse])
        record = app.state.store.get_run(record.run_id)
        record.stage = RunStatus.COMPLETED
        record.status = RunStatus.COMPLETED
        app.state.store.save_run(record)

        response = client.post(f"/api/runs/{record.run_id}/exports")
        assert response.status_code == 201
        response = client.get(f"/api/runs/{record.run_id}/exports")
        assert response.status_code == 200
        las = response.json()["formats"]["LAS"]["exports"][0]
        las_url = next(item["url"] for item in las["files"] if item["url"].endswith("cloud.las"))
        assert client.get(las_url).status_code == 200
        readiness = client.get(f"/api/runs/{record.run_id}/readiness").json()
        assert readiness["export_report_ready"] is True
        assert readiness["export_report_url"] == f"/api/runs/{record.run_id}/exports"
        assert readiness["export_status"]["LAS"] == "AVAILABLE_VALIDATED"


def test_obj_bundle_contains_complete_relative_package(tmp_path: Path) -> None:
    store, record, run_dir = _record_with_geometry(tmp_path)
    record.stage = RunStatus.COMPLETED
    record.status = RunStatus.COMPLETED
    store.save_run(record)
    report = build_export_readiness(store.get_run(record.run_id), run_dir)
    report_path = run_dir / "export_readiness.json"
    write_export_readiness(report_path, report)
    store.register_artifacts(record, [report_path, *export_artifact_paths(report, run_dir)])
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        payload = client.get(f"/api/runs/{record.run_id}/exports").json()
        obj = payload["formats"]["OBJ"]["exports"][0]
        response = client.get(obj["bundle_url"])

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        archive_path = tmp_path / "bundle.zip"
        archive_path.write_bytes(response.content)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            assert any(name.endswith(".obj") for name in names)
            assert any(name.endswith(".mtl") for name in names)
            assert any(name.endswith(".png") for name in names)


def test_exports_endpoint_is_not_fabricated_before_report_exists(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        video = tmp_path / "missing.mp4"
        telemetry = tmp_path / "missing.csv"
        video.write_bytes(b"fixture")
        telemetry.write_text("timestamp_s,lat,lon,alt_m\n", encoding="utf-8")
        project = app.state.store.create_project(
            name="missing export",
            description="",
            video_name=video.name,
            video=video,
            telemetry_name=telemetry.name,
            telemetry=telemetry,
        )
        record = app.state.store.create_run(project.project_id, RunConfig())

        response = client.get(f"/api/runs/{record.run_id}/exports")

        assert response.status_code == 409
        assert response.json()["detail"] == "Export report is not ready"

        create = client.post(f"/api/runs/{record.run_id}/exports")
        assert create.status_code == 409
        assert create.json()["detail"] == (
            "Geometry export requires a completed, failed, or cancelled run"
        )

        unknown = client.get("/api/runs/run_does_not_exist/exports")
        assert unknown.status_code == 404
        assert unknown.json()["detail"] == "Run not found"
