import time
from pathlib import Path

from fastapi.testclient import TestClient

from sih26158.app import create_app
from sih26158.models import RunConfig
from sih26158.storage import atomic_json


def _create_project(client: TestClient) -> str:
    response = client.post(
        "/api/projects",
        data={
            "name": "viewer fixture",
            "video_origin": "SYNTHETIC",
            "telemetry_origin": "SYNTHETIC",
        },
        files={
            "video": ("fixture.mp4", b"not-real-video", "video/mp4"),
            "telemetry": (
                "fixture.csv",
                b"timestamp_s,lat,lon,alt_m\n0,1,2,3\n",
                "text/csv",
            ),
        },
    )
    assert response.status_code == 201
    return response.json()["project_id"]


def test_viewer_manifest_uses_declared_completed_artifacts(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_id = _create_project(client)
        response = client.post(
            f"/api/projects/{project_id}/runs",
            json={
                "execution_mode": "SYNTHETIC_DEMO",
                "profile": "smoke",
                "known_distance_m": 10,
                "measured_distance_m": 10.6,
            },
        )
        run_id = response.json()["run_id"]
        for _ in range(100):
            state = client.get(f"/api/runs/{run_id}").json()
            if state["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.01)

        response = client.get(f"/api/runs/{run_id}/viewer-manifest")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "1.0"
        assert payload["synthetic_fixture"] is True
        assert payload["source_provenance"] == "SYNTHETIC"
        assert payload["cloud"]["url"].endswith("/sparse/sparse_local.ply")
        assert payload["ingest_report_url"].endswith("/ingest_report.json")
        assert abs(payload["measurement_reference"]["percent_error"] - 6) < 1e-9
        assert payload["ai_overlay"]["measurement"] == "DISABLED"
        assert payload["cloud"]["color_mode_label"] == "Photographic RGB"
        assert payload["visual_models"]["evidence_cloud"]["available"] is True
        assert payload["visual_models"]["evidence_cloud"]["default"] is True
        assert payload["visual_models"]["evidence_cloud"]["measurement_eligible"] is False
        assert payload["visual_models"]["textured_mesh"]["available"] is False
        assert payload["visual_models"]["gaussian_splat"]["available"] is False
        assert payload["visual_models"]["gaussian_splat"]["measurement_eligible"] is False
        assert payload["confidence"]["available"] is False
        assert payload["confidence"]["reason"] == "Confidence unavailable for this run"
        assert {item["label"] for item in payload["confidence_legend"]} == {
            "OBSERVED_HIGH",
            "OBSERVED_MEDIUM",
            "OBSERVED_LOW",
            "AI_ASSISTED_NOT_MEASURABLE",
            "UNSEEN",
        }

        record = app.state.store.get_run(run_id)
        confidence_path = app.state.store.run_dir(project_id, run_id) / "point_confidence.json"
        atomic_json(
            confidence_path,
            {
                "schema_version": "1.0",
                "point_order": "PLY_VERTEX_ORDER",
                "points": [
                    {
                        "point_id": index,
                        "supporting_views": 3,
                        "track_length": 3,
                        "reprojection_error": 0.5,
                        "triangulation_angle": 8.0,
                        "confidence_class": "OBSERVED_MEDIUM",
                    }
                    for index in range(10)
                ],
            },
        )
        app.state.store.register_artifacts(record, [confidence_path])
        with_confidence = client.get(f"/api/runs/{run_id}/viewer-manifest").json()
        assert with_confidence["confidence"]["available"] is True
        assert with_confidence["confidence"]["url"].endswith("/point_confidence.json")

        dense_cloud = app.state.store.run_dir(project_id, run_id) / "dense/fused.ply"
        textured_mesh = app.state.store.run_dir(project_id, run_id) / "dense/textured/model.ply"
        texture = app.state.store.run_dir(project_id, run_id) / "dense/textured/atlas.png"
        dense_report = app.state.store.run_dir(project_id, run_id) / "dense_report.json"
        textured_mesh.parent.mkdir(parents=True, exist_ok=True)
        dense_cloud.write_text("declared dense cloud", encoding="utf-8")
        textured_mesh.write_text("declared textured mesh", encoding="utf-8")
        texture.write_bytes(b"declared texture")
        atomic_json(
            dense_report,
            {
                "texture_validation": {
                    "accepted": True,
                    "viewer_face_filter_contract": {
                        "strategy": "ATLAS_EMPTY_COLOR",
                        "empty_rgb": [255, 0, 255],
                        "empty_tolerance": 2,
                        "minimum_supported_samples": 1,
                    },
                }
            },
        )
        app.state.store.register_artifacts(
            record, [dense_cloud, textured_mesh, texture, dense_report]
        )
        with_dense = client.get(f"/api/runs/{run_id}/viewer-manifest").json()
        assert with_dense["visual_models"]["dense_cloud"]["available"] is True
        assert with_dense["visual_models"]["textured_mesh"]["available"] is True
        assert with_dense["visual_models"]["textured_mesh"]["measurement_eligible"] is False
        assert len(with_dense["visual_models"]["textured_mesh"]["texture_urls"]) == 1
        assert with_dense["visual_models"]["textured_mesh"]["texture_validity"] == {
            "strategy": "ATLAS_EMPTY_COLOR",
            "empty_rgb": [255, 0, 255],
            "empty_tolerance": 2,
            "minimum_supported_samples": 1,
        }


def test_viewer_manifest_is_not_fabricated_for_queued_run(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_id = _create_project(client)
        record = app.state.store.create_run(project_id, RunConfig())
        response = client.get(f"/api/runs/{record.run_id}/viewer-manifest")
        assert response.status_code == 409
        assert "missing declared artifacts" in response.json()["detail"]
