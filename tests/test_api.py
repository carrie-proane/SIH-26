import time
from pathlib import Path

from fastapi.testclient import TestClient

from sih26158.app import create_app


def test_upload_run_poll_and_artifact_index(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        response = client.post(
            "/api/projects",
            data={"name": "fixture"},
            files={
                "video": ("fixture.mp4", b"not-real-video", "video/mp4"),
                "telemetry": ("fixture.csv", b"timestamp_s,lat,lon,alt_m\n0,1,2,3\n", "text/csv"),
            },
        )
        assert response.status_code == 201
        project_id = response.json()["project_id"]
        response = client.post(
            f"/api/projects/{project_id}/runs",
            json={"execution_mode": "SYNTHETIC_DEMO", "profile": "smoke"},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        for _ in range(100):
            state = client.get(f"/api/runs/{run_id}").json()
            if state["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.01)
        assert state["status"] == "COMPLETED"
        index = client.get(f"/api/runs/{run_id}/artifact-index")
        assert index.status_code == 200
        paths = {item["relative_path"] for item in index.json()["artifacts"]}
        assert "quality_report.json" in paths
        assert all(item["url"].startswith(f"/api/runs/{run_id}/artifacts/") for item in index.json()["artifacts"])
        assert client.get(f"/api/runs/{run_id}/artifacts/quality_report.json").status_code == 200
        assert client.get(f"/api/runs/{run_id}/artifacts/run_manifest.json").status_code == 404
