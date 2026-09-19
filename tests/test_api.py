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

        record = app.state.store.get_run(run_id)
        frame = app.state.store.run_dir(project_id, run_id) / "frames" / "selected.jpg"
        frame.write_bytes(b"declared-source-frame")
        app.state.store.register_artifacts(record, [frame])
        assert client.get(f"/api/runs/{run_id}/artifacts/frames/selected.jpg").status_code == 200
        assert client.get(f"/api/runs/{run_id}/artifacts/frames/undeclared.jpg").status_code == 404
        assert client.get(f"/api/runs/{run_id}/artifacts/../run_manifest.json").status_code == 404


def test_startup_recovers_interrupted_runs_and_preserves_artifacts(tmp_path: Path) -> None:
    import io

    import pytest

    from sih26158.models import RunConfig, RunStatus
    from sih26158.storage import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    project = store.create_project(name="recovery", description="", video_name="v.mp4",
        video=io.BytesIO(b"video"), telemetry_name="t.csv", telemetry=io.BytesIO(b"telemetry"))
    interrupted = []
    for status in (RunStatus.QUEUED, RunStatus.INGESTING, RunStatus.PREPROCESSING,
                   RunStatus.RECONSTRUCTING, RunStatus.REPORTING):
        record = store.create_run(project.project_id, RunConfig())
        record.status = record.stage = status
        artifact = store.run_dir(project.project_id, record.run_id) / "partial.txt"
        artifact.write_text("retained evidence")
        store.register_artifacts(record, [artifact])
        interrupted.append((record, artifact))
    completed = store.create_run(project.project_id, RunConfig())
    completed.status = completed.stage = RunStatus.COMPLETED
    store.save_run(completed)
    before = (store.run_dir(project.project_id, completed.run_id) / "run_manifest.json").read_bytes()
    with TestClient(create_app(store.root)):
        for record, artifact in interrupted:
            recovered = store.get_run(record.run_id)
            assert recovered.status == RunStatus.FAILED
            assert recovered.failure_reason == "interrupted_by_restart"
            assert recovered.artifacts == record.artifacts
            assert artifact.read_text() == "retained evidence"
        assert (store.run_dir(project.project_id, completed.run_id) / "run_manifest.json").read_bytes() == before
        with pytest.raises(RuntimeError, match="single worker process"), TestClient(create_app(store.root)):
            pass
    # Lock is released on shutdown; terminal records remain untouched on next startup.
    with TestClient(create_app(store.root)):
        assert store.get_run(completed.run_id).status == RunStatus.COMPLETED
