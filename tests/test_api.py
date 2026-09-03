import time
from pathlib import Path

from fastapi.testclient import TestClient

from sih26158.app import create_app
from sih26158.models import RunConfig, RunStatus


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


def test_failed_run_can_be_queued_for_checkpoint_resume(
    tmp_path: Path, monkeypatch
) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_response = client.post(
            "/api/projects",
            data={"name": "resume fixture"},
            files={
                "video": ("fixture.mp4", b"fixture", "video/mp4"),
                "telemetry": (
                    "fixture.csv",
                    b"timestamp_s,lat,lon,alt_m\n0,1,2,3\n",
                    "text/csv",
                ),
            },
        )
        project_id = project_response.json()["project_id"]
        record = app.state.store.create_run(
            project_id, RunConfig(execution_mode="SYNTHETIC_DEMO")
        )
        record.stage = RunStatus.FAILED
        record.status = RunStatus.FAILED
        record.failure_reason = "simulated interruption"
        app.state.store.save_run(record)
        submitted: list[str] = []

        def accept(run_id: str) -> bool:
            submitted.append(run_id)
            return True

        monkeypatch.setattr(app.state.runner, "submit", accept)
        response = client.post(f"/api/runs/{record.run_id}/resume")

        assert response.status_code == 202
        assert response.json()["status"] == "QUEUED"
        assert response.json()["failure_reason"] is None
        assert response.json()["recovery_count"] == 1
        assert submitted == [record.run_id]


def test_completed_run_resume_is_rejected(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_response = client.post(
            "/api/projects",
            data={"name": "completed fixture"},
            files={
                "video": ("fixture.mp4", b"fixture", "video/mp4"),
                "telemetry": (
                    "fixture.csv",
                    b"timestamp_s,lat,lon,alt_m\n0,1,2,3\n",
                    "text/csv",
                ),
            },
        )
        record = app.state.store.create_run(
            project_response.json()["project_id"],
            RunConfig(execution_mode="SYNTHETIC_DEMO"),
        )
        record.stage = RunStatus.COMPLETED
        record.status = RunStatus.COMPLETED
        app.state.store.save_run(record)

        response = client.post(f"/api/runs/{record.run_id}/resume")

        assert response.status_code == 409
        assert response.json()["detail"] == "Completed runs cannot be resumed"


def test_active_run_accepts_cross_process_cancellation_marker(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_response = client.post(
            "/api/projects",
            data={"name": "cancel fixture"},
            files={
                "video": ("fixture.mp4", b"fixture", "video/mp4"),
                "telemetry": (
                    "fixture.csv",
                    b"timestamp_s,lat,lon,alt_m\n0,1,2,3\n",
                    "text/csv",
                ),
            },
        )
        record = app.state.store.create_run(
            project_response.json()["project_id"],
            RunConfig(execution_mode="SYNTHETIC_DEMO"),
        )

        response = client.post(f"/api/runs/{record.run_id}/cancel")

        assert response.status_code == 202
        assert response.json()["cancel_requested_at"] is not None
        marker = app.state.store.run_dir(record.project_id, record.run_id) / ".cancel_requested"
        assert marker.is_file()


def test_completed_run_cancellation_is_rejected(tmp_path: Path) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_response = client.post(
            "/api/projects",
            data={"name": "completed cancel fixture"},
            files={
                "video": ("fixture.mp4", b"fixture", "video/mp4"),
                "telemetry": ("fixture.csv", b"timestamp_s,lat,lon,alt_m\n", "text/csv"),
            },
        )
        record = app.state.store.create_run(
            project_response.json()["project_id"],
            RunConfig(execution_mode="SYNTHETIC_DEMO"),
        )
        record.stage = RunStatus.COMPLETED
        record.status = RunStatus.COMPLETED
        app.state.store.save_run(record)

        response = client.post(f"/api/runs/{record.run_id}/cancel")

        assert response.status_code == 409
        assert response.json()["detail"] == "Run is not active"
