import time
from pathlib import Path

from fastapi.testclient import TestClient

from sih26158.app import create_app
from sih26158.models import ProvenanceOrigin, RunConfig, RunStatus


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


def test_project_run_catalog_and_unsupported_matcher_are_truthful(
    tmp_path: Path, monkeypatch
) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        created = client.post(
            "/api/projects",
            data={"name": "catalog fixture"},
            files={
                "video": ("fixture.mp4", b"fixture", "video/mp4"),
                "telemetry": ("fixture.csv", b"timestamp_s,lat,lon,alt_m\n", "text/csv"),
            },
        ).json()
        monkeypatch.setattr(app.state.runner, "submit", lambda _: True)
        accepted = client.post(
            f"/api/projects/{created['project_id']}/runs",
            json={"execution_mode": "COLMAP", "matcher": "SIFT"},
        )
        rejected = client.post(
            f"/api/projects/{created['project_id']}/runs",
            json={"execution_mode": "COLMAP", "matcher": "SUPERPOINT_LIGHTGLUE"},
        )

        assert accepted.status_code == 202
        assert accepted.json()["requested_matcher"] == "SIFT"
        assert accepted.json()["executed_matcher"] is None
        assert rejected.status_code == 422
        assert "not an executable reconstruction provider" in rejected.json()["detail"]
        projects = client.get("/api/projects").json()["projects"]
        assert [item["project_id"] for item in projects] == [created["project_id"]]
        runs = client.get(f"/api/projects/{created['project_id']}/runs").json()["runs"]
        assert [item["run_id"] for item in runs] == [accepted.json()["run_id"]]


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


def test_readiness_measurement_and_linked_rerun_api(
    tmp_path: Path, monkeypatch
) -> None:
    app = create_app(tmp_path / "projects")
    with TestClient(app) as client:
        project_response = client.post(
            "/api/projects",
            data={
                "name": "evidence fixture",
                "video_origin": ProvenanceOrigin.REAL.value,
                "telemetry_origin": ProvenanceOrigin.REAL.value,
            },
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
        record = app.state.store.create_run(project_id, RunConfig())
        run_dir = app.state.store.run_dir(project_id, record.run_id)
        geometry = run_dir / "sparse" / "sparse_local.ply"
        geometry.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n0 0 0\n3 4 0\n",
            encoding="ascii",
        )
        poses = run_dir / "camera_poses.csv"
        poses.write_text("image_id,image_name,sfm_x,sfm_y,sfm_z\n", encoding="utf-8")
        keyframes = run_dir / "keyframes.json"
        keyframes.write_text('{"schema_version":"1.0","frames":[]}\n', encoding="utf-8")
        app.state.store.register_artifacts(record, [geometry, poses, keyframes])
        record.status = RunStatus.COMPLETED
        record.stage = RunStatus.COMPLETED
        app.state.store.save_run(record)
        record = app.state.store.get_run(record.run_id)
        geometry_artifact = next(
            item for item in record.artifacts if item.relative_path == "sparse/sparse_local.ply"
        )

        readiness = client.get(f"/api/runs/{record.run_id}/readiness")
        assert readiness.status_code == 200
        assert readiness.json()["sparse_preview_ready"] is True
        measurement = client.post(
            f"/api/runs/{record.run_id}/measurements",
            json={
                "geometry_artifact_path": geometry_artifact.relative_path,
                "geometry_artifact_sha256": geometry_artifact.sha256,
                "start": {"coordinates": [0, 0, 0], "point_id": 0},
                "end": {"coordinates": [3, 4, 0], "point_id": 1},
                "reference_value_m": 5.2,
                "reference_method": "tape",
                "reference_role": "HELD_OUT_EVALUATION",
            },
        )
        assert measurement.status_code == 201
        assert measurement.json()["backend_distance_m"] == 5
        listed = client.get(f"/api/runs/{record.run_id}/measurements")
        assert listed.json()["evaluation"]["all_held_out"]["sample_count"] == 1

        monkeypatch.setattr(app.state.runner, "submit", lambda _: True)
        rerun = client.post(
            f"/api/runs/{record.run_id}/rerun",
            json={"profile": "balanced", "execution_mode": "COLMAP"},
        )
        assert rerun.status_code == 202
        assert rerun.json()["run_id"] != record.run_id
        assert rerun.json()["derived_from_run_id"] == record.run_id
