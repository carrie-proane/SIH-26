import json
from pathlib import Path

import pytest

from sih26158.models import RunConfig, RunStatus
from sih26158.pipeline import PipelineRunner
from sih26158.storage import ProjectStore, sha256_file


def make_project(store: ProjectStore, tmp_path: Path):
    video = tmp_path / "sample.mp4"
    telemetry = tmp_path / "sample.csv"
    video.write_bytes(b"fixture")
    telemetry.write_text("timestamp_s,lat,lon,alt_m\n0,1,2,3\n", encoding="utf-8")
    return store.create_project(
        name="test",
        description="",
        video_name=video.name,
        video=video,
        telemetry_name=telemetry.name,
        telemetry=telemetry,
    )


def test_project_inputs_are_hashed_and_immutable(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    for asset in project.assets:
        path = store.project_dir(project.project_id) / asset.relative_path
        assert sha256_file(path) == asset.sha256
    assert project.immutable is True


def test_synthetic_pipeline_exercises_exact_states_and_declares_artifacts(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    record = store.create_run(
        project.project_id,
        RunConfig(execution_mode="SYNTHETIC_DEMO", known_distance_m=10, measured_distance_m=10.5),
    )
    result = PipelineRunner(store).run(record.run_id)
    assert result.status == RunStatus.COMPLETED
    assert [event.stage for event in result.events] == [
        RunStatus.INGESTING,
        RunStatus.PREPROCESSING,
        RunStatus.RECONSTRUCTING,
        RunStatus.REPORTING,
        RunStatus.COMPLETED,
    ]
    declared = {artifact.relative_path for artifact in result.artifacts}
    assert "quality_report.json" in declared
    quality = json.loads((store.run_dir(project.project_id, result.run_id) / "quality_report.json").read_text())
    assert quality["synthetic_fixture"] is True
    assert quality["metrics"]["known_distance"]["passes_10_percent_gate"] is True
    assert quality["metrics"]["metric_alignment"]["scale"] == 1.0
    assert any(warning["code"] == "SYNTHETIC_TELEMETRY" for warning in quality["warnings"])
    ingest = json.loads(
        (store.run_dir(project.project_id, result.run_id) / "ingest_report.json").read_text()
    )
    assert ingest["telemetry_normalization"]["schema_version"] == "1.0"


def test_artifact_serving_rejects_undeclared_and_traversal(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    record = store.create_run(project.project_id, RunConfig(execution_mode="SYNTHETIC_DEMO"))
    result = PipelineRunner(store).run(record.run_id)
    assert store.resolve_declared_artifact(result.run_id, "quality_report.json").is_file()
    with pytest.raises(FileNotFoundError):
        store.resolve_declared_artifact(result.run_id, "run_manifest.json")
    with pytest.raises(FileNotFoundError):
        store.resolve_declared_artifact(result.run_id, "../manifest.json")


def test_real_run_retains_ingest_artifact_when_preprocessing_is_missing(tmp_path: Path, monkeypatch) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    record = store.create_run(project.project_id, RunConfig())
    monkeypatch.setattr("sih26158.pipeline.shutil.which", lambda name: "/usr/bin/ffprobe")

    class Completed:
        returncode = 0
        stdout = '{"format": {}, "streams": []}'

    monkeypatch.setattr("sih26158.pipeline.subprocess.run", lambda *args, **kwargs: Completed())
    result = PipelineRunner(store).run(record.run_id)
    assert result.status == RunStatus.FAILED
    assert "preprocessing_run" in (result.failure_reason or "")
    assert "ingest_report.json" in {item.relative_path for item in result.artifacts}
