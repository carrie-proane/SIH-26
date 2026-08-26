import csv
import json
import math
from pathlib import Path

import pytest

from sih26158.models import ProvenanceOrigin, RunConfig, RunStatus
from sih26158.pipeline import PipelineRunner
from sih26158.storage import ProjectStore, sha256_file


def make_project(
    store: ProjectStore,
    tmp_path: Path,
    *,
    video_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN,
    telemetry_origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN,
):
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
        video_origin=video_origin,
        telemetry_origin=telemetry_origin,
    )


def test_project_inputs_are_hashed_and_immutable(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(
        store,
        tmp_path,
        video_origin=ProvenanceOrigin.SYNTHETIC,
        telemetry_origin=ProvenanceOrigin.SYNTHETIC,
    )
    for asset in project.assets:
        path = store.project_dir(project.project_id) / asset.relative_path
        assert sha256_file(path) == asset.sha256
    assert project.immutable is True
    assert project.source_provenance == ProvenanceOrigin.SYNTHETIC
    assert {asset.origin for asset in project.assets} == {ProvenanceOrigin.SYNTHETIC}


def test_synthetic_pipeline_exercises_exact_states_and_declares_artifacts(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(
        store,
        tmp_path,
        video_origin=ProvenanceOrigin.SYNTHETIC,
        telemetry_origin=ProvenanceOrigin.SYNTHETIC,
    )
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
    assert "sync_report.json" in declared
    assert quality["source_provenance"] == "SYNTHETIC"
    assert quality["confidence_artifact"]["available"] is False


@pytest.mark.parametrize("execution_mode", ["COLMAP", "SYNTHETIC_DEMO"])
def test_synthetic_telemetry_remains_synthetic_in_every_execution_mode(
    tmp_path: Path, execution_mode: str
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(
        store,
        tmp_path,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.SYNTHETIC,
    )
    record = store.create_run(project.project_id, RunConfig(execution_mode=execution_mode))
    assert record.telemetry_origin == ProvenanceOrigin.SYNTHETIC
    assert record.source_provenance == ProvenanceOrigin.SYNTHETIC
    assert record.synthetic_fixture is True


def test_synthetic_telemetry_metadata_downgrades_a_colmap_run(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(
        store,
        tmp_path,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.UNKNOWN,
    )
    record = store.create_run(project.project_id, RunConfig(execution_mode="COLMAP"))
    run_dir = store.run_dir(project.project_id, record.run_id)
    (run_dir / "ingest_report.json").write_text("{}", encoding="utf-8")
    (run_dir / "normalized_telemetry.meta.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "warnings": [
                    {"code": "SYNTHETIC_TELEMETRY", "detail": "Generated fixture"}
                ],
            }
        ),
        encoding="utf-8",
    )

    PipelineRunner(store)._merge_telemetry_metadata(record, [])

    persisted = store.get_run(record.run_id)
    assert persisted.telemetry_origin == ProvenanceOrigin.SYNTHETIC
    assert persisted.source_provenance == ProvenanceOrigin.SYNTHETIC
    assert persisted.synthetic_fixture is True


def test_calibrated_telemetry_offset_is_persisted_per_run(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(
        store,
        tmp_path,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
    )
    record = store.create_run(
        project.project_id,
        RunConfig(telemetry_offset_s=0.30, telemetry_offset_source="calibrated"),
    )
    run_dir = store.run_dir(project.project_id, record.run_id)
    frame_times = [0.5, 1.5, 2.5, 3.5, 4.5]
    keyframes = []
    with (run_dir / "camera_poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "image_name", "sfm_x", "sfm_y", "sfm_z"])
        for index, timestamp in enumerate(frame_times, start=1):
            telemetry_time = timestamp + 0.30
            east = telemetry_time
            north = 0.25 * telemetry_time**2
            up = 0.2 * math.sin(telemetry_time)
            name = f"frame_{index:04d}.jpg"
            writer.writerow([index, name, east, north, up])
            keyframes.append(
                {"image_name": name, "timestamp_s": timestamp, "selected": True}
            )
    (run_dir / "keyframes.json").write_text(
        json.dumps({"frames": keyframes}), encoding="utf-8"
    )
    with (run_dir / "normalized_telemetry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_s", "lat", "lon", "alt_m", "alt_source", "fix_quality", "source_row"]
        )
        lat0, lon0, alt0 = 18.5, 73.8, 10.0
        for row_index in range(61):
            timestamp = row_index / 10
            east = timestamp
            north = 0.25 * timestamp**2
            up = 0.2 * math.sin(timestamp)
            lat = lat0 + north / 111_320
            lon = lon0 + east / (111_320 * math.cos(math.radians(lat0)))
            writer.writerow([timestamp, lat, lon, alt0 + up, "relative", "ok", row_index])
    (run_dir / "sparse" / "sparse.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )

    artifacts = PipelineRunner(store)._align_to_local_metric(record)

    assert run_dir / "sync_report.json" in artifacts
    persisted = store.get_run(record.run_id)
    assert persisted.telemetry_offset_s == pytest.approx(0.30)
    assert persisted.offset_source == "calibrated"
    assert persisted.rmse_before_m is not None
    assert persisted.rmse_after_m is not None
    sync = json.loads((run_dir / "sync_report.json").read_text(encoding="utf-8"))
    assert sync["telemetry_offset_s"] == pytest.approx(0.30)
    assert sync["offset_source"] == "calibrated"


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
