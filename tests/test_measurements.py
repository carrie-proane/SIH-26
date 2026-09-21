from __future__ import annotations

from pathlib import Path

import pytest

from sih26158.models import MeasurementCreate, MeasurementEndpoint, ProvenanceOrigin, RunConfig
from sih26158.report import measurement_evaluation
from sih26158.storage import ProjectStore


def _run_with_geometry(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    video = tmp_path / "capture.mp4"
    telemetry = tmp_path / "capture.csv"
    video.write_bytes(b"real-fixture-placeholder")
    telemetry.write_text("timestamp_s,lat,lon,alt_m\n0,1,2,3\n", encoding="utf-8")
    project = store.create_project(
        name="measurement",
        description="",
        video_name=video.name,
        video=video,
        telemetry_name=telemetry.name,
        telemetry=telemetry,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
    )
    record = store.create_run(project.project_id, RunConfig())
    geometry = store.run_dir(project.project_id, record.run_id) / "sparse" / "sparse_local.ply"
    geometry.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n3 4 0\n",
        encoding="ascii",
    )
    store.register_artifacts(record, [geometry])
    return store, store.get_run(record.run_id)


def test_measurement_is_bound_to_declared_hash_and_distance_is_backend_computed(
    tmp_path: Path,
) -> None:
    store, record = _run_with_geometry(tmp_path)
    artifact = record.artifacts[0]
    created = store.add_measurement(
        record.run_id,
        MeasurementCreate(
            geometry_artifact_path=artifact.relative_path,
            geometry_artifact_sha256=artifact.sha256,
            start=MeasurementEndpoint(coordinates=(0, 0, 0), point_id=0),
            end=MeasurementEndpoint(coordinates=(3, 4, 0), point_id=1),
            measurement_kind="RELATIVE_DIMENSION",
            reference_value_m=5.2,
            reference_method="tape",
            reference_evidence="field-note-1",
            reference_role="HELD_OUT_EVALUATION",
        ),
    )

    assert created.backend_distance_m == 5
    assert created.geometry_provenance == "OBSERVED"
    assert created.measurement_eligible is True
    summary = measurement_evaluation(store.list_measurements(record.run_id))
    assert summary["all_held_out"]["sample_count"] == 1
    assert summary["all_held_out"]["median_absolute_error_m"] == pytest.approx(0.2)
    assert summary["official_requirement"]["compliance_status"] == "PROTOCOL_UNDEFINED"


def test_measurement_rejects_wrong_hash_and_out_of_range_point(tmp_path: Path) -> None:
    store, record = _run_with_geometry(tmp_path)
    artifact = record.artifacts[0]
    base = {
        "geometry_artifact_path": artifact.relative_path,
        "start": MeasurementEndpoint(coordinates=(0, 0, 0), point_id=0),
        "end": MeasurementEndpoint(coordinates=(1, 0, 0), point_id=1),
    }
    with pytest.raises(ValueError, match="hash"):
        store.add_measurement(
            record.run_id,
            MeasurementCreate(geometry_artifact_sha256="0" * 64, **base),
        )
    with pytest.raises(ValueError, match="outside"):
        store.add_measurement(
            record.run_id,
            MeasurementCreate(
                geometry_artifact_sha256=artifact.sha256,
                **(base | {"end": MeasurementEndpoint(coordinates=(1, 0, 0), point_id=2)}),
            ),
        )


def test_scale_control_is_excluded_from_independent_evaluation(tmp_path: Path) -> None:
    store, record = _run_with_geometry(tmp_path)
    artifact = record.artifacts[0]
    store.add_measurement(
        record.run_id,
        MeasurementCreate(
            geometry_artifact_path=artifact.relative_path,
            geometry_artifact_sha256=artifact.sha256,
            start=MeasurementEndpoint(coordinates=(0, 0, 0)),
            end=MeasurementEndpoint(coordinates=(3, 4, 0)),
            reference_value_m=5,
            reference_role="SCALE_CONTROL",
        ),
    )
    summary = measurement_evaluation(store.list_measurements(record.run_id))["all_held_out"]
    assert summary["status"] == "UNAVAILABLE"
    assert summary["excluded_scale_control_count"] == 1
