import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sih26158.ai_integration import (
    CompletionRequest,
    run_completion,
    segmentation_fingerprint,
    write_coverage_report,
)
from sih26158.app import create_app
from sih26158.colmap import ColmapRunner, ExternalToolError
from sih26158.completion import boundary_id
from sih26158.completion_contract import GapReview
from sih26158.dataset import DatasetAsset
from sih26158.models import MeasurementCreate, MeasurementEndpoint, ProvenanceOrigin, RunConfig
from sih26158.pipeline import PipelineRunner
from sih26158.process_control import ProcessCancelledError, ProcessTimeoutError
from sih26158.storage import ProjectStore, atomic_json, sha256_file


def make_project(store: ProjectStore, root: Path):
    video = root / "video.mp4"
    telemetry = root / "telemetry.csv"
    video.write_bytes(b"fixture")
    telemetry.write_text("timestamp_s,lat,lon,alt_m\n0,1,2,3\n")
    return store.create_project(
        name="ai integration",
        description="",
        video_name=video.name,
        video=video,
        telemetry_name=telemetry.name,
        telemetry=telemetry,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
    )


def prepare_frames(run_dir: Path) -> None:
    frame = run_dir / "frames" / "frame.png"
    cv2.imwrite(str(frame), np.full((12, 16, 3), 80, np.uint8))
    atomic_json(
        run_dir / "keyframes.json",
        {"schema_version": "1.0", "frames": [{"frame_index": 1, "image_name": frame.name, "selected": True}]},
    )


def prepare_dense_run(tmp_path: Path, *, completion_timeout_s: float = 60):
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    record = store.create_run(
        project.project_id,
        RunConfig(completion_timeout_s=completion_timeout_s),
    )
    run_dir = store.run_dir(project.project_id, record.run_id)
    prepare_frames(run_dir)
    mesh = run_dir / "dense" / "meshed-openmvs.ply"
    mesh.write_bytes((Path(__file__).parent / "fixtures/completion/planar_ring.ply").read_bytes())
    transform = run_dir / "local_transform.json"
    atomic_json(
        transform,
        {
            "coordinate_frame": "LOCAL_ENU_METRES",
            "altitude_reference": "SYNTHETIC_LOCAL_Z",
            "scale": 1,
            "rotation": np.eye(3).tolist(),
            "translation_m": [0, 0, 0],
        },
    )
    image = run_dir / "source.png"
    cv2.imwrite(str(image), np.zeros((12, 12, 3), np.uint8))
    store.register_artifacts(record, [run_dir / "keyframes.json", mesh, transform, image])
    return store, record, run_dir, mesh, image


def test_segmentation_fingerprint_binds_frames_weights_and_options(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = make_project(store, tmp_path)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights-v1")
    record = store.create_run(
        project.project_id,
        RunConfig(
            enable_segmentation=True,
            masking_mode="AUTO",
            segmentation_model_path=str(weights),
            segmentation_model_name="fixture",
            segmentation_model_version="1",
        ),
    )
    run_dir = store.run_dir(project.project_id, record.run_id)
    prepare_frames(run_dir)

    baseline = segmentation_fingerprint(record, run_dir)
    assert baseline == segmentation_fingerprint(record, run_dir)
    (run_dir / "frames" / "frame.png").write_bytes(b"changed-frame")
    assert baseline != segmentation_fingerprint(record, run_dir)
    prepare_frames(run_dir)
    weights.write_bytes(b"weights-v2")
    changed_weights = segmentation_fingerprint(record, run_dir)
    assert baseline != changed_weights
    for update in (
        {"segmentation_confidence": 0.8},
        {"segmentation_excluded_classes": ["person"]},
        {"segmentation_image_size": 320},
    ):
        changed = record.model_copy(update={"config": record.config.model_copy(update=update)})
        assert changed_weights != segmentation_fingerprint(changed, run_dir)


def test_colmap_requires_valid_masks_and_records_explicit_unmasked_fallback(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "sparse").mkdir(parents=True)
    masks = run_dir / "masks" / "reconstruction"
    masks.mkdir(parents=True)
    for index in range(3):
        name = f"frame-{index}.png"
        cv2.imwrite(str(frames / name), np.zeros((10, 12, 3), np.uint8))
        cv2.imwrite(str(masks / f"{name}.png"), np.full((10, 12), 255, np.uint8))
    atomic_json(run_dir / "segmentation_status.json", {"status": "completed"})
    commands = ColmapRunner().build_commands(frames, run_dir, RunConfig())
    assert "--ImageReader.mask_path" in commands[0]

    cv2.imwrite(str(masks / "frame-1.png.png"), np.zeros((3, 3), np.uint8))
    with pytest.raises(ExternalToolError, match="malformed"):
        ColmapRunner().build_commands(frames, run_dir, RunConfig())

    atomic_json(
        run_dir / "segmentation_status.json",
        {"status": "unavailable", "fallback_to_unmasked": True},
    )
    commands = ColmapRunner().build_commands(frames, run_dir, RunConfig())
    assert "--ImageReader.mask_path" not in commands[0]


def test_coverage_and_completion_publish_separate_ineligible_geometry(tmp_path: Path) -> None:
    store, record, run_dir, source_mesh, image = prepare_dense_run(tmp_path)
    original = source_mesh.read_bytes()
    coverage_path = write_coverage_report(record, run_dir)
    coverage = json.loads(coverage_path.read_text())
    assert coverage["status"] == "COMPLETED"
    assert coverage["observed_surface_completeness_percent"] is None
    assert coverage["sufficient_for_completion"] is True

    source_hash = sha256_file(source_mesh)
    image_asset = DatasetAsset(path="source.png", sha256=sha256_file(image))
    request = CompletionRequest(
        source_geometry_sha256=source_hash,
        reviews=(
            GapReview(
                boundary_id=boundary_id(list(range(8))),
                decision="CONFIRMED_SMALL_GAP",
                reviewer="fixture",
                explanation="Synthetic removed patch",
                source_images=(image_asset,),
            ),
        ),
    )
    payload, paths, reused = run_completion(record, run_dir, request)
    assert payload["status"] == "completed" and reused is False
    assert payload["eligible_for_measurement"] is False
    assert payload["artifact_sha256"] != source_hash
    assert source_mesh.read_bytes() == original
    store.register_artifacts(record, paths)
    completed_relative = payload["artifact_url"].split("/artifacts/", 1)[1]
    measurement = store.add_measurement(
        record.run_id,
        MeasurementCreate(
            geometry_artifact_path=completed_relative,
            geometry_artifact_sha256=payload["artifact_sha256"],
            start=MeasurementEndpoint(coordinates=(0, 0, 0), point_id=0),
            end=MeasurementEndpoint(coordinates=(1, 0, 0), point_id=1),
        ),
    )
    assert measurement.geometry_provenance == "INFERRED"
    assert measurement.measurement_eligible is False
    repeated, _, reused = run_completion(record, run_dir, request)
    assert reused is True and repeated["fingerprint"] == payload["fingerprint"]


def test_completion_refuses_missing_hash_mismatch_and_ambiguous_symmetry(tmp_path: Path) -> None:
    _, record, run_dir, _, _ = prepare_dense_run(tmp_path)
    mismatch, _, _ = run_completion(
        record,
        run_dir,
        CompletionRequest(source_geometry_sha256="0" * 64),
    )
    assert mismatch["status"] == "refused"
    symmetry, _, _ = run_completion(
        record,
        run_dir,
        CompletionRequest(method="SYMMETRY", symmetry_confidence=0.2),
    )
    assert symmetry["status"] == "refused"

    empty_store = ProjectStore(tmp_path / "empty-projects")
    project = make_project(empty_store, tmp_path)
    empty_record = empty_store.create_run(project.project_id, RunConfig())
    empty_dir = empty_store.run_dir(project.project_id, empty_record.run_id)
    prepare_frames(empty_dir)
    unavailable, _, _ = run_completion(empty_record, empty_dir, CompletionRequest())
    assert unavailable["status"] == "unavailable"


def test_completion_cancellation_and_timeout_are_cooperative(tmp_path: Path) -> None:
    _, record, run_dir, _, image = prepare_dense_run(tmp_path)
    review = GapReview(
        boundary_id=boundary_id(list(range(8))),
        decision="CONFIRMED_SMALL_GAP",
        reviewer="fixture",
        explanation="fixture",
        source_images=(DatasetAsset(path="source.png", sha256=sha256_file(image)),),
    )
    with pytest.raises(ProcessCancelledError):
        run_completion(
            record,
            run_dir,
            CompletionRequest(reviews=(review,)),
            cancel_requested=lambda: True,
        )
    record.config.completion_timeout_s = 1e-12
    with pytest.raises(ProcessTimeoutError):
        run_completion(record, run_dir, CompletionRequest(reviews=(review,)))


def test_ai_status_api_and_duplicate_completion_lock(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api-projects")
    with TestClient(app) as client:
        project = make_project(app.state.store, tmp_path)
        record = app.state.store.create_run(
            project.project_id, RunConfig(execution_mode="SYNTHETIC_DEMO")
        )
        PipelineRunner(app.state.store).run(record.run_id)
        for endpoint in ("segmentation", "coverage", "completion"):
            response = client.get(f"/api/runs/{record.run_id}/{endpoint}")
            assert response.status_code == 200
            assert "status" in response.json()
        with app.state.store.execution_lock(record.run_id) as lock:
            assert lock.acquired
            response = client.post(
                f"/api/runs/{record.run_id}/completion",
                json={"method": "BOUNDED_PLANAR_GAP"},
            )
        assert response.status_code == 409
