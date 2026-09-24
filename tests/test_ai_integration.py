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
from sih26158.coverage import depth_array_sha256
from sih26158.dataset import DatasetAsset
from sih26158.models import (
    MeasurementCreate,
    MeasurementEndpoint,
    ProvenanceOrigin,
    RunConfig,
    RunStatus,
)
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


def prepare_dense_run(
    tmp_path: Path,
    *,
    completion_timeout_s: float = 60,
    store: ProjectStore | None = None,
):
    store = store or ProjectStore(tmp_path / "projects")
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
    image = run_dir / "frames" / "frame.png"
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
    assert coverage["review_evidence_options"][0]["artifact_path"] == "frames/frame.png"
    assert coverage["candidate_missing_region_statistics"]["regions"][0][
        "boundary_coordinates_enu_m"
    ]
    image_asset = DatasetAsset(path="frames/frame.png", sha256=sha256_file(image))
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


def test_completion_requires_hash_bound_selected_frame_evidence(tmp_path: Path) -> None:
    _, record, run_dir, source_mesh, image = prepare_dense_run(tmp_path)
    write_coverage_report(record, run_dir)
    boundary = boundary_id(list(range(8)))
    no_image, _, _ = run_completion(
        record,
        run_dir,
        CompletionRequest(
            reviews=(
                GapReview(
                    boundary_id=boundary,
                    decision="CONFIRMED_SMALL_GAP",
                    reviewer="operator",
                    explanation="appears bounded",
                ),
            )
        ),
    )
    assert no_image["status"] == "refused"
    mismatched, _, _ = run_completion(
        record,
        run_dir,
        CompletionRequest(
            source_geometry_sha256=sha256_file(source_mesh),
            reviews=(
                GapReview(
                    boundary_id=boundary,
                    decision="CONFIRMED_SMALL_GAP",
                    reviewer="operator",
                    explanation="appears bounded",
                    source_images=(
                        DatasetAsset(path="frames/frame.png", sha256="0" * 64),
                    ),
                ),
            ),
        ),
    )
    assert mismatched["status"] == "refused"
    assert "hash-matched" in mismatched["failure_reason"]
    structural, _, _ = run_completion(
        record,
        run_dir,
        CompletionRequest(
            reviews=(
                GapReview(
                    boundary_id=boundary,
                    decision="STRUCTURAL_OPENING",
                    reviewer="operator",
                    explanation="doorway",
                ),
            )
        ),
    )
    assert structural["status"] == "refused"
    assert structural["inferred_regions_present"] is False
    assert image.is_file()


def test_coverage_uses_only_hash_bound_metric_camera_z_depth(tmp_path: Path) -> None:
    store, record, run_dir, _, _ = prepare_dense_run(tmp_path)
    video_hash = next(
        item.sha256 for item in store.get_project(record.project_id).assets if item.role == "video"
    )
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    depth_path = run_dir / "coverage" / "depth.npy"
    depth_path.parent.mkdir()
    np.save(depth_path, depth, allow_pickle=False)
    contract = run_dir / "coverage_depth_views.json"
    row = {
        "frame_id": "frame.png",
        "video_sha256": video_hash,
        "selected": True,
        "world_to_camera": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
        "intrinsics": [[10, 0, 20], [0, 10, 20], [0, 0, 1]],
        "depth_artifact_path": "coverage/depth.npy",
        "depth_artifact_sha256": sha256_file(depth_path),
        "depth_array_sha256": depth_array_sha256(depth),
        "depth_semantics": "CAMERA_Z_METRES",
    }
    atomic_json(contract, {"schema_version": "1.0", "video_sha256": video_hash, "views": [row]})
    store.register_artifacts(record, [depth_path, contract])
    payload = json.loads(write_coverage_report(record, run_dir, video_sha256=video_hash).read_text())
    assert payload["metric_depth_status"] == "COMPLETED"
    assert "CAMERA_Z" in payload["method"]

    row["depth_semantics"] = "RELATIVE_MONOCULAR"
    atomic_json(contract, {"schema_version": "1.0", "video_sha256": video_hash, "views": [row]})
    store.register_artifacts(record, [contract])
    rejected = json.loads(
        write_coverage_report(record, run_dir, video_sha256=video_hash).read_text()
    )
    assert rejected["status"] == "COMPLETED"
    assert rejected["metric_depth_status"] == "REJECTED"
    assert "camera-Z" in rejected["warnings"][0]["message"]


def test_completion_cancellation_and_timeout_are_cooperative(tmp_path: Path) -> None:
    _, record, run_dir, _, image = prepare_dense_run(tmp_path)
    write_coverage_report(record, run_dir)
    review = GapReview(
        boundary_id=boundary_id(list(range(8))),
        decision="CONFIRMED_SMALL_GAP",
        reviewer="fixture",
        explanation="fixture",
        source_images=(DatasetAsset(path="frames/frame.png", sha256=sha256_file(image)),),
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
        assert client.post(f"/api/runs/{record.run_id}/exports").status_code == 201
        completion = client.post(
            f"/api/runs/{record.run_id}/completion",
            json={"method": "BOUNDED_PLANAR_GAP"},
        )
        assert completion.status_code == 201
        assert completion.json()["status"] == "unavailable"
        assert client.get(f"/api/runs/{record.run_id}/exports").status_code == 409
        updated = app.state.store.get_run(record.run_id)
        declared = {item.relative_path for item in updated.artifacts}
        assert "post_run_operations.json" in declared
        assert "export_readiness.json" not in declared
        assert any(path.startswith("reports/export_readiness-before-completion-") for path in declared)
        versioned = completion.json()["versioned_status_url"].split("/artifacts/", 1)[1]
        assert versioned in declared


def test_post_run_success_export_then_refusal_preserves_versioned_attempt(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "post-run-projects")
    with TestClient(app) as client:
        store, record, run_dir, mesh, image = prepare_dense_run(
            tmp_path, store=app.state.store
        )
        coverage_path = write_coverage_report(record, run_dir)
        store.register_artifacts(record, [coverage_path])
        record.status = record.stage = RunStatus.COMPLETED
        store.save_run(record)
        accepted_request = {
            "method": "BOUNDED_PLANAR_GAP",
            "source_geometry_sha256": sha256_file(mesh),
            "reviews": [
                {
                    "boundary_id": boundary_id(list(range(8))),
                    "decision": "CONFIRMED_SMALL_GAP",
                    "reviewer": "fixture",
                    "explanation": "bounded removed patch",
                    "source_images": [
                        {"path": "frames/frame.png", "sha256": sha256_file(image)}
                    ],
                }
            ],
        }
        accepted = client.post(
            f"/api/runs/{record.run_id}/completion", json=accepted_request
        )
        assert accepted.status_code == 201 and accepted.json()["status"] == "completed"
        inferred_relative = accepted.json()["inferred_artifact_url"].split(
            "/artifacts/", 1
        )[1]
        assert client.post(
            f"/api/runs/{record.run_id}/exports",
            json={"include_completed_geometry": True},
        ).status_code == 201

        refused_request = {
            "method": "BOUNDED_PLANAR_GAP",
            "source_geometry_sha256": "0" * 64,
            "reviews": [],
        }
        refused = client.post(
            f"/api/runs/{record.run_id}/completion", json=refused_request
        )
        assert refused.status_code == 201 and refused.json()["status"] == "refused"
        assert refused.json()["previous_completed_attempt"]["fingerprint"] == accepted.json()[
            "fingerprint"
        ]
        assert client.get(f"/api/runs/{record.run_id}/exports").status_code == 409
        updated = store.get_run(record.run_id)
        assert inferred_relative in {item.relative_path for item in updated.artifacts}
        operation_count = len(
            json.loads((run_dir / "post_run_operations.json").read_text())["operations"]
        )
        repeated = client.post(
            f"/api/runs/{record.run_id}/completion", json=refused_request
        )
        assert repeated.json()["fingerprint"] == refused.json()["fingerprint"]
        assert len(
            json.loads((run_dir / "post_run_operations.json").read_text())["operations"]
        ) == operation_count
