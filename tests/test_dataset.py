from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sih26158.benchmark import build_benchmark_report, compare_benchmark_reports
from sih26158.cli import parser
from sih26158.dataset import (
    CheckpointCorrespondenceSet,
    DatasetManifest,
    prepare_dataset,
)
from sih26158.models import MeasurementRecord, ProvenanceOrigin, RunCheckpoint, RunConfig, RunRecord
from sih26158.report import build_dataset_evaluation, positional_checkpoint_evaluation
from sih26158.storage import sha256_file


def _write_bundle(tmp_path: Path, *, references: dict | None = None) -> Path:
    video = tmp_path / "capture.mp4"
    telemetry = tmp_path / "telemetry.csv"
    config = tmp_path / "benchmark.json"
    video.write_bytes(b"fixture-video")
    telemetry.write_text(
        "timestamp_s,lat,lon,alt_m\n0,18.5,73.8,100\n10,18.5001,73.8001,101\n",
        encoding="utf-8",
    )
    config.write_text(RunConfig(profile="smoke").model_dump_json(indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "fixture_capture",
        "dataset_role": "DEVELOPMENT",
        "description": "Synthetic test bytes; not reconstruction evidence.",
        "synthetic_example": False,
        "video": {"path": video.name, "sha256": sha256_file(video)},
        "telemetry": {"path": telemetry.name, "sha256": sha256_file(telemetry)},
        "benchmark_config": {"path": config.name, "sha256": sha256_file(config)},
        "capture": {
            "device_make": "fixture",
            "device_model": "fixture",
            "recording_mode": "1920x1080",
            "calibration": {
                "camera_model": "SIMPLE_RADIAL",
                "parameters": None,
                "image_width": 1920,
                "image_height": 1080,
                "coordinate_reference": "SOURCE_VIDEO",
                "source": "synthetic test calibration",
            },
        },
        "telemetry_metadata": {
            "time_basis": "VIDEO_START_RELATIVE",
            "altitude_reference": "ELLIPSOIDAL",
        },
        "references": references or {},
    }
    path = tmp_path / "dataset.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _ffprobe_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    assert command[0] == "/fixture/ffprobe"
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            }
        ],
        "format": {"duration": "10.0"},
    }
    return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def test_preparation_validates_bundle_without_reconstruction_tools(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    requested_tools: list[str] = []

    def which(name: str) -> str | None:
        requested_tools.append(name)
        return "/fixture/ffprobe" if name == "ffprobe" else None

    report = prepare_dataset(manifest, runner=_ffprobe_runner, which=which)

    assert report["status"] == "READY_WITH_WARNINGS"
    assert report["ready_for_explicit_execution"] is True
    assert report["accuracy_claim_status"] == "UNVERIFIED_NO_HELD_OUT_REFERENCE"
    assert report["reconstruction_started"] is False
    assert report["reconstruction_tools_checked"] is False
    assert requested_tools == ["ffprobe"]
    assert all("colmap" not in check["id"] for check in report["checks"])


def test_dataset_execution_requires_explicit_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = parser().parse_args(["run-dataset", "--manifest", str(tmp_path / "missing.json")])

    assert arguments.func(arguments) == 2
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "REFUSED_EXPLICIT_EXECUTION_REQUIRED"
    assert response["reconstruction_started"] is False


def test_preparation_blocks_hash_and_calibration_mismatch(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["video"]["sha256"] = "0" * 64
    payload["capture"]["calibration"]["image_width"] = 3840
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = prepare_dataset(
        manifest,
        runner=_ffprobe_runner,
        which=lambda name: "/fixture/ffprobe" if name == "ffprobe" else None,
    )

    assert report["status"] == "BLOCKED"
    assert "asset.video" in report["blockers"]
    assert "capture.calibration" in report["blockers"]


def test_invalid_reference_frame_returns_actionable_preparation_report(tmp_path: Path) -> None:
    manifest = _write_bundle(
        tmp_path,
        references={
            "coordinate_system": {
                "frame": "LOCAL_ENU_METRES",
                "units": "m",
                "axis_convention": "+X east, +Y north, +Z up",
            },
            "positional_checkpoints": [
                {
                    "id": "bad-frame",
                    "coordinates": [0, 0, 0],
                    "coordinate_frame": "WGS84_DEGREES",
                    "units": "degrees",
                    "altitude_reference": "UNKNOWN",
                    "acquisition_method": "synthetic invalid fixture",
                    "role": "HELD_OUT_EVALUATION",
                }
            ],
        },
    )
    output = tmp_path / "preparation.json"

    report = prepare_dataset(manifest, output=output, which=lambda _: None)

    assert report["status"] == "BLOCKED"
    assert report["blockers"] == ["manifest.schema"]
    assert "checkpoint frames differ" in report["checks"][0]["details"]["error"]
    assert json.loads(output.read_text())["accuracy_claim_status"] == (
        "UNVERIFIED_INVALID_MANIFEST"
    )


def _checkpoint_manifest(*, transformed: bool = False, leaked: bool = False) -> DatasetManifest:
    transform = None
    reconstruction_frame = "LOCAL_ENU_METRES"
    if transformed:
        reconstruction_frame = "RECONSTRUCTION_LOCAL_METRES"
        transform = {
            "source_frame": "LOCAL_ENU_METRES",
            "target_frame": reconstruction_frame,
            "matrix_4x4": [
                [1, 0, 0, 10],
                [0, 1, 0, 20],
                [0, 0, 1, 30],
                [0, 0, 0, 1],
            ],
            "description": "Survey control transform",
            "fitted_using_checkpoint_ids": ["eval-a"] if leaked else ["control"],
        }
    return DatasetManifest.model_validate(
        {
            "dataset_id": "accuracy_fixture",
            "dataset_role": "VALIDATION",
            "synthetic_example": True,
            "video": {"path": "video.mp4", "sha256": "1" * 64},
            "telemetry": {"path": "telemetry.csv", "sha256": "2" * 64},
            "benchmark_config": {"path": "benchmark.json", "sha256": "3" * 64},
            "capture": {},
            "telemetry_metadata": {
                "time_basis": "VIDEO_START_RELATIVE",
                "altitude_reference": "ELLIPSOIDAL",
            },
            "references": {
                "coordinate_system": {
                    "frame": "LOCAL_ENU_METRES",
                    "units": "m",
                    "axis_convention": "+X east, +Y north, +Z up",
                    "vertical_datum": "ELLIPSOIDAL",
                    "reference_to_reconstruction_transform": transform,
                },
                "positional_checkpoints": [
                    {
                        "id": "control",
                        "coordinates": [100, 100, 100],
                        "coordinate_frame": "LOCAL_ENU_METRES",
                        "units": "m",
                        "altitude_reference": "ELLIPSOIDAL",
                        "acquisition_method": "survey",
                        "role": "SCALE_CONTROL",
                    },
                    {
                        "id": "eval-a",
                        "coordinates": [0, 0, 0],
                        "coordinate_frame": "LOCAL_ENU_METRES",
                        "units": "m",
                        "altitude_reference": "ELLIPSOIDAL",
                        "acquisition_method": "survey",
                        "role": "HELD_OUT_EVALUATION",
                    },
                    {
                        "id": "eval-b",
                        "coordinates": [3, 4, 0],
                        "coordinate_frame": "LOCAL_ENU_METRES",
                        "units": "m",
                        "altitude_reference": "ELLIPSOIDAL",
                        "acquisition_method": "survey",
                        "role": "HELD_OUT_EVALUATION",
                    },
                ],
            },
        }
    )


def test_positional_evaluation_reports_signed_axes_and_distance_statistics() -> None:
    manifest = _checkpoint_manifest()
    correspondences = CheckpointCorrespondenceSet.model_validate(
        {
            "reconstruction_coordinate_frame": "LOCAL_ENU_METRES",
            "units": "m",
            "altitude_reference": "ELLIPSOIDAL",
            "checkpoints": [
                {"checkpoint_id": "control", "reconstructed_coordinates": [999, 999, 999]},
                {"checkpoint_id": "eval-a", "reconstructed_coordinates": [0.3, -0.4, 0]},
                {"checkpoint_id": "eval-b", "reconstructed_coordinates": [3, 4, 1]},
            ],
        }
    )

    result = positional_checkpoint_evaluation(manifest, correspondences)

    assert result["status"] == "EVALUATED"
    assert result["evaluated_checkpoint_count"] == 2
    assert result["excluded_scale_control_count"] == 1
    assert result["metrics"]["signed_axes"]["x"]["mean_signed_error_m"] == pytest.approx(0.15)
    assert result["metrics"]["horizontal"]["median_absolute_error_m"] == pytest.approx(0.25)
    assert result["metrics"]["spatial_3d"]["maximum_absolute_error_m"] == pytest.approx(1.0)
    assert result["official_requirement"]["compliance_status"] == "PROTOCOL_UNDEFINED"


def test_positional_evaluation_rejects_heldout_transform_leakage() -> None:
    manifest = _checkpoint_manifest(transformed=True, leaked=True)
    correspondences = CheckpointCorrespondenceSet.model_validate(
        {
            "reconstruction_coordinate_frame": "RECONSTRUCTION_LOCAL_METRES",
            "altitude_reference": "ELLIPSOIDAL",
            "checkpoints": [{"checkpoint_id": "eval-a", "reconstructed_coordinates": [10, 20, 30]}],
        }
    )

    result = positional_checkpoint_evaluation(manifest, correspondences)

    assert result["status"] == "UNAVAILABLE_CONTROL_LEAKAGE"
    assert result["leaked_checkpoint_ids"] == ["eval-a"]


def test_positional_evaluation_refuses_angular_coordinates_without_transform() -> None:
    payload = _checkpoint_manifest().model_dump(mode="json")
    payload["references"]["coordinate_system"]["units"] = "degrees"
    payload["references"]["positional_checkpoints"][0]["units"] = "degrees"
    payload["references"]["positional_checkpoints"][1]["units"] = "degrees"
    payload["references"]["positional_checkpoints"][2]["units"] = "degrees"
    manifest = DatasetManifest.model_validate(payload)
    correspondences = CheckpointCorrespondenceSet.model_validate(
        {
            "reconstruction_coordinate_frame": "LOCAL_ENU_METRES",
            "altitude_reference": "ELLIPSOIDAL",
            "checkpoints": [{"checkpoint_id": "eval-a", "reconstructed_coordinates": [0, 0, 0]}],
        }
    )
    assert positional_checkpoint_evaluation(manifest, correspondences)["status"] == (
        "UNAVAILABLE_INCOMPATIBLE_UNITS"
    )


def test_dataset_distance_evaluation_requires_manifest_binding() -> None:
    payload = _checkpoint_manifest().model_dump(mode="json")
    payload["references"]["relative_distances"] = [
        {
            "id": "width-1",
            "distance_m": 5.0,
            "acquisition_method": "synthetic exact fixture",
            "role": "HELD_OUT_EVALUATION",
        }
    ]
    manifest = DatasetManifest.model_validate(payload)

    def measurement(identifier: str, reference_id: str | None) -> MeasurementRecord:
        return MeasurementRecord(
            measurement_id=identifier,
            run_id="run",
            geometry_artifact_path="sparse/sparse_local.ply",
            geometry_artifact_sha256="a" * 64,
            start={"coordinates": [0, 0, 0]},
            end={"coordinates": [5.2, 0, 0]},
            measurement_kind="RELATIVE_DIMENSION",
            reference_id=reference_id,
            reference_value_m=5.0,
            reference_role="HELD_OUT_EVALUATION",
            backend_distance_m=5.2,
            geometry_provenance="OBSERVED",
            measurement_eligible=True,
            eligibility_reason="synthetic fixture",
        )

    report = build_dataset_evaluation(
        manifest,
        [measurement("bound", "width-1"), measurement("unbound", None)],
        None,
    )

    assert report["relative_distance"]["all_held_out"]["sample_count"] == 1
    binding = report["relative_distance"]["manifest_binding"]
    assert binding["status"] == "BOUND"
    assert binding["rejected_measurements"][0]["measurement_id"] == "unbound"


def test_benchmark_comparison_excludes_different_inputs_and_cache_classes() -> None:
    def report(run_id: str, video_hash: str, cache: str, duration: float) -> dict:
        return {
            "run_id": run_id,
            "cache_classification": cache,
            "input": {
                "assets": [
                    {"role": "video", "sha256": video_hash},
                    {"role": "telemetry", "sha256": "b" * 64},
                ]
            },
            "timing_contract": {"processing_wall_s": duration},
            "execution": {"effective_configuration": {"profile": "accurate"}},
        }

    comparison = compare_benchmark_reports(
        [
            report("baseline", "a" * 64, "COLD_FRESH", 800),
            report("warm", "a" * 64, "WARM_OR_RESUMED", 400),
            report("other", "c" * 64, "COLD_FRESH", 300),
        ]
    )

    assert comparison["entries"][0]["comparison_status"] == "COMPARABLE"
    assert comparison["entries"][1]["comparison_status"] == "INCOMPATIBLE_CACHE_CLASS"
    assert comparison["entries"][2]["comparison_status"] == "INCOMPATIBLE_INPUTS"
    assert comparison["comparable_run_count"] == 1


def test_runtime_gate_distinguishes_fresh_real_cached_and_unknown_provenance() -> None:
    record = RunRecord(
        project_id="project",
        run_id="run",
        config=RunConfig(),
        source_provenance=ProvenanceOrigin.REAL,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
        processing_started_at="2026-01-01T00:00:00+00:00",
        processing_completed_at="2026-01-01T00:13:20+00:00",
    )
    checkpoint = RunCheckpoint(run_id="run")
    ingest = {
        "video_probe": {
            "format": {"duration": "600"},
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
        },
        "input_assets": [],
    }

    fresh = build_benchmark_report(record, checkpoint, ingest, {"frames": []}, {}, reused_stages=[])
    cached = build_benchmark_report(
        record, checkpoint, ingest, {"frames": []}, {}, reused_stages=["SPARSE"]
    )
    record.source_provenance = ProvenanceOrigin.UNKNOWN
    unknown = build_benchmark_report(
        record, checkpoint, ingest, {"frames": []}, {}, reused_stages=[]
    )

    assert fresh["official_runtime_gate"]["status"] == "PASS"
    assert cached["official_runtime_gate"]["status"] == "NOT_VALIDATED_CACHED_STAGE_REUSE"
    assert unknown["official_runtime_gate"]["status"] == "NOT_VALIDATED_NON_REAL_PROVENANCE"
