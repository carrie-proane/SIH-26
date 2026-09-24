from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .benchmark import compare_benchmark_reports
from .colmap import write_matcher_benchmark
from .dataset import (
    load_correspondences,
    load_dataset_manifest,
    prepare_dataset,
    resolve_dataset_asset,
)
from .execution_probe import run_colmap_execution_probe
from .models import MatcherMetrics, MeasurementRecord, ProvenanceOrigin, RunConfig
from .pipeline import PipelineRunner
from .preflight import collect_preflight
from .report import build_dataset_evaluation
from .storage import ProjectStore, atomic_json


def _demo(args: argparse.Namespace) -> int:
    root = Path(args.data_root)
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        video = temp / "synthetic_demo.mp4"
        telemetry = temp / "synthetic_telemetry.csv"
        video.write_bytes(b"SYNTHETIC_DEMO_NOT_A_REAL_VIDEO\n")
        telemetry.write_text(
            "timestamp_s,lat,lon,alt_m\n0,28.6139,77.2090,42\n5,28.61391,77.20905,42\n",
            encoding="utf-8",
        )
        store = ProjectStore(root)
        project = store.create_project(
            name="Synthetic contract smoke test",
            description="Pipeline/API fixture only; never reconstruction evidence.",
            video_name=video.name,
            video=video,
            telemetry_name=telemetry.name,
            telemetry=telemetry,
            video_origin=ProvenanceOrigin.SYNTHETIC,
            telemetry_origin=ProvenanceOrigin.SYNTHETIC,
        )
        record = store.create_run(
            project.project_id,
            RunConfig(
                execution_mode="SYNTHETIC_DEMO",
                profile="smoke",
                known_distance_m=10.0,
                measured_distance_m=10.6,
            ),
        )
        result = PipelineRunner(store).run(record.run_id)
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0 if result.status == "COMPLETED" else 1


def _run(args: argparse.Namespace) -> int:
    store = ProjectStore(args.data_root)
    project = store.create_project(
        name=args.name,
        description=args.description,
        video_name=Path(args.video).name,
        video=Path(args.video),
        telemetry_name=Path(args.telemetry).name,
        telemetry=Path(args.telemetry),
        video_origin=args.video_origin,
        telemetry_origin=args.telemetry_origin,
    )
    config = RunConfig(
        execution_mode="COLMAP",
        profile=args.profile,
        preprocessing_run=args.preprocessing_run,
        known_distance_m=args.known_distance,
        measured_distance_m=args.measured_distance,
        telemetry_offset_s=args.telemetry_offset,
        telemetry_offset_source=args.telemetry_offset_source,
        force_include_frame_indices=args.force_include,
        force_exclude_frame_indices=args.force_exclude,
        use_gpu=args.use_gpu,
        camera_model=args.camera_model,
        camera_model_policy=args.camera_model_policy,
        camera_params=args.camera_params,
        camera_params_reference=args.camera_params_reference,
        refine_intrinsics=not args.fix_intrinsics,
        sequential_overlap=args.sequential_overlap,
        matching_strategy=args.matching_strategy,
        vocab_tree_path=args.vocab_tree,
        enable_segmentation=args.masking_mode != "OFF",
        segmentation_model_path=args.segmentation_model,
        segmentation_device=args.segmentation_device,
        segmentation_allow_cpu_fallback=args.segmentation_allow_cpu_fallback,
        reconstruction_target=args.reconstruction_target,
        masking_mode=args.masking_mode,
        enable_dense_reconstruction=args.dense,
        dense_provider=args.dense_provider,
        sparse_timeout_s=args.sparse_timeout,
        dense_timeout_s=args.dense_timeout,
        command_heartbeat_s=args.command_heartbeat,
        max_candidate_frames=args.max_candidate_frames,
        max_selected_frames=args.max_selected_frames,
        processing_max_image_dimension=args.processing_max_image_dimension,
        worker_threads=args.worker_threads,
        coverage_interval_s=args.coverage_interval,
    )
    record = store.create_run(project.project_id, config)
    result = PipelineRunner(store).run(record.run_id)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.status == "COMPLETED" else 1


def _benchmark(args: argparse.Namespace) -> int:
    sift = MatcherMetrics.model_validate_json(Path(args.sift).read_text(encoding="utf-8"))
    learned = None
    if args.learned:
        learned = MatcherMetrics.model_validate_json(Path(args.learned).read_text(encoding="utf-8"))
    report = write_matcher_benchmark(Path(args.output), sift, learned)
    print(json.dumps(report, indent=2))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    result = collect_preflight(
        data_root=args.data_root,
        video=args.video,
        telemetry=args.telemetry,
        dense_provider=args.dense_provider,
        require_dense=args.require_dense,
        require_gpu=args.require_gpu,
        minimum_free_disk_gb=args.minimum_free_disk_gb,
    )
    if args.output:
        atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "BLOCKED" else 1


def _execution_probe(args: argparse.Namespace) -> int:
    result = run_colmap_execution_probe(use_gpu=args.use_gpu, timeout_s=args.timeout)
    if args.output:
        atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def _prepare_dataset(args: argparse.Namespace) -> int:
    report = prepare_dataset(args.manifest, output=args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "BLOCKED" else 1


def _run_dataset(args: argparse.Namespace) -> int:
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "REFUSED_EXPLICIT_EXECUTION_REQUIRED",
                    "reason": (
                        "Dataset execution requires --execute; run and inspect prepare-dataset first."
                    ),
                    "reconstruction_started": False,
                },
                indent=2,
            )
        )
        return 2
    manifest_path = Path(args.manifest).resolve()
    manifest = load_dataset_manifest(manifest_path)
    report = prepare_dataset(manifest_path, output=args.preparation_report)
    if report["status"] == "BLOCKED":
        print(json.dumps(report, indent=2))
        return 1
    if manifest.synthetic_example:
        raise ValueError("run-dataset refuses manifests marked synthetic_example")
    config_path = resolve_dataset_asset(manifest_path, manifest.benchmark_config)
    config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    config = config.model_copy(
        update={
            "dataset_id": manifest.dataset_id,
            "dataset_role": manifest.dataset_role,
            "dataset_manifest_sha256": report["manifest_sha256"],
            "dataset_benchmark_config_sha256": manifest.benchmark_config.sha256,
        }
    )
    store = ProjectStore(args.data_root)
    video_path = resolve_dataset_asset(manifest_path, manifest.video)
    telemetry_path = resolve_dataset_asset(manifest_path, manifest.telemetry)
    project = store.create_project(
        name=args.name or manifest.dataset_id,
        description=manifest.description,
        video_name=video_path.name,
        video=video_path,
        telemetry_name=telemetry_path.name,
        telemetry=telemetry_path,
        data_classification=args.data_classification,
        video_origin=ProvenanceOrigin.REAL,
        telemetry_origin=ProvenanceOrigin.REAL,
    )
    record = store.create_run(project.project_id, config)
    run_dir = store.run_dir(project.project_id, record.run_id)
    manifest_snapshot = run_dir / "dataset_manifest.json"
    manifest_temporary = manifest_snapshot.with_suffix(".json.tmp")
    manifest_temporary.write_bytes(manifest_path.read_bytes())
    os.replace(manifest_temporary, manifest_snapshot)
    preparation_snapshot = run_dir / "dataset_preparation_report.json"
    atomic_json(preparation_snapshot, report)
    store.register_artifacts(record, [manifest_snapshot, preparation_snapshot])
    result = PipelineRunner(store).run(record.run_id)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.status == "COMPLETED" else 1


def _evaluate_dataset(args: argparse.Namespace) -> int:
    manifest = load_dataset_manifest(args.manifest)
    measurements: list[MeasurementRecord] = []
    if args.measurements:
        payload = json.loads(Path(args.measurements).read_text(encoding="utf-8"))
        items = payload.get("measurements", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(
                "Measurements input must be a JSON list or an object with measurements"
            )
        measurements = [MeasurementRecord.model_validate(item) for item in items]
    correspondences = load_correspondences(args.checkpoints) if args.checkpoints else None
    report = build_dataset_evaluation(manifest, measurements, correspondences)
    atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


def _compare_benchmarks(args: argparse.Namespace) -> int:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.report]
    comparison = compare_benchmark_reports(reports)
    atomic_json(Path(args.output), comparison)
    print(json.dumps(comparison, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SIH26158 pipeline command")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run an explicitly synthetic orchestration smoke test")
    demo.add_argument("--data-root", default="data/projects")
    demo.set_defaults(func=_demo)
    run = sub.add_parser("run", help="Run a real COLMAP pipeline with automatic preprocessing")
    run.add_argument("--video", required=True)
    run.add_argument("--telemetry", required=True)
    run.add_argument("--preprocessing-run")
    run.add_argument("--force-include", type=int, action="append", default=[])
    run.add_argument("--force-exclude", type=int, action="append", default=[])
    run.add_argument("--data-root", default="data/projects")
    run.add_argument("--name", default="CLI mission")
    run.add_argument("--description", default="")
    run.add_argument(
        "--profile",
        choices=["smoke", "preview", "balanced", "accurate", "diagnostic"],
        default="preview",
    )
    run.add_argument(
        "--camera-params",
        help="COLMAP camera parameters from a trusted calibration, in model-specific order.",
    )
    run.add_argument(
        "--camera-params-reference",
        choices=["PROCESSED_FRAMES", "SOURCE_VIDEO"],
        default="PROCESSED_FRAMES",
        help="Pixel coordinate system used by supplied intrinsics.",
    )
    run.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Keep supplied focal/distortion calibration fixed during mapping and final adjustment.",
    )
    run.add_argument("--known-distance", type=float)
    run.add_argument("--measured-distance", type=float)
    run.add_argument(
        "--video-origin", choices=[item.value for item in ProvenanceOrigin], default="UNKNOWN"
    )
    run.add_argument(
        "--telemetry-origin", choices=[item.value for item in ProvenanceOrigin], default="UNKNOWN"
    )
    run.add_argument("--telemetry-offset", type=float)
    run.add_argument("--telemetry-offset-source", choices=["manual", "calibrated"])
    run.add_argument("--use-gpu", action="store_true")
    run.add_argument(
        "--camera-model",
        choices=["SIMPLE_RADIAL", "RADIAL", "OPENCV"],
        default="SIMPLE_RADIAL",
        help="Use OPENCV only for cameras whose distortion is sufficiently constrained.",
    )
    run.add_argument(
        "--camera-model-policy",
        choices=["AUTO", "FIXED"],
        default="AUTO",
        help="AUTO compares bounded camera models; FIXED runs only --camera-model.",
    )
    run.add_argument("--sequential-overlap", type=int, default=10)
    run.add_argument(
        "--matching-strategy",
        choices=["AUTO", "SEQUENTIAL", "EXHAUSTIVE"],
        default="AUTO",
    )
    run.add_argument(
        "--vocab-tree",
        help="Optional local COLMAP vocabulary tree enabling loop detection in sequential mode.",
    )
    run.add_argument(
        "--reconstruction-target",
        choices=["FULL_SCENE", "PRIMARY_SUBJECT"],
        default="FULL_SCENE",
    )
    run.add_argument("--masking-mode", choices=["OFF", "AUTO", "REQUIRED"], default="AUTO")
    run.add_argument("--segmentation-model", help="Path to local segmentation weights")
    run.add_argument(
        "--segmentation-device", choices=["cpu", "cuda", "mps"], default="cpu"
    )
    run.add_argument(
        "--segmentation-allow-cpu-fallback",
        action="store_true",
        help="Explicitly permit CPU segmentation when the requested accelerator is unavailable.",
    )
    run.add_argument(
        "--dense", action="store_true", help="Attempt optional visual-only dense reconstruction"
    )
    run.add_argument("--dense-provider", choices=["auto", "colmap", "openmvs"], default="auto")
    run.add_argument(
        "--sparse-timeout",
        type=float,
        default=7200,
        help="Maximum total seconds for managed COLMAP sparse commands.",
    )
    run.add_argument(
        "--dense-timeout",
        type=float,
        default=21600,
        help="Maximum total seconds for managed dense-provider commands.",
    )
    run.add_argument(
        "--command-heartbeat",
        type=float,
        default=10,
        help="Seconds between persisted heartbeats while an external command is active.",
    )
    run.add_argument("--max-candidate-frames", type=int, default=240)
    run.add_argument("--max-selected-frames", type=int, default=100)
    run.add_argument("--processing-max-image-dimension", type=int)
    run.add_argument(
        "--worker-threads",
        type=int,
        default=0,
        help="Maximum COLMAP/OpenMVS threads; zero leaves the installed tool default.",
    )
    run.add_argument("--coverage-interval", type=float, default=60.0)
    run.set_defaults(func=_run)
    benchmark = sub.add_parser("benchmark-matchers")
    benchmark.add_argument("--sift", required=True)
    benchmark.add_argument("--learned")
    benchmark.add_argument("--output", default="matcher_benchmark.json")
    benchmark.set_defaults(func=_benchmark)
    doctor = sub.add_parser(
        "doctor",
        help="Inspect exact sparse/dense capabilities, resources, and optional inputs",
    )
    doctor.add_argument("--data-root", default="data/projects")
    doctor.add_argument("--video", help="Optional video to validate with ffprobe")
    doctor.add_argument("--telemetry", help="Optional SRT/CSV telemetry to parse and validate")
    doctor.add_argument("--dense-provider", choices=["auto", "colmap", "openmvs"], default="auto")
    doctor.add_argument("--require-dense", action="store_true")
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--minimum-free-disk-gb", type=float, default=10.0)
    doctor.add_argument("--output", help="Write the immutable JSON preflight report to this path")
    doctor.set_defaults(func=_doctor)
    probe = sub.add_parser(
        "execution-probe",
        help="Opt in to a small actual COLMAP feature/matching probe (not reconstruction evidence)",
    )
    probe.add_argument("--use-gpu", action="store_true")
    probe.add_argument("--timeout", type=float, default=120)
    probe.add_argument("--output", default="execution_probe.json")
    probe.set_defaults(func=_execution_probe)
    prepare = sub.add_parser(
        "prepare-dataset",
        help="Validate a versioned dataset bundle without checking or running reconstruction tools",
    )
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", default="preparation_report.json")
    prepare.set_defaults(func=_prepare_dataset)
    run_dataset = sub.add_parser(
        "run-dataset",
        help="Run a prepared real dataset; requires an explicit execution opt-in",
    )
    run_dataset.add_argument("--manifest", required=True)
    run_dataset.add_argument("--data-root", default="data/projects")
    run_dataset.add_argument("--name")
    run_dataset.add_argument(
        "--data-classification",
        choices=["PUBLIC_DEMO", "INTERNAL", "SENSITIVE", "RESTRICTED"],
        default="INTERNAL",
    )
    run_dataset.add_argument("--preparation-report", default="preparation_report.json")
    run_dataset.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize reconstruction after preparation succeeds",
    )
    run_dataset.set_defaults(func=_run_dataset)
    evaluate = sub.add_parser(
        "evaluate-dataset",
        help="Evaluate relative distances and positional checkpoints independently",
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--measurements")
    evaluate.add_argument("--checkpoints")
    evaluate.add_argument("--output", default="dataset_evaluation_report.json")
    evaluate.set_defaults(func=_evaluate_dataset)
    compare = sub.add_parser(
        "compare-benchmarks",
        help="Compare same-input benchmark reports and expose incompatible runs",
    )
    compare.add_argument("--report", action="append", required=True)
    compare.add_argument("--output", default="benchmark_comparison.json")
    compare.set_defaults(func=_compare_benchmarks)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
