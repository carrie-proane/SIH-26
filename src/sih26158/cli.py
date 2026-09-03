from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .colmap import write_matcher_benchmark
from .models import MatcherMetrics, ProvenanceOrigin, RunConfig
from .pipeline import PipelineRunner
from .preflight import collect_preflight
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
        refine_intrinsics=not args.fix_intrinsics,
        sequential_overlap=args.sequential_overlap,
        matching_strategy=args.matching_strategy,
        vocab_tree_path=args.vocab_tree,
        enable_segmentation=args.masking_mode != "OFF",
        segmentation_model_path=args.segmentation_model,
        reconstruction_target=args.reconstruction_target,
        masking_mode=args.masking_mode,
        enable_dense_reconstruction=args.dense,
        dense_provider=args.dense_provider,
        sparse_timeout_s=args.sparse_timeout,
        dense_timeout_s=args.dense_timeout,
        command_heartbeat_s=args.command_heartbeat,
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
    doctor.add_argument(
        "--dense-provider", choices=["auto", "colmap", "openmvs"], default="auto"
    )
    doctor.add_argument("--require-dense", action="store_true")
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--minimum-free-disk-gb", type=float, default=10.0)
    doctor.add_argument("--output", help="Write the immutable JSON preflight report to this path")
    doctor.set_defaults(func=_doctor)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
