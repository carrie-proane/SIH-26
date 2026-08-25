from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .colmap import write_matcher_benchmark
from .models import MatcherMetrics, RunConfig
from .pipeline import PipelineRunner
from .storage import ProjectStore


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
    )
    config = RunConfig(
        execution_mode="COLMAP",
        profile=args.profile,
        preprocessing_run=args.preprocessing_run,
        known_distance_m=args.known_distance,
        measured_distance_m=args.measured_distance,
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


def _doctor(_: argparse.Namespace) -> int:
    tools = ["ffprobe", "ffmpeg", "colmap"]
    result = {name: shutil.which(name) for name in tools}
    result["python"] = sys.executable
    print(json.dumps(result, indent=2))
    return 0 if all(result[name] for name in tools) else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SIH26158 pipeline command")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run an explicitly synthetic orchestration smoke test")
    demo.add_argument("--data-root", default="data/projects")
    demo.set_defaults(func=_demo)
    run = sub.add_parser("run", help="Run a real COLMAP pipeline from a preprocessing handoff")
    run.add_argument("--video", required=True)
    run.add_argument("--telemetry", required=True)
    run.add_argument("--preprocessing-run", required=True)
    run.add_argument("--data-root", default="data/projects")
    run.add_argument("--name", default="CLI mission")
    run.add_argument("--description", default="")
    run.add_argument("--profile", choices=["smoke", "preview", "balanced", "accurate", "diagnostic"], default="preview")
    run.add_argument("--known-distance", type=float)
    run.add_argument("--measured-distance", type=float)
    run.set_defaults(func=_run)
    benchmark = sub.add_parser("benchmark-matchers")
    benchmark.add_argument("--sift", required=True)
    benchmark.add_argument("--learned")
    benchmark.add_argument("--output", default="matcher_benchmark.json")
    benchmark.set_defaults(func=_benchmark)
    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=_doctor)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

