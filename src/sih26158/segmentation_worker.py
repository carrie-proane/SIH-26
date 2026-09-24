"""Isolated entry point for optional segmentation model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .segmentation import SegmentationSettings, run_optional_segmentation
from .storage import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--reconstruction-target", required=True)
    parser.add_argument("--masking-mode", required=True)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--outcome", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    keyframes_path = run_dir / "keyframes.json"
    payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    settings_payload = json.loads(args.settings_json)
    settings_payload["excluded_classes"] = tuple(settings_payload.get("excluded_classes", ()))
    artifacts, warnings = run_optional_segmentation(
        run_dir,
        args.run_id,
        frames,
        args.model_path,
        reconstruction_target=args.reconstruction_target,
        masking_mode=args.masking_mode,
        settings=SegmentationSettings(**settings_payload),
        accelerator_authorized=settings_payload.get("device") != "cpu",
    )
    atomic_json(keyframes_path, payload)
    atomic_json(
        Path(args.outcome),
        {
            "artifacts": [path.relative_to(run_dir).as_posix() for path in artifacts],
            "warnings": warnings,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
