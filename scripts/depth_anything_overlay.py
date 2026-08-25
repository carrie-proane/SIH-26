#!/usr/bin/env python3
"""Run Depth Anything V2 Small as a visual-only selected-frame experiment.

Outputs are labelled AI_ASSISTED_NOT_MEASURABLE and are never consumed by the
COLMAP pipeline or by a verified distance calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "32d03942121d29edb49de4e2cc15831558af3f36"
MODEL_LICENSE = "Apache-2.0"
CONFIDENCE_LABEL = "AI_ASSISTED_NOT_MEASURABLE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_inputs(input_dir: Path, limit: int) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png"}
    return [path for path in sorted(input_dir.iterdir()) if path.suffix.lower() in allowed][:limit]


def colorize_depth(depth_array):
    """Small dependency-free colour ramp using a normalized uint8 array."""
    import numpy as np

    value = np.asarray(depth_array, dtype=np.float32) / 255.0
    red = np.clip(1.8 * value, 0, 1)
    green = np.clip(1.7 - np.abs(value - 0.55) * 3.0, 0, 1)
    blue = np.clip(1.35 * (1.0 - value), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    inputs = selected_inputs(args.input_dir, max(args.limit, 1))
    if not inputs:
        print("error: no JPG/PNG selected frames found; no evidence was fabricated", file=sys.stderr)
        return 2

    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        print(
            "error: optional experiment dependencies are missing. Install "
            "experiments/depth-anything/requirements.txt first. "
            f"Missing module: {exc.name}",
            file=sys.stderr,
        )
        return 2

    if args.device == "auto":
        device: str | int = "mps" if torch.backends.mps.is_available() else (
            0 if torch.cuda.is_available() else -1
        )
    elif args.device == "cuda":
        device = 0
    elif args.device == "cpu":
        device = -1
    else:
        device = args.device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimator = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
    )
    evidence: dict[str, object] = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "label": CONFIDENCE_LABEL,
        "measurement_allowed": False,
        "model": {
            "name": "Depth Anything V2 Small",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
        },
        "device": str(device),
        "samples": [],
        "limitations": [
            "Monocular output is relative visual depth, not verified metric geometry.",
            "Outputs never enter COLMAP, local alignment, the point cloud, or the distance tool.",
        ],
    }

    samples: list[dict[str, object]] = []
    for source in inputs:
        image = Image.open(source).convert("RGB")
        started = time.perf_counter()
        prediction = estimator(image)
        runtime_s = time.perf_counter() - started
        depth = prediction["depth"].convert("L").resize(image.size)
        depth_path = args.output_dir / f"{source.stem}_depth.png"
        overlay_path = args.output_dir / f"{source.stem}_overlay.png"
        depth.save(depth_path)

        colors = Image.fromarray(colorize_depth(np.asarray(depth)), mode="RGB")
        Image.blend(image, colors, 0.46).save(overlay_path)
        samples.append(
            {
                "input": source.name,
                "input_sha256": sha256_file(source),
                "depth": depth_path.name,
                "depth_sha256": sha256_file(depth_path),
                "overlay": overlay_path.name,
                "overlay_sha256": sha256_file(overlay_path),
                "runtime_s": round(runtime_s, 4),
            }
        )

    evidence["samples"] = samples
    evidence_path = args.output_dir / "depth_anything_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} visual-only overlay(s) and {evidence_path}")
    print("measurement: DISABLED (AI_ASSISTED_NOT_MEASURABLE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
