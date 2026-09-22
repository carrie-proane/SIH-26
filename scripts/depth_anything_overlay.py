#!/usr/bin/env python3
"""Run Depth Anything V2 Small as a visual-only selected-frame experiment.

Outputs are labelled AI_ASSISTED_NOT_MEASURABLE and are never consumed by the
COLMAP pipeline or by a verified distance calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
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


def local_model_manifest(model_path: Path) -> dict[str, str]:
    root = model_path.expanduser().resolve()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("Supply an existing local Transformers model directory with config.json")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if root not in path.resolve().parents:
                raise ValueError("Model files must remain inside the approved local directory")
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Local model directory contains no checkpoint weights")
    return files


def load_local_estimator(model_path: Path, device: str | int):
    """No remote model ID, downloader fallback, or remote custom code."""
    local_model_manifest(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline

    location = str(model_path.expanduser().resolve())
    model = AutoModelForDepthEstimation.from_pretrained(
        location,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor = AutoImageProcessor.from_pretrained(
        location,
        local_files_only=True,
        trust_remote_code=False,
    )
    return pipeline(task="depth-estimation", model=model, image_processor=processor, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    started_total = time.perf_counter()
    if not 1 <= args.limit <= 32:
        parser.error("--limit must be between 1 and 32")
    try:
        model_files = local_model_manifest(args.model_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}; no model was downloaded", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    inputs = selected_inputs(args.input_dir, max(args.limit, 1))
    if not inputs:
        print(
            "error: no JPG/PNG selected frames found; no evidence was fabricated", file=sys.stderr
        )
        return 2

    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        print(
            "error: optional experiment dependencies are missing. Install "
            "experiments/depth-anything/requirements.txt first. "
            f"Missing module: {exc.name}",
            file=sys.stderr,
        )
        return 2

    device: str | int = {"cpu": -1, "cuda": 0, "mps": "mps"}[args.device]
    try:
        estimator = load_local_estimator(args.model_path, device)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: local-only depth model unavailable: {exc}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, object] = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "label": CONFIDENCE_LABEL,
        "measurement_allowed": False,
        "model": {
            "reference_name": "Depth Anything V2 Small",
            "reference_model_id": MODEL_ID,
            "executed_model_type": estimator.model.config.model_type,
            "reference_revision_not_verified": MODEL_REVISION,
            "local_path": str(args.model_path.resolve()),
            "files_sha256": model_files,
            "loading": "LOCAL_FILES_ONLY",
            "transformers_version": version("transformers"),
            "reference_license_not_verified_for_local_weights": MODEL_LICENSE,
        },
        "requested_device": args.device,
        "device": str(estimator.device),
        "depth_semantics": "RELATIVE_NOT_METRIC",
        "parameters": {"limit": args.limit},
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
        raw = np.asarray(
            prediction["predicted_depth"].detach().cpu().numpy(), dtype=np.float32
        ).squeeze()
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError("Depth provider returned invalid raw relative depth")
        raw_path = args.output_dir / f"{source.stem}_relative_depth.npy"
        np.save(raw_path, raw, allow_pickle=False)
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
                "raw_relative_depth": raw_path.name,
                "raw_relative_depth_sha256": sha256_file(raw_path),
                "raw_shape": list(raw.shape),
                "depth_png_semantics": "NORMALIZED_8BIT_VISUALIZATION_ONLY",
                "depth": depth_path.name,
                "depth_sha256": sha256_file(depth_path),
                "overlay": overlay_path.name,
                "overlay_sha256": sha256_file(overlay_path),
                "runtime_s": round(runtime_s, 4),
            }
        )

    evidence["runtime_s"] = time.perf_counter() - started_total
    evidence["samples"] = samples
    evidence_path = args.output_dir / "depth_anything_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} visual-only overlay(s) and {evidence_path}")
    print("measurement: DISABLED (AI_ASSISTED_NOT_MEASURABLE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
