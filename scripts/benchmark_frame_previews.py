#!/usr/bin/env python3
"""Compare original full-resolution scoring against bounded previews on a real video.

Each mode runs in a fresh process so peak RSS measurements are independent.
Default sample: every 30th frame, top 20 with unchanged scoring and temporal spacing.
Acceptance: >=90% exact selected-frame overlap, >=20% peak RSS reduction.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

from frames.extractor import extract_frames
from frames.scoring import blur_scores, exposure_scores, load_images, redundancy_scores
from frames.selector import SelectionWeights, select_indices, select_keyframes


def worker(args) -> dict:
    started = time.monotonic()
    frames = extract_frames(args.video, args.work, every_nth=args.every_nth,
                            preview_max_dimension=None if args.mode == "baseline" else args.dimension).frames
    if args.mode == "baseline":
        images = load_images([frame.frame_path for frame in frames])
        weights = SelectionWeights()
        scores = [weights.blur * b + weights.exposure * e + weights.redundancy * r
                  for b, e, r in zip(blur_scores(images), exposure_scores(images), redundancy_scores(images), strict=True)]
        times = [frame.timestamp_s for frame in frames]
        spacing = (max(times) - min(times)) / max(1, args.target - 1) * 0.8
        selected = select_indices(scores, times, args.target, spacing)
        selected_ids = [frame.frame_index for i, frame in enumerate(frames) if i in selected]
    else:
        rows = select_keyframes(frames, args.work, target_frames=args.target,
                                scoring_max_dimension=args.dimension)
        selected_ids = [int(row["frame_index"]) for row in rows if row["selected"]]
    selected_shapes = [list(cv2.imread(frame.frame_path).shape[:2]) for frame in frames if frame.frame_index in selected_ids]
    return {"selected_frame_indices": selected_ids, "candidate_count": len(frames),
            "selected_shapes": selected_shapes, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_s": round(time.monotonic() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--dimension", type=int, default=1920)
    parser.add_argument("--every-nth", type=int, default=30)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["baseline", "preview"])
    parser.add_argument("--work")
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(worker(args)))
        return
    results = {}
    for mode in ("baseline", "preview"):
        with tempfile.TemporaryDirectory(prefix=f"sih-preview-{mode}-") as work:
            completed = subprocess.run([sys.executable, __file__, "--video", args.video,
                "--dimension", str(args.dimension), "--every-nth", str(args.every_nth), "--target", str(args.target),
                "--mode", mode, "--work", work], check=True, capture_output=True, text=True)
            results[mode] = json.loads(completed.stdout)
    baseline, preview = results["baseline"], results["preview"]
    overlap = len(set(baseline["selected_frame_indices"]) & set(preview["selected_frame_indices"])) / len(baseline["selected_frame_indices"])
    reduction = 1 - preview["peak_rss_kib"] / baseline["peak_rss_kib"]
    report = {"video": Path(args.video).name, "preview_max_dimension": args.dimension,
              "sampling_every_nth": args.every_nth, "target_frames": args.target,
              "exact_selection_overlap": overlap, "peak_rss_reduction": reduction,
              "acceptance": {"minimum_overlap": 0.9, "minimum_memory_reduction": 0.2}, **results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
