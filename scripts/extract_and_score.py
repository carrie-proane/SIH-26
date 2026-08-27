#!/usr/bin/env python3
"""Extract, score, select, and visualize reconstruction keyframes."""

from __future__ import annotations

import argparse

from frames.contact_sheet import create_contact_sheet
from frames.extractor import extract_frames
from frames.selector import SelectionWeights, select_keyframes


def parse_weights(value: str) -> SelectionWeights:
    try:
        parts = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be three comma-separated numbers") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("weights must contain blur,exposure,redundancy")
    try:
        return SelectionWeights(*parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def selection_count(value: str) -> int:
    count = int(value)
    if not 60 <= count <= 120:
        raise argparse.ArgumentTypeError("target-frames must be between 60 and 120")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-frames", type=selection_count, default=80)
    parser.add_argument("--weights", type=parse_weights, default=SelectionWeights())
    parser.add_argument("--every-nth", type=int)
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--min-spacing-s", type=float)
    args = parser.parse_args()
    extraction = extract_frames(args.video, args.out_dir, args.every_nth, args.target_fps)
    rows = select_keyframes(extraction.frames, args.out_dir, args.target_frames, args.weights, args.min_spacing_s)
    create_contact_sheet(rows, f"{args.out_dir}/contact_sheet.png")
    print(f"Extracted {len(rows)} frames; selected {sum(bool(row['selected']) for row in rows)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
