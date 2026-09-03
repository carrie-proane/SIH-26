#!/usr/bin/env python3
"""Parse a telemetry file into schema-v1 normalized_telemetry.csv.

Usage
-----
    python scripts/parse_telemetry.py --input data/samples/primary/DJI_0042.SRT \\
        --out-dir data/samples/primary/

    python scripts/parse_telemetry.py --input flight.csv --format csv --out-dir out/

Format is auto-detected from the file extension unless --format is given.
Exit codes: 0 = parsed, 2 = parsed but unusable (no records), 1 = hard error.

The production pipeline calls the library functions in
``sih26158.preprocessing.telemetry`` directly rather than shelling out to this utility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sih26158.preprocessing.telemetry.csv_parser import parse_csv
from sih26158.preprocessing.telemetry.models import sha256_file, write_csv, write_meta
from sih26158.preprocessing.telemetry.srt_parser import parse_srt


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".srt":
        return "srt"
    if ext in {".csv", ".tsv", ".txt"}:
        return "csv"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to .SRT or .csv telemetry file")
    ap.add_argument("--format", choices=["srt", "csv", "auto"], default="auto")
    ap.add_argument("--out-dir", default=".", help="directory for output files")
    ap.add_argument("--basename", default="normalized_telemetry",
                    help="output basename (default: normalized_telemetry)")
    ap.add_argument("--dialect", default=None,
                    help="force an SRT dialect instead of auto-detecting")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1

    fmt = args.format if args.format != "auto" else detect_format(src)
    if fmt == "unknown":
        print(f"error: cannot infer format from {src.suffix!r}; pass --format", file=sys.stderr)
        return 1

    result = parse_srt(src, dialect=args.dialect) if fmt == "srt" else parse_csv(src)

    out_dir = Path(args.out_dir)
    csv_path = out_dir / f"{args.basename}.csv"
    meta_path = out_dir / f"{args.basename}.meta.json"

    n = write_csv(result.records, csv_path)
    write_meta(result, meta_path, source_checksum=sha256_file(src))

    if not args.quiet:
        cov = result.field_coverage()
        print(f"input        : {src}")
        print(f"format       : {result.source_format} / {result.source_dialect}")
        print(f"records      : {n}")
        print(f"duration     : {result.duration_s} s")
        print(f"rate (est)   : {result.estimated_rate_hz} Hz")
        print(f"coverage     : lat {cov['lat']:.1%}  lon {cov['lon']:.1%}  alt {cov['alt_m']:.1%}")
        print(f"time origin  : {result.time_origin}")
        print(f"warnings     : {result.warnings.summary()}")
        print(f"wrote        : {csv_path}")
        print(f"wrote        : {meta_path}")

    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
