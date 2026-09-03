#!/usr/bin/env python3
"""Generate synthetic drone telemetry for a fallback bundle.

Why this exists
---------------
Two situations need it:

1. No drone available on Day 1. A phone camera walked around a building gives
   usable video but no telemetry. This produces a plausible companion track so
   the full pipeline can be exercised end to end.
2. Deterministic testing. Real GPS is noisy and non-repeatable. A seeded
   synthetic track lets Jay verify that local-frame alignment is correct against
   a path whose true geometry is known exactly.

HONESTY REQUIREMENT
-------------------
Output is always stamped ``SYNTHETIC_TELEMETRY`` in the sidecar and carries a
header comment in the SRT. Per the contract we do not present generated data as
recorded flight data. A bundle built on this must be labelled in the UI and must
never be the source of a "verified" known-distance measurement claim.

Usage
-----
    # 45 s orbit at 25 m radius, 30 m altitude, around a point in Pune
    python scripts/make_synthetic_telemetry.py --pattern orbit \\
        --center-lat 18.5204 --center-lon 73.8567 \\
        --radius-m 25 --alt-m 30 --duration-s 45 --rate-hz 10 \\
        --out-dir data/samples/backup/

Patterns
--------
``orbit``   Circular path around a centre point, camera facing inward. Best
            match for the contract's courtyard/facade scene.
``arc``     Partial orbit (default 140 degrees). Matches a realistic handheld
            walk along one side of a building.
``line``    Straight pass at constant altitude. Weakest for reconstruction —
            included so we can demonstrate what poor baseline geometry does to
            registration rate, which is worth showing in the trust report.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

# Metres per degree, good to well under our error budget at mid latitudes.
_M_PER_DEG_LAT = 111132.0


def _m_per_deg_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def _offset(lat: float, lon: float, dnorth_m: float, deast_m: float) -> tuple[float, float]:
    return lat + dnorth_m / _M_PER_DEG_LAT, lon + deast_m / _m_per_deg_lon(lat)


def generate(
    pattern: str,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    alt_m: float,
    duration_s: float,
    rate_hz: float,
    arc_degrees: float,
    jitter_m: float,
    alt_jitter_m: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    n = max(2, int(duration_s * rate_hz))
    rows: list[dict] = []

    # Temporally correlated GPS error, as an AR(1) / Ornstein-Uhlenbeck process.
    #
    # Independent per-sample gaussian noise is wrong here and not harmlessly so.
    # At 10 Hz it makes consecutive fixes differ by ~1.4*sigma of pure noise, so
    # summed path length picks up a random walk: a 157 m orbit measured 345 m,
    # and apparent ground speed doubled from 3.5 to 7.7 m/s. Any velocity or
    # path-length analysis downstream would be reading noise.
    #
    # Real consumer GPS error drifts over seconds — multipath and satellite
    # geometry change slowly. phi is set from a ~5 s correlation time so the
    # error wanders realistically instead of resampling every tick.
    corr_time_s = 5.0
    phi = math.exp(-1.0 / (corr_time_s * rate_hz))
    # Scale innovations so the stationary variance still equals jitter_m^2.
    innov = math.sqrt(1 - phi * phi)
    err_n = rng.gauss(0, jitter_m)
    err_e = rng.gauss(0, jitter_m)
    err_a = rng.gauss(0, alt_jitter_m)

    for i in range(n):
        t = i / rate_hz
        frac = i / (n - 1)

        if pattern == "orbit":
            theta = 2 * math.pi * frac
            dn = radius_m * math.cos(theta)
            de = radius_m * math.sin(theta)
            heading = (math.degrees(theta) + 180.0) % 360.0  # facing inward
        elif pattern == "arc":
            theta = math.radians(arc_degrees) * frac - math.radians(arc_degrees) / 2
            dn = radius_m * math.cos(theta)
            de = radius_m * math.sin(theta)
            heading = (math.degrees(theta) + 180.0) % 360.0
        elif pattern == "line":
            dn = 0.0
            de = -radius_m + 2 * radius_m * frac
            heading = 90.0
        else:
            raise ValueError(f"unknown pattern: {pattern}")

        # Gentle altitude drift, as a real operator would produce.
        a = alt_m + 1.5 * math.sin(2 * math.pi * frac)

        # Advance the correlated error state, then apply it.
        err_n = phi * err_n + innov * rng.gauss(0, jitter_m)
        err_e = phi * err_e + innov * rng.gauss(0, jitter_m)
        err_a = phi * err_a + innov * rng.gauss(0, alt_jitter_m)
        dn += err_n
        de += err_e
        a += err_a

        lat, lon = _offset(center_lat, center_lon, dn, de)
        rows.append(
            {
                "t": round(t, 3),
                "lat": lat,
                "lon": lon,
                "alt": a,
                "heading": heading,
                "north_m": dn,
                "east_m": de,
            }
        )

    return rows


def write_srt(rows: list[dict], path: Path, rate_hz: float) -> None:
    """Emit mavic3_bracket dialect so the real SRT parser path gets exercised."""
    dt = 1.0 / rate_hz
    lines: list[str] = []
    for i, r in enumerate(rows, start=1):
        start = r["t"]
        end = start + dt

        def tc(v: float) -> str:
            h = int(v // 3600)
            m = int((v % 3600) // 60)
            s = int(v % 60)
            ms = round((v - int(v)) * 1000)
            if ms == 1000:
                s, ms = s + 1, 0
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines.append(str(i))
        lines.append(f"{tc(start)} --> {tc(end)}")
        lines.append(
            f'<font size="28">SrtCnt : {i}, DiffTime : {int(dt * 1000)}ms'
        )
        lines.append("SYNTHETIC_TELEMETRY : generated, not recorded")
        lines.append(
            f"[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] "
            f"[focal_len : 240] [latitude: {r['lat']:.7f}] "
            f"[longitude: {r['lon']:.7f}] "
            f"[rel_alt: {r['alt']:.3f} abs_alt: {r['alt'] + 560.0:.3f}] </font>"
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("timestamp_s,latitude,longitude,altitude(m),heading_deg\n")
        fh.writelines(
            (
                f"{r['t']:.3f},{r['lat']:.7f},{r['lon']:.7f},"
                f"{r['alt']:.3f},{r['heading']:.1f}\n"
            )
            for r in rows
        )


def write_truth(rows: list[dict], path: Path, args) -> None:
    """Ground truth for this synthetic track.

    Jay can check his local-frame alignment against ``north_m``/``east_m``,
    which are the exact metric offsets used to generate the coordinates before
    jitter. This is the only 'known distance' a synthetic bundle legitimately
    provides, and it validates the alignment maths, not the reconstruction.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    span_n = max(r["north_m"] for r in rows) - min(r["north_m"] for r in rows)
    span_e = max(r["east_m"] for r in rows) - min(r["east_m"] for r in rows)
    payload = {
        "synthetic": True,
        "warning": "Generated telemetry. Not a recorded flight. Must be labelled "
                   "SYNTHETIC_TELEMETRY in any UI and must not back a verified "
                   "measurement claim.",
        "generator_version": "0.1.0",
        "parameters": vars(args),
        "true_geometry": {
            "center_lat": args.center_lat,
            "center_lon": args.center_lon,
            "radius_m": args.radius_m,
            "north_span_m": round(span_n, 3),
            "east_span_m": round(span_e, 3),
            "note": "spans include jitter; nominal diameter is 2*radius_m",
        },
        "sample_count": len(rows),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pattern", choices=["orbit", "arc", "line"], default="orbit")
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--radius-m", type=float, default=25.0)
    ap.add_argument("--alt-m", type=float, default=30.0)
    ap.add_argument("--duration-s", type=float, default=45.0)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--arc-degrees", type=float, default=140.0)
    ap.add_argument("--jitter-m", type=float, default=0.4,
                    help="horizontal GPS noise sigma (0.4 m is typical consumer RTK-off)")
    ap.add_argument("--alt-jitter-m", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=26158)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--basename", default="synthetic_telemetry")
    args = ap.parse_args()

    rows = generate(
        args.pattern, args.center_lat, args.center_lon, args.radius_m,
        args.alt_m, args.duration_s, args.rate_hz, args.arc_degrees,
        args.jitter_m, args.alt_jitter_m, args.seed,
    )

    out = Path(args.out_dir)
    write_srt(rows, out / f"{args.basename}.srt", args.rate_hz)
    write_csv(rows, out / f"{args.basename}.csv")
    write_truth(rows, out / f"{args.basename}.truth.json", args)

    print(f"pattern    : {args.pattern}")
    print(f"samples    : {len(rows)} over {args.duration_s}s at {args.rate_hz} Hz")
    print(f"radius     : {args.radius_m} m   altitude: {args.alt_m} m")
    print(f"seed       : {args.seed} (deterministic)")
    print(f"wrote      : {out / f'{args.basename}.srt'}")
    print(f"wrote      : {out / f'{args.basename}.csv'}")
    print(f"wrote      : {out / f'{args.basename}.truth.json'}")
    print()
    print("REMINDER: synthetic data. Label it in the UI. Never back a verified")
    print("measurement with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
