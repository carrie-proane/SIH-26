"""Whole-series validation shared by every parser.

Previously duplicated inside srt_parser and csv_parser. Centralised here so the
two cannot drift apart — a bundle must be judged by identical rules regardless
of which format it arrived in.
"""

from __future__ import annotations

from math import radians, sin, cos, asin, sqrt

from .models import ParseResult, WarningCollector

MAX_PLAUSIBLE_SPEED_MS = 30.0   # ~108 km/h, far above any controlled campus flight
SPEED_WINDOW_S = 1.0            # see _check_speed for why this is not per-sample
MIN_DURATION_S = 20.0
GAP_FACTOR = 3.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8  # mean Earth radius, metres
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _check_duration(result: ParseResult, w: WarningCollector) -> None:
    if result.duration_s < MIN_DURATION_S:
        w.add(
            "SHORT_DURATION",
            f"{result.duration_s}s telemetry; contract targets a 30-60s scene",
        )


def _check_rate(result: ParseResult, w: WarningCollector) -> None:
    rate = result.estimated_rate_hz
    if rate is not None and rate < 1.0:
        w.add("SPARSE_SAMPLING", f"~{rate} Hz; interpolation will be coarse")


def _check_gaps(result: ParseResult, w: WarningCollector) -> None:
    recs = result.records
    deltas = [recs[i + 1].timestamp_s - recs[i].timestamp_s for i in range(len(recs) - 1)]
    if not deltas:
        return
    ordered = sorted(deltas)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return
    for i, d in enumerate(deltas):
        if d > GAP_FACTOR * median:
            w.add("TIME_GAP", f"gap of {d:.3f}s (median {median:.3f}s)")
            recs[i].fix_quality = "suspect"
            recs[i + 1].fix_quality = "suspect"


def _check_speed(result: ParseResult, w: WarningCollector) -> None:
    """Flag implausible ground speed, measured over a time window.

    Why a window and not consecutive samples: DJI writes SRT at the video frame
    rate, commonly 30 Hz, so consecutive samples are ~33 ms apart. Consumer GPS
    jitters on the order of a metre between samples even while hovering. One
    metre over 33 ms implies 30 m/s, so a per-sample check flags essentially
    every real 30 Hz file — the warning becomes noise and stops meaning
    anything. Comparing each fix against the last one at least SPEED_WINDOW_S
    earlier lets jitter average out while still catching the failures this check
    exists for: swapped lat/lon, a misplaced decimal, or a corrupt record.
    """
    recs = result.records
    fixed = [r for r in recs if r.has_fix]
    if len(fixed) < 2:
        return

    anchor_idx = 0
    for i in range(1, len(fixed)):
        cur = fixed[i]
        # Advance the anchor to the newest sample still >= SPEED_WINDOW_S behind.
        while (
            anchor_idx + 1 < i
            and cur.timestamp_s - fixed[anchor_idx + 1].timestamp_s >= SPEED_WINDOW_S
        ):
            anchor_idx += 1

        anchor = fixed[anchor_idx]
        dt = cur.timestamp_s - anchor.timestamp_s
        if dt < SPEED_WINDOW_S:
            # Not enough elapsed time yet for a meaningful measurement.
            continue

        dist = haversine_m(anchor.lat, anchor.lon, cur.lat, cur.lon)
        speed = dist / dt
        if speed > MAX_PLAUSIBLE_SPEED_MS:
            w.add("IMPLAUSIBLE_SPEED", f"{speed:.1f} m/s over a {dt:.2f}s window")
            cur.fix_quality = "suspect"


def run_post_checks(result: ParseResult, w: WarningCollector) -> None:
    """Run every whole-series check. Order matters only in that gap detection
    marks rows suspect before the speed check may mark more."""
    if not result.records:
        w.add("EMPTY_RESULT", "no records recovered")
        return
    _check_duration(result, w)
    _check_rate(result, w)
    _check_gaps(result, w)
    _check_speed(result, w)
