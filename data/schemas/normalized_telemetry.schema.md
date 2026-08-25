# `normalized_telemetry.csv` — schema v1

**Owner:** Yosha · **Primary consumer:** Jay (local-coordinate alignment, COLMAP georeferencing) · **Secondary consumer:** Arnav (flight path in viewer-manifest)

This is the single agreed output of every telemetry parser. Any input format (DJI SRT, generic CSV, future adapters) is normalized to exactly these columns, in exactly this order. **Do not add, remove, or reorder columns without all three members agreeing** — Jay's alignment code and Arnav's flight-path renderer both index this file.

Per the contract: all timestamps are **seconds from video start**.

## Columns

| # | Column | Type | Units | Null allowed | Notes |
|---|---|---|---|---|---|
| 1 | `timestamp_s` | float | seconds | no | Seconds from video start (t=0 at first video frame). Monotonically increasing, strictly. |
| 2 | `lat` | float | degrees (WGS84) | yes | Signed decimal. Empty string if the source row had no fix. |
| 3 | `lon` | float | degrees (WGS84) | yes | Signed decimal. Empty string if the source row had no fix. |
| 4 | `alt_m` | float | metres | yes | **Relative** altitude above launch point, not absolute/MSL. See note below. |
| 5 | `alt_source` | enum | — | no | `rel_alt` \| `abs_alt_minus_home` \| `barometer` \| `absolute_unadjusted` \| `synthetic` \| `none` |
| 6 | `fix_quality` | enum | — | no | `ok` \| `interpolated` \| `missing` \| `suspect` |
| 7 | `source_row` | int | — | no | 0-based index of the originating record in the source file. Traceability for debugging. |

## Sidecar: `normalized_telemetry.meta.json`

Written alongside the CSV. Jay's `ingest_report.json` should embed or reference this.

```json
{
  "schema_version": "1.0",
  "source_file": "input/DJI_0042.SRT",
  "source_format": "dji_srt",
  "source_dialect": "mavic3_bracket",
  "parser_version": "0.1.0",
  "row_count": 1782,
  "duration_s": 59.4,
  "sample_rate_hz_estimated": 30.0,
  "time_origin": "srt_cue_start",
  "coordinate_frame": "WGS84",
  "altitude_reference": "relative_to_launch",
  "warnings": [
    {"code": "DUPLICATE_TIMESTAMP", "count": 3, "detail": "collapsed, first kept"}
  ],
  "field_coverage": {"lat": 1.0, "lon": 1.0, "alt_m": 0.998},
  "checksum_sha256_source": "..."
}
```

## Design decisions (and why)

**Altitude is relative, not absolute.** DJI writes both `rel_alt` (height above launch) and `abs_alt` (MSL-ish, barometric). `abs_alt` on consumer DJI hardware is barometric and can be tens of metres off. Jay's local metric frame only needs height above a local datum, so `rel_alt` is both more accurate and directly usable. `alt_source` records which one we actually got so a bad reconstruction can be traced back.

**No `frame_index` column.** Telemetry sample rate and video frame rate are different and not necessarily aligned — DJI SRT is often one cue per video frame but not always, and generic CSV logs are typically 5–10 Hz. Forcing a frame index at parse time would bake in a false assumption. Frame association happens later, at interpolation time (Day 3), against the actual extracted frame timestamps.

**`fix_quality` is a column, not a warning.** Per the contract, we do not hide weak data. A per-row flag lets the viewer grey out interpolated path segments instead of drawing a confident line through a GPS dropout.

**Time origin is recorded, not assumed.** `time_origin` in the sidecar states how t=0 was derived. Automatic time-offset estimation is explicitly cut scope, so any offset between telemetry clock and video clock is a known, declared limitation — not something we silently paper over.

## Warning codes

Emitted into the sidecar `warnings` array. Jay surfaces these in `ingest_report.json`; Yosha surfaces them in the data-quality panel.

| Code | Meaning | Parser behaviour |
|---|---|---|
| `DUPLICATE_TIMESTAMP` | Two or more records share a timestamp | Keep first, drop rest, count them |
| `NON_MONOTONIC_TIME` | Timestamp decreases | Drop the offending row, flag |
| `TIME_GAP` | Gap > 3× median sample interval | Keep, mark neighbours `suspect` |
| `MISSING_FIX` | Row has no lat/lon | Keep row, blank coords, `fix_quality=missing` |
| `ZERO_ISLAND` | lat==0 and lon==0 (classic null-fix sentinel) | Treat as `MISSING_FIX` |
| `IMPLAUSIBLE_SPEED` | Implied ground speed > 30 m/s | Keep, mark `suspect` |
| `ALT_FALLBACK` | `rel_alt` absent, fell back to another source | Record in `alt_source` |
| `SHORT_DURATION` | Telemetry span < 20 s | Warn — below the contract's 30–60 s scene target |
| `SPARSE_SAMPLING` | Estimated rate < 1 Hz | Warn — interpolation quality will be poor |
| `SYNTHETIC_TELEMETRY` | Bundle uses generated, not recorded, telemetry | Must propagate to UI. Never present as real flight data. |
