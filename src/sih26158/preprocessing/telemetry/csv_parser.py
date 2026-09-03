"""Parser for generic CSV flight logs.

There is no single CSV flight-log standard. We alias the column names used by
the exporters our sample bundles are likely to come from, and we detect units
from the header rather than assuming metres.

Unit trap: Airdata exports ``height_above_takeoff(feet)`` by default. Reading
that as metres makes the drone appear to have flown 3.28x higher than it did,
which silently corrupts Jay's local metric frame. We parse the unit out of the
header and convert.

Time trap: several exporters write elapsed time in **milliseconds**, and some
write an absolute wall-clock datetime with no elapsed column at all. Since
``timestamp_s`` is contractually seconds-from-video-start, an absolute-time
source is rebased onto its own first sample and the assumption is recorded in
``time_origin`` — where it is visible as a limitation rather than buried.
"""

from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path

from .checks import run_post_checks
from .models import ParseResult, TelemetryRecord, WarningCollector

# Column aliases, lowercased and stripped of unit suffixes before lookup.
_TIME_ALIASES = (
    "timestamp_s", "time_s", "elapsed_s", "elapsed", "seconds",
    "time(millisecond)", "time(milliseconds)", "time_ms", "milliseconds",
    "timestamp", "time",
)
_LAT_ALIASES = ("lat", "latitude", "gps_lat", "gpslat")
_LON_ALIASES = ("lon", "lng", "long", "longitude", "longtitude", "gps_lon", "gpslong")
_ALT_ALIASES = (
    "alt_m", "alt", "altitude", "height", "rel_alt",
    "height_above_takeoff", "altitude_above_sealevel", "ascent",
)
_DATETIME_ALIASES = ("datetime(utc)", "datetime", "utc", "date_time", "iso_time")

_UNIT_IN_HEADER = re.compile(r"[\(\[]\s*([a-zA-Z]+)\s*[\)\]]")

_FEET_TO_M = 0.3048


def _norm_header(name: str) -> tuple[str, str | None]:
    """Return (bare_name, unit) for a header cell.

    ``height_above_takeoff(feet)`` -> ``("height_above_takeoff", "feet")``
    """
    raw = name.strip().lower().replace(" ", "_")
    unit_match = _UNIT_IN_HEADER.search(raw)
    unit = unit_match.group(1) if unit_match else None
    bare = _UNIT_IN_HEADER.sub("", raw).strip("_ ")
    return bare, unit


def _find_column(
    headers: dict[str, tuple[int, str | None]], aliases: tuple[str, ...]
) -> tuple[int, str | None] | None:
    for alias in aliases:
        bare, _ = _norm_header(alias)
        if bare in headers:
            return headers[bare]
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in {"na", "n/a", "nan", "null", "none", "-"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_datetime_to_epoch(value: str) -> float | None:
    """Best-effort absolute-time parse. Returns epoch seconds."""
    v = value.strip().replace("T", " ").replace("Z", "")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
    return None


def _cell(row: list[str], column: tuple[int, str | None] | None) -> str | None:
    if column is None:
        return None
    index = column[0]
    return row[index] if index < len(row) else None




def parse_csv(path: str | Path) -> ParseResult:
    """Parse a generic CSV flight log into normalized records."""
    path = Path(path)
    warnings = WarningCollector()

    result = ParseResult(
        source_file=str(path),
        source_format="generic_csv",
        source_dialect="unknown",
        time_origin="unknown",
        warnings=warnings,
    )

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel  # fall back to comma
        reader = csv.reader(fh, dialect)

        try:
            header_row = next(reader)
        except StopIteration:
            warnings.add("EMPTY_RESULT", "file has no header row")
            return result

        headers: dict[str, tuple[int, str | None]] = {}
        for i, cell in enumerate(header_row):
            bare, unit = _norm_header(cell)
            if bare and bare not in headers:
                headers[bare] = (i, unit)

        time_col = _find_column(headers, _TIME_ALIASES)
        lat_col = _find_column(headers, _LAT_ALIASES)
        lon_col = _find_column(headers, _LON_ALIASES)
        alt_col = _find_column(headers, _ALT_ALIASES)
        dt_col = _find_column(headers, _DATETIME_ALIASES)

        if lat_col is None or lon_col is None:
            warnings.add(
                "MISSING_REQUIRED_COLUMN",
                f"no latitude/longitude column found in: {list(headers)}",
            )
            return result

        # --- decide how time is represented -------------------------------
        time_mode = None
        time_scale = 1.0
        if time_col is not None:
            _, t_unit = time_col
            header_name = header_row[time_col[0]].strip().lower()
            if (t_unit and "milli" in t_unit) or "millisecond" in header_name or header_name.endswith("_ms"):
                time_scale = 0.001
                time_mode = "elapsed"
                result.source_dialect = "elapsed_ms"
            else:
                time_mode = "elapsed"
                time_scale = 1.0
                result.source_dialect = "elapsed_s"
            result.time_origin = "csv_elapsed_column"
        elif dt_col is not None:
            time_mode = "absolute"
            result.source_dialect = "absolute_datetime"
            result.time_origin = "csv_datetime_rebased_to_first_sample"
            warnings.add(
                "TIME_REBASED",
                "no elapsed-time column; rebased absolute datetime onto first "
                "sample. Assumes telemetry starts with the video.",
            )
        else:
            warnings.add(
                "MISSING_REQUIRED_COLUMN", "no time or datetime column found"
            )
            return result

        # --- altitude units ------------------------------------------------
        alt_unit_is_feet = False
        if alt_col is not None:
            _, a_unit = alt_col
            header_name = header_row[alt_col[0]].strip().lower()
            if (a_unit and a_unit.startswith(("f", "ft"))) or "feet" in header_name or "(ft" in header_name:
                alt_unit_is_feet = True
                warnings.add("ALT_UNIT_CONVERTED", "altitude header declared feet; converted to metres")

        # --- rows ----------------------------------------------------------
        raw_rows: list[tuple[float, float | None, float | None, float | None, int]] = []
        first_abs: float | None = None

        for row_idx, row in enumerate(reader):
            if not row or all(c.strip() == "" for c in row):
                continue

            # time
            if time_mode == "elapsed":
                t_raw = _to_float(_cell(row, time_col))
                if t_raw is None:
                    warnings.add("ROW_NO_TIME", f"row {row_idx} has no usable time")
                    continue
                ts = t_raw * time_scale
            else:
                dt_val = _cell(row, dt_col)
                epoch = _parse_datetime_to_epoch(dt_val) if dt_val else None
                if epoch is None:
                    warnings.add("ROW_NO_TIME", f"row {row_idx} has unparsable datetime")
                    continue
                if first_abs is None:
                    first_abs = epoch
                ts = epoch - first_abs

            lat = _to_float(_cell(row, lat_col))
            lon = _to_float(_cell(row, lon_col))
            alt = _to_float(_cell(row, alt_col))

            if alt is not None and alt_unit_is_feet:
                alt *= _FEET_TO_M

            raw_rows.append((ts, lat, lon, alt, row_idx))

    if not raw_rows:
        warnings.add("EMPTY_RESULT", "no data rows recovered")
        return result

    # Elapsed columns are not guaranteed sorted; absolute-rebased ones are.
    raw_rows.sort(key=lambda r: r[0])

    records: list[TelemetryRecord] = []
    last_ts: float | None = None
    has_alt = any(r[3] is not None for r in raw_rows)

    for ts, lat, lon, alt, row_idx in raw_rows:
        if lat == 0.0 and lon == 0.0:
            warnings.add("ZERO_ISLAND", "lat/lon both exactly 0; treated as no fix")
            lat = lon = None

        if last_ts is not None and ts == last_ts:
            warnings.add("DUPLICATE_TIMESTAMP", "collapsed, first kept")
            continue

        fix = "ok" if (lat is not None and lon is not None) else "missing"
        if fix == "missing":
            warnings.add("MISSING_FIX", "row with no usable lat/lon")

        records.append(
            TelemetryRecord(
                timestamp_s=ts,
                lat=lat,
                lon=lon,
                alt_m=alt,
                alt_source="rel_alt" if alt is not None else "none",
                fix_quality=fix,
                source_row=row_idx,
            )
        )
        last_ts = ts

    result.records = records

    if not has_alt:
        warnings.add("NO_ALTITUDE", "no altitude column; alt_m is empty for all rows")

    run_post_checks(result, warnings)
    return result

