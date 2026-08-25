"""Parser for DJI-style SRT telemetry sidecars.

DJI has shipped several mutually incompatible SRT layouts. We detect the dialect
from the first data-bearing cue rather than assuming, because getting this wrong
fails silently — you get a plausible-looking flight path in the wrong place.

Dialects handled
----------------
``mavic3_bracket``   Modern (Mini 3/4, Mavic 3, Air 2S+). Bracketed key:value
                     pairs, with rel_alt and abs_alt as separate fields inside a
                     shared bracket::

                       [latitude: 28.613900] [longitude: 77.209000]
                       [rel_alt: 42.100 abs_alt: 542.100]

``legacy_spaced``    Older Mavic/Phantom firmware. Spaces around the colon, a
                     single ``altitude`` field, and — in several firmware
                     versions — longitude misspelled as ``longtitude``::

                       [latitude : 28.6139] [longtitude : 77.2090] [altitude: 42.1]

``gps_tuple``        DJI GO / older Phantom. A positional tuple plus a separate
                     barometer field::

                       GPS(77.2090,28.6139,14) BAROMETER:42.1

                     TRAP: the third tuple element is the **satellite count**,
                     not altitude. Altitude comes from BAROMETER. Reading the
                     tuple's third value as altitude yields a flight that thinks
                     it flew at 14 m when it flew at 42 m. We do not do that.
                     The tuple is also (lon, lat) — longitude first.

Time origin is the SRT cue start time, which DJI writes relative to the start of
the video. That makes ``timestamp_s`` directly comparable to extracted frame
timestamps with no offset estimation, which is explicitly cut scope.
"""

from __future__ import annotations

import re
from pathlib import Path

from .checks import run_post_checks
from .models import ParseResult, TelemetryRecord, WarningCollector

# --- cue splitting ---------------------------------------------------------

# SRT cues are separated by blank lines. DJI sometimes emits \r\n, sometimes a
# trailing BOM, and occasionally a final cue with no trailing newline.
_CUE_SPLIT = re.compile(r"\n\s*\n")

_TIMECODE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

_HTML_TAG = re.compile(r"<[^>]+>")

# --- field extraction ------------------------------------------------------

# Bracketed pairs: [key: value] or [key : value]. Value runs to the next ']'.
# We capture the whole bracket body then split it, because DJI packs multiple
# key:value pairs into one bracket (rel_alt + abs_alt).
_BRACKET = re.compile(r"\[([^\]]+)\]")
_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^\s\]]+)")

_GPS_TUPLE = re.compile(
    r"GPS\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
    re.IGNORECASE,
)
_BAROMETER = re.compile(r"BAROMETER\s*[:\s]\s*(-?\d+\.?\d*)", re.IGNORECASE)

# DJI firmware spells longitude at least three ways.
_LAT_KEYS = ("latitude", "lat", "gps_lat")
_LON_KEYS = ("longitude", "longtitude", "lon", "lng", "gps_long", "gps_lon")



def _timecode_to_seconds(m: re.Match, group_offset: int = 0) -> float:
    h = int(m.group(1 + group_offset))
    mi = int(m.group(2 + group_offset))
    s = int(m.group(3 + group_offset))
    ms_raw = m.group(4 + group_offset)
    ms = int(ms_raw.ljust(3, "0"))  # ",5" means 500 ms, not 5 ms
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip().rstrip("mM°"))
    except (ValueError, AttributeError):
        return None


def _extract_kv(text: str) -> dict[str, str]:
    """Pull every key:value pair out of every bracket."""
    out: dict[str, str] = {}
    for body in _BRACKET.findall(text):
        for key, value in _KV.findall(body):
            out[key.lower()] = value
    return out


def _first_key(kv: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in kv:
            v = _to_float(kv[k])
            if v is not None:
                return v
    return None


def detect_dialect(text: str) -> str:
    """Identify the SRT dialect from the whole file body.

    Checked most-specific first: the GPS tuple is unambiguous, the presence of
    rel_alt/abs_alt marks modern firmware, and everything else with a latitude
    field is treated as legacy.
    """
    if _GPS_TUPLE.search(text):
        return "gps_tuple"
    kv = _extract_kv(text)
    if "rel_alt" in kv or "abs_alt" in kv:
        return "mavic3_bracket"
    if any(k in kv for k in _LAT_KEYS):
        return "legacy_spaced"
    return "unknown"


def _parse_cue_mavic3(text: str, warnings: WarningCollector) -> tuple:
    kv = _extract_kv(text)
    lat = _first_key(kv, _LAT_KEYS)
    lon = _first_key(kv, _LON_KEYS)

    alt = _to_float(kv.get("rel_alt", "")) if "rel_alt" in kv else None
    alt_source = "rel_alt"
    if alt is None:
        # No rel_alt. abs_alt is barometric MSL and not directly usable as
        # height-above-launch, so we take it but label it honestly rather than
        # pretending it is a relative altitude.
        alt = _to_float(kv.get("abs_alt", "")) if "abs_alt" in kv else None
        if alt is not None:
            alt_source = "absolute_unadjusted"
            warnings.add("ALT_FALLBACK", "rel_alt absent; using abs_alt unadjusted")
        else:
            alt_source = "none"
    return lat, lon, alt, alt_source


def _parse_cue_legacy(text: str, warnings: WarningCollector) -> tuple:
    kv = _extract_kv(text)
    lat = _first_key(kv, _LAT_KEYS)
    lon = _first_key(kv, _LON_KEYS)
    alt = _first_key(kv, ("rel_alt", "altitude", "alt", "height"))
    alt_source = "rel_alt" if alt is not None else "none"
    return lat, lon, alt, alt_source


def _parse_cue_gps_tuple(text: str, warnings: WarningCollector) -> tuple:
    lat = lon = alt = None
    alt_source = "none"

    m = _GPS_TUPLE.search(text)
    if m:
        # Positional and longitude-first. The third element is satellite count.
        lon = _to_float(m.group(1))
        lat = _to_float(m.group(2))

    b = _BAROMETER.search(text)
    if b:
        alt = _to_float(b.group(1))
        alt_source = "barometer"

    return lat, lon, alt, alt_source


_CUE_PARSERS = {
    "mavic3_bracket": _parse_cue_mavic3,
    "legacy_spaced": _parse_cue_legacy,
    "gps_tuple": _parse_cue_gps_tuple,
}




def parse_srt(path: str | Path, dialect: str | None = None) -> ParseResult:
    """Parse a DJI SRT sidecar into normalized records.

    Never raises on malformed data. Bad cues are skipped and counted; the result
    always comes back with whatever was recoverable plus the warnings needed to
    judge whether it is usable.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    warnings = WarningCollector()
    detected = dialect or detect_dialect(raw)

    result = ParseResult(
        source_file=str(path),
        source_format="dji_srt",
        source_dialect=detected,
        time_origin="srt_cue_start",
        warnings=warnings,
    )

    if detected == "unknown":
        warnings.add(
            "UNKNOWN_DIALECT",
            "no recognizable DJI telemetry fields; returning empty result",
        )
        return result

    cue_parser = _CUE_PARSERS[detected]
    records: list[TelemetryRecord] = []
    last_ts: float | None = None

    for idx, block in enumerate(_CUE_SPLIT.split(raw.strip())):
        if not block.strip():
            continue

        tc = _TIMECODE.search(block)
        if not tc:
            warnings.add("CUE_NO_TIMECODE", f"cue {idx} has no parsable timecode")
            continue
        ts = _timecode_to_seconds(tc, group_offset=0)

        payload = _HTML_TAG.sub(" ", block[tc.end():])
        lat, lon, alt, alt_source = cue_parser(payload, warnings)

        # Classic null-fix sentinel. A real (0,0) is in the Atlantic off Ghana;
        # if a campus flight reports it, the GPS had no lock.
        if lat == 0.0 and lon == 0.0:
            warnings.add("ZERO_ISLAND", "lat/lon both exactly 0; treated as no fix")
            lat = lon = None

        fix = "ok" if (lat is not None and lon is not None) else "missing"
        if fix == "missing":
            warnings.add("MISSING_FIX", "cue with no usable lat/lon")

        # Time monotonicity, checked before append so we never emit a
        # non-monotonic series downstream.
        if last_ts is not None:
            if ts == last_ts:
                warnings.add("DUPLICATE_TIMESTAMP", "collapsed, first kept")
                continue
            if ts < last_ts:
                warnings.add("NON_MONOTONIC_TIME", "timestamp decreased; row dropped")
                continue

        records.append(
            TelemetryRecord(
                timestamp_s=ts,
                lat=lat,
                lon=lon,
                alt_m=alt,
                alt_source=alt_source if alt is not None else "none",
                fix_quality=fix,
                source_row=idx,
            )
        )
        last_ts = ts

    result.records = records
    run_post_checks(result, warnings)
    return result


