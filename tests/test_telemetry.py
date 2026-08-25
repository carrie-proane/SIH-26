"""Tests for telemetry parsing. Run: python -m pytest tests/ -v

Every test here corresponds to a failure mode we expect to actually hit on real
bundles. If a test looks paranoid, it is guarding a trap documented in the
parser docstrings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.telemetry.csv_parser import parse_csv
from src.telemetry.models import COLUMNS, TelemetryRecord, write_csv
from src.telemetry.srt_parser import detect_dialect, parse_srt

FIX = Path(__file__).parent / "fixtures"


# --- dialect detection ------------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("dji_mavic3.srt", "mavic3_bracket"),
        ("dji_legacy.srt", "legacy_spaced"),
        ("dji_gps_tuple.srt", "gps_tuple"),
    ],
)
def test_dialect_detection(filename, expected):
    assert detect_dialect((FIX / filename).read_text()) == expected


def test_unknown_dialect_returns_empty_not_crash():
    """A non-DJI SRT must degrade gracefully, not throw. Jay's pipeline should
    get an empty result with a warning, not an exception mid-run."""
    p = FIX / "_tmp_subtitles.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello world\n")
    try:
        r = parse_srt(p)
        assert r.records == []
        assert r.warnings.has("UNKNOWN_DIALECT")
    finally:
        p.unlink()


# --- mavic3 dialect ---------------------------------------------------------

def test_mavic3_prefers_rel_alt_over_abs_alt():
    """abs_alt is barometric MSL (~590 m here); rel_alt is height above launch
    (~30 m). Taking abs_alt would wreck the local metric frame."""
    r = parse_srt(FIX / "dji_mavic3.srt")
    assert r.records[0].alt_m == pytest.approx(30.1)
    assert r.records[0].alt_source == "rel_alt"


def test_mavic3_timestamps_are_seconds_from_video_start():
    r = parse_srt(FIX / "dji_mavic3.srt")
    assert r.records[0].timestamp_s == pytest.approx(0.0)
    assert r.records[1].timestamp_s == pytest.approx(0.033)
    assert r.time_origin == "srt_cue_start"


def test_high_rate_srt_does_not_trigger_false_speed_warning():
    """REGRESSION: a per-consecutive-sample speed check fires on every real
    30 Hz DJI file because GPS jitter over 33 ms implies >30 m/s. The check is
    windowed for exactly this reason. If this test fails, the warning has become
    noise and operators will learn to ignore it."""
    r = parse_srt(FIX / "dji_mavic3.srt")
    assert not r.warnings.has("IMPLAUSIBLE_SPEED")


# --- legacy dialect ---------------------------------------------------------

def test_legacy_handles_misspelled_longtitude():
    """Several DJI firmware versions ship 'longtitude'. Missing this yields a
    file that parses with latitude only and a flight path along a meridian."""
    r = parse_srt(FIX / "dji_legacy.srt")
    assert all(rec.lon is not None for rec in r.records)
    assert r.records[0].lon == pytest.approx(73.85674)


def test_legacy_spaced_colons_parse():
    r = parse_srt(FIX / "dji_legacy.srt")
    assert len(r.records) == 3
    assert r.records[0].lat == pytest.approx(18.52041)


# --- gps tuple dialect ------------------------------------------------------

def test_gps_tuple_is_longitude_first():
    """GPS(73.85,18.52,14) is (lon, lat, satellites). Reading it positionally as
    (lat, lon) puts a Pune flight in the Indian Ocean."""
    r = parse_srt(FIX / "dji_gps_tuple.srt")
    assert r.records[0].lat == pytest.approx(18.52041)
    assert r.records[0].lon == pytest.approx(73.85674)


def test_gps_tuple_third_element_is_not_altitude():
    """The third tuple element is satellite count (14). Altitude comes from
    BAROMETER (22.4). Confusing them makes a 22 m flight look like 14 m."""
    r = parse_srt(FIX / "dji_gps_tuple.srt")
    assert r.records[0].alt_m == pytest.approx(22.4)
    assert r.records[0].alt_source == "barometer"


def test_zero_island_treated_as_missing_fix():
    """lat==lon==0 is the classic no-lock sentinel, not a real position."""
    r = parse_srt(FIX / "dji_gps_tuple.srt")
    assert r.warnings.has("ZERO_ISLAND")
    last = r.records[-1]
    assert last.lat is None and last.lon is None
    assert last.fix_quality == "missing"


# --- CSV --------------------------------------------------------------------

def test_csv_converts_feet_to_metres():
    """Airdata exports feet by default. 98.4 ft is 30.0 m, not 98.4 m."""
    r = parse_csv(FIX / "airdata_style.csv")
    assert r.records[0].alt_m == pytest.approx(29.99, abs=0.01)
    assert r.warnings.has("ALT_UNIT_CONVERTED")


def test_csv_milliseconds_scaled_to_seconds():
    r = parse_csv(FIX / "airdata_style.csv")
    assert r.records[1].timestamp_s == pytest.approx(0.5)
    assert r.source_dialect == "elapsed_ms"


def test_csv_collapses_duplicate_timestamps():
    r = parse_csv(FIX / "litchi_style.csv")
    ts = [rec.timestamp_s for rec in r.records]
    assert len(ts) == len(set(ts))
    assert r.warnings.has("DUPLICATE_TIMESTAMP")


def test_csv_blank_coords_become_missing_fix():
    r = parse_csv(FIX / "litchi_style.csv")
    missing = [rec for rec in r.records if rec.fix_quality == "missing"]
    assert len(missing) == 1
    assert missing[0].timestamp_s == pytest.approx(2.0)


def test_csv_absolute_datetime_rebased_and_flagged():
    """Rebasing assumes telemetry starts with the video. That assumption must be
    visible as a warning, because time-offset estimation is cut scope."""
    r = parse_csv(FIX / "absolute_only.csv")
    assert r.records[0].timestamp_s == pytest.approx(0.0)
    assert r.warnings.has("TIME_REBASED")
    assert r.time_origin == "csv_datetime_rebased_to_first_sample"


def test_csv_detects_time_gap():
    r = parse_csv(FIX / "absolute_only.csv")
    assert r.warnings.has("TIME_GAP")
    assert any(rec.fix_quality == "suspect" for rec in r.records)


def test_csv_missing_required_column_does_not_crash():
    p = FIX / "_tmp_bad.csv"
    p.write_text("foo,bar\n1,2\n")
    try:
        r = parse_csv(p)
        assert r.records == []
        assert r.warnings.has("MISSING_REQUIRED_COLUMN")
    finally:
        p.unlink()


# --- speed check still catches real problems --------------------------------

def test_swapped_lat_lon_is_caught():
    """The windowed speed check must still catch the failure it exists for.
    Here lat/lon are transposed partway through, which teleports the aircraft
    thousands of km — that must not pass silently."""
    p = FIX / "_tmp_swapped.csv"
    p.write_text(
        "timestamp,lat,lon,alt\n"
        "0.0,18.5204100,73.8567400,20.0\n"
        "1.0,18.5204200,73.8567500,20.1\n"
        "2.0,73.8567600,18.5204300,20.2\n"   # transposed
        "3.0,73.8567700,18.5204400,20.3\n"
    )
    try:
        r = parse_csv(p)
        assert r.warnings.has("IMPLAUSIBLE_SPEED")
        assert any(rec.fix_quality == "suspect" for rec in r.records)
    finally:
        p.unlink()


# --- output contract --------------------------------------------------------

def test_csv_output_column_order_is_contractual(tmp_path):
    """Jay and Arnav both index this file. Column order is part of the schema."""
    out = tmp_path / "n.csv"
    write_csv([TelemetryRecord(timestamp_s=0.0, lat=1.0, lon=2.0, alt_m=3.0)], out)
    header = out.read_text().splitlines()[0]
    assert header == ",".join(COLUMNS)


def test_null_is_empty_string_not_nan(tmp_path):
    """Empty cell, not the string 'nan' or 'None' — those parse as data
    downstream and are hard to spot."""
    out = tmp_path / "n.csv"
    write_csv([TelemetryRecord(timestamp_s=0.0)], out)
    row = out.read_text().splitlines()[1]
    assert row.startswith("0.000,,,,none,ok,")


def test_coordinate_precision_is_seven_places(tmp_path):
    """7 dp is ~1 cm. Fewer loses precision below our error budget; more just
    prints float noise."""
    out = tmp_path / "n.csv"
    write_csv([TelemetryRecord(timestamp_s=0.0, lat=18.52041, lon=73.85674)], out)
    assert "18.5204100,73.8567400" in out.read_text()


def test_invalid_enum_rejected():
    with pytest.raises(ValueError):
        TelemetryRecord(timestamp_s=0.0, fix_quality="probably_fine")


# --- synthetic generator ----------------------------------------------------

def test_synthetic_orbit_is_metrically_correct():
    """The generator's coordinate maths must be right, because Jay validates
    local-frame alignment against this track's known geometry."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    gen = importlib.import_module("make_synthetic_telemetry")
    from src.telemetry.checks import haversine_m

    rows = gen.generate("orbit", 18.5204, 73.8567, 25.0, 30.0, 45.0, 10.0,
                        140.0, 0.4, 0.15, seed=26158)
    d = [haversine_m(18.5204, 73.8567, r["lat"], r["lon"]) for r in rows]
    assert sum(d) / len(d) == pytest.approx(25.0, abs=0.5)


def test_synthetic_noise_is_temporally_correlated():
    """REGRESSION: independent per-sample GPS noise adds a random walk to the
    track. At 10 Hz it measured a 157 m orbit as 345 m and doubled apparent
    ground speed. Real GPS error drifts over seconds, so the generator uses an
    AR(1) process. If this fails, any velocity or path-length analysis built on
    synthetic bundles is reading noise."""
    import math, importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    gen = importlib.import_module("make_synthetic_telemetry")
    from src.telemetry.checks import haversine_m

    rows = gen.generate("orbit", 18.5204, 73.8567, 25.0, 30.0, 45.0, 10.0,
                        140.0, 0.4, 0.15, seed=26158)
    path = sum(
        haversine_m(rows[i]["lat"], rows[i]["lon"], rows[i + 1]["lat"], rows[i + 1]["lon"])
        for i in range(len(rows) - 1)
    )
    ideal = 2 * math.pi * 25.0
    assert path < ideal * 1.25, f"path {path:.1f} m vs ideal {ideal:.1f} m — noise is uncorrelated"


def test_synthetic_is_deterministic_for_a_seed():
    """Jay needs a reproducible fixture, not a new track every run."""
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    gen = importlib.import_module("make_synthetic_telemetry")
    a = gen.generate("orbit", 18.5204, 73.8567, 25.0, 30.0, 10.0, 5.0, 140.0, 0.4, 0.15, 7)
    b = gen.generate("orbit", 18.5204, 73.8567, 25.0, 30.0, 10.0, 5.0, 140.0, 0.4, 0.15, 7)
    assert [r["lat"] for r in a] == [r["lat"] for r in b]
