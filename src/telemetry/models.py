"""Core data structures for telemetry normalization.

Schema v1. See data/schemas/normalized_telemetry.schema.md — that document is
the contract; this module is its implementation. If they disagree, the document
wins and this module is a bug.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"
PARSER_VERSION = "0.1.0"

# Column order is part of the contract. Do not reorder.
COLUMNS = [
    "timestamp_s",
    "lat",
    "lon",
    "alt_m",
    "alt_source",
    "fix_quality",
    "source_row",
]

ALT_SOURCES = {
    "rel_alt",
    "abs_alt_minus_home",
    "barometer",
    "absolute_unadjusted",
    "synthetic",
    "none",
}

FIX_QUALITIES = {"ok", "interpolated", "missing", "suspect"}


@dataclass
class TelemetryRecord:
    """One normalized telemetry sample."""

    timestamp_s: float
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    alt_source: str = "none"
    fix_quality: str = "ok"
    source_row: int = -1

    def __post_init__(self) -> None:
        if self.alt_source not in ALT_SOURCES:
            raise ValueError(f"invalid alt_source: {self.alt_source!r}")
        if self.fix_quality not in FIX_QUALITIES:
            raise ValueError(f"invalid fix_quality: {self.fix_quality!r}")

    @property
    def has_fix(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass
class Warning_:
    """A data-quality warning. Named with a trailing underscore to avoid
    shadowing the builtin `Warning`."""

    code: str
    detail: str
    count: int = 1


class WarningCollector:
    """Accumulates warnings, collapsing repeats of the same code into a count.

    We deliberately do not raise on data problems. The contract says we surface
    limitations rather than hide them, so a file with problems still parses and
    still produces output — the problems ride along in the sidecar.
    """

    def __init__(self) -> None:
        self._items: dict[str, Warning_] = {}

    def add(self, code: str, detail: str = "") -> None:
        if code in self._items:
            self._items[code].count += 1
        else:
            self._items[code] = Warning_(code=code, detail=detail, count=1)

    def has(self, code: str) -> bool:
        return code in self._items

    def as_list(self) -> list[dict[str, Any]]:
        return [asdict(w) for w in self._items.values()]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def summary(self) -> str:
        if not self._items:
            return "no warnings"
        return "; ".join(f"{w.code} x{w.count}" for w in self._items.values())


@dataclass
class ParseResult:
    """Everything a parser produces: the records plus the provenance needed to
    reconstruct how they were produced."""

    records: list[TelemetryRecord] = field(default_factory=list)
    source_file: str = ""
    source_format: str = ""
    source_dialect: str = ""
    time_origin: str = ""
    altitude_reference: str = "relative_to_launch"
    warnings: WarningCollector = field(default_factory=WarningCollector)
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- derived metrics -------------------------------------------------

    @property
    def duration_s(self) -> float:
        if len(self.records) < 2:
            return 0.0
        return round(self.records[-1].timestamp_s - self.records[0].timestamp_s, 3)

    @property
    def estimated_rate_hz(self) -> float | None:
        """Median sample rate. Median, not mean, so one big gap doesn't skew it."""
        if len(self.records) < 3:
            return None
        deltas = sorted(
            self.records[i + 1].timestamp_s - self.records[i].timestamp_s
            for i in range(len(self.records) - 1)
        )
        median = deltas[len(deltas) // 2]
        if median <= 0:
            return None
        return round(1.0 / median, 3)

    def field_coverage(self) -> dict[str, float]:
        """Fraction of rows with a real value, per field. Jay uses this to decide
        whether a bundle is usable before spending COLMAP time on it."""
        n = len(self.records)
        if n == 0:
            return {"lat": 0.0, "lon": 0.0, "alt_m": 0.0}
        return {
            "lat": round(sum(r.lat is not None for r in self.records) / n, 4),
            "lon": round(sum(r.lon is not None for r in self.records) / n, 4),
            "alt_m": round(sum(r.alt_m is not None for r in self.records) / n, 4),
        }

    def meta(self, source_checksum: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "source_dialect": self.source_dialect,
            "parser_version": PARSER_VERSION,
            "row_count": len(self.records),
            "duration_s": self.duration_s,
            "sample_rate_hz_estimated": self.estimated_rate_hz,
            "time_origin": self.time_origin,
            "coordinate_frame": "WGS84",
            "altitude_reference": self.altitude_reference,
            "warnings": self.warnings.as_list(),
            "field_coverage": self.field_coverage(),
            "checksum_sha256_source": source_checksum,
            **self.extra,
        }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(value: float | None, places: int) -> str:
    """Empty string for null. Fixed decimal places otherwise, so the CSV diffs
    cleanly in review and doesn't sprout float noise like 28.613899999999997."""
    if value is None:
        return ""
    return f"{value:.{places}f}"


def write_csv(records: Iterable[TelemetryRecord], path: str | Path) -> int:
    """Write schema-v1 CSV. Returns row count.

    Hand-rolled rather than pandas: the column order and null representation are
    contractual, and this makes both impossible to get wrong by accident.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        for r in records:
            fh.write(
                ",".join(
                    [
                        _fmt(r.timestamp_s, 3),
                        _fmt(r.lat, 7),   # ~1 cm of longitude at the equator
                        _fmt(r.lon, 7),
                        _fmt(r.alt_m, 3),
                        r.alt_source,
                        r.fix_quality,
                        str(r.source_row),
                    ]
                )
                + "\n"
            )
            n += 1
    return n


def write_meta(result: ParseResult, path: str | Path, source_checksum: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.meta(source_checksum), fh, indent=2)
        fh.write("\n")
