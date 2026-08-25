# Telemetry & preprocessing (Yosha)

Day-1 spike: telemetry parsers + sample bundle tooling for SIH26158.

## Quick start

```bash
python -m pytest tests/ -v                  # 27 tests

# parse any DJI SRT or CSV flight log
python scripts/parse_telemetry.py --input <file> --out-dir <dir>

# generate a synthetic track (fallback bundle / deterministic fixture)
python scripts/make_synthetic_telemetry.py --pattern orbit \
  --center-lat 18.5204 --center-lon 73.8567 --radius-m 25 \
  --alt-m 30 --duration-s 45 --rate-hz 10 --out-dir <dir>
```

## Layout

```
src/telemetry/
  models.py       schema v1 records, warning collection, CSV/meta writers
  srt_parser.py   DJI SRT — 3 dialects, auto-detected
  csv_parser.py   generic CSV — column aliasing, unit detection
  checks.py       shared whole-series validation
scripts/
  parse_telemetry.py           CLI
  make_synthetic_telemetry.py  synthetic track generator
docs/TELEMETRY_FIELDS.md       what DJI emits, traps found
data/schemas/                  normalized_telemetry.schema.md  <-- the contract
SAMPLES.md                     bundle registry + capture protocol
```

## Interface with the rest of the team

Everything downstream reads `normalized_telemetry.csv` (schema v1) plus its
`.meta.json` sidecar. Column order is contractual — see
`data/schemas/normalized_telemetry.schema.md`.

- **Jay** consumes the CSV for local-frame alignment and folds `.meta.json`
  warnings into `ingest_report.json`.
- **Arnav** renders the flight path, and should grey out any segment whose
  `fix_quality` is not `ok`.

## Status

Parsers: done, tested against 6 fixtures covering 3 SRT dialects and 3 CSV
layouts. Sample bundles: blocked on capture — see SAMPLES.md.
