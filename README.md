# SIH26158 - Jay backend deliverable

This repository implements Jay's seven-day contract scope for the trustworthy single-pass drone
video reconstruction prototype: immutable ingest, FastAPI orchestration, exact run states, COLMAP
execution, local-metric alignment utilities, SIFT-vs-learned matcher selection, declared artifact
serving, and quality/known-distance reporting.

It does not claim that a reconstruction has been produced without real synchronized drone data and
COLMAP. `SYNTHETIC_DEMO` is only a deterministic orchestration fixture and is labeled in its PLY,
run manifest, metrics, and quality report.

## Quick start

Requirements: Python 3.11+, FFmpeg/ffprobe, and COLMAP for genuine runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make test
make lint
make doctor
make api
```

In another terminal, upload a video/telemetry pair using the example in `docs/api.md`.

## Reproducible orchestration smoke test

```bash
make demo
```

The command generates immutable input/run folders under `data/projects/`, exercises all stages, and
emits a quality report. Its content is synthetic and cannot be used to pass the contract's real
registration, reprojection, scale, or COLMAP gates.

## Real CLI run

```bash
PYTHONPATH=src python -m sih26158.cli run \
  --video /path/pass.mp4 \
  --telemetry /path/pass.csv \
  --known-distance 12.4 \
  --measured-distance 12.1
```

The normal upload/CLI route extracts candidates with decoded timestamps, computes normalized blur,
exposure and redundancy scores, selects a temporally distributed subset, and sends only selected
images to COLMAP. `--preprocessing-run` remains an optional advanced override for debugging an
existing handoff. Real runs never substitute synthetic geometry when a dependency or input is
missing.

Run the complete backend, lint, frontend build and browser verification gate with `make verify`.

## Important files

- `docs/jay-seven-day-evidence.md` - contract-to-evidence ledger.
- `docs/source-review.md` - full-document decisions and resolved scope differences.
- `docs/architecture.md` - ownership boundary and invariants.
- `docs/api.md` - endpoint and preprocessing handoff contract.
- `docs/matcher-benchmark.md` - SIFT/SuperPoint+LightGlue promotion rule.
- `examples/viewer-manifest.json` - exact frontend payload sample for Arnav.

## Arnav operator frontend

The React/TypeScript/Three.js workspace lives in `frontend/`. It provides upload/demo selection,
exact pipeline progress, PLY + flight-path viewing, source-frame inspection, confidence filters,
confidence-aware measurement and the quality/limitations report.

```bash
make ui-install
make ui
```

For deterministic browser QA without the API, open `http://127.0.0.1:5173/?fixture=1`. It is
prominently labelled as a synthetic UI fixture and does not count as reconstruction evidence.

Arnav's delivery ledger and cross-team review are in `docs/arnav-seven-day-evidence.md` and
`docs/arnav-integration-review.md`.

## Honest limitations

- Only directly observed multi-view geometry is measurable.
- Ordinary GNSS is a soft prior; absolute horizontal and vertical errors require independent checks.
- AI-assisted geometry is never accepted as verified measurement.
- Dense reconstruction, mesh/GLB and orthomosaic remain stretch work until the sparse/metric gates
  pass on the real primary sample. Telemetry offset estimation is a bounded per-run search, but
  ordinary GNSS remains a soft prior.
- YOLO segmentation, SuperPoint/LightGlue and Depth Anything are optional experiments. Missing
  models never block the core SIFT reconstruction path.
