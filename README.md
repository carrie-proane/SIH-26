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

Optional Phase 2 dense visual reconstruction can be requested without changing the sparse evidence
result:

```bash
PYTHONPATH=src python -m sih26158.cli run \
  --video /path/pass.mp4 \
  --telemetry /path/pass.srt \
  --dense \
  --dense-provider auto \
  --reconstruction-target FULL_SCENE \
  --masking-mode AUTO \
  --segmentation-model /path/to/local-segmentation-weights.pt
```

The provider runs only after sparse registration and local-metric alignment gates pass. CUDA COLMAP
is preferred; an installed OpenMVS command suite is the external fallback. If neither is available,
or if dense processing fails, `dense_report.json`, `dense_commands.json`, and `logs/dense.log`
record the blocker while the valid sparse run remains completed. Dense and textured artifacts are
visual-only and are never promoted to verified measurement geometry.

Every run now writes `scene_analysis.json`. Scene-aware masking is optional and configurable:

- `FULL_SCENE` preserves the static environment while excluding supported dynamic/sky classes.
- `PRIMARY_SUBJECT` keeps one central, dominant detected subject and requires valid masks.
- `AUTO` falls back honestly to unmasked reconstruction when local weights are unavailable;
  `REQUIRED` stops before reconstruction instead.
- No weights are downloaded automatically. A local path or `SIH_SEGMENTATION_MODEL` is required.

When applied, the same declared mask set is consumed by COLMAP sparse feature extraction and by
OpenMVS dense matching/texturing. Masked dense auto-selection prefers OpenMVS when the installed
COLMAP PatchMatch command has no documented mask option. OpenMVS also applies target-aware
spurious-component removal; the exact settings and rationale are persisted in `dense_report.json`.

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
- Dense reconstruction and textured meshes are optional visual outputs after sparse/metric gates;
  their availability depends on CUDA COLMAP or an external OpenMVS installation. GLB conversion,
  Gaussian Splatting and orthomosaics are not implemented. Telemetry offset estimation is a bounded
  per-run search, but ordinary GNSS remains a soft prior.
- YOLO segmentation, SuperPoint/LightGlue and Depth Anything are optional experiments. Missing
  segmentation weights fall back only in `AUTO`; `REQUIRED` and `PRIMARY_SUBJECT` stop honestly.
