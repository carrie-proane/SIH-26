# Jay seven-day delivery evidence

This ledger maps every contract item to a reviewable repository output. Empirical gates require a
real synchronized bundle and installed external tools; they are never marked passed by a synthetic run.

## Day 1 - Baseline

- Backend package, run layout, exact states: `src/sih26158/`.
- COLMAP CPU command builder and strict doctor: `colmap.py` and `make doctor`.
- Smoke path: `make demo` (orchestration only, explicitly synthetic).
- Real sparse-cloud evidence: pending real selected frames and COLMAP installation.

## Day 2 - Feasibility

- WGS84 -> ECEF -> ENU, robust Sim(3), and metric PLY transform: `geo.py`, covered by synthetic tests.
- Sparse metrics schema and 80% gate: `quality_report.json` generation.
- Known-distance error and 10% gate: `report.py`.
- Stable UI payload: `examples/viewer-manifest.json`.
- Real registration/scale evidence: pending primary bundle and independent known distance.

## Day 3 - Reliable ingest

- Project and run endpoints: `app.py`.
- ffprobe metadata, input sizes and SHA-256: `pipeline.py` and `storage.py`.
- Declared-artifact-only serving and traversal rejection: `storage.py` plus API tests.

## Day 4 - ML/CV proof

- Reproducible SIFT vs learned-matcher decision utility: `colmap.py` and CLI command.
- Same-sample promotion rule and model evidence checklist: `docs/matcher-benchmark.md`.
- Real SuperPoint+LightGlue benchmark: pending primary frames/model environment; no result fabricated.

## Day 5 - Vertical slice

- Automatic scored frame extraction/selection, optional handoff validation, and COLMAP
  feature/match/map/PLY flow: `pipeline.py`, `src/frames/`, and `colmap.py`.
- Sparse metrics, matcher decision, and trust report registered as immutable artifacts.
- Synthetic upload-to-report slice: `make demo`; real vertical slice awaits the data/tool prerequisites.

## Day 6 - Evidence and polish

- Actionable errors for missing ffprobe, COLMAP, preprocessing, too few frames, failed commands,
  corrupt video, and empty model output.
- Every failure preserves the last completed artifact and event history.
- Run manifest records config, events, hashes, declared outputs, and synthetic status.

## Day 7 - Freeze and submit

- Locked dependency ranges and console command: `pyproject.toml`.
- Clean start commands: `README.md`, `Makefile`, and `.env.example`.
- Backend walkthrough: `docs/demo-script.md`.
- `make verify` covers backend tests, Ruff, frontend tests/build, and browser integration.
- Tests cover state, hashing, path safety, scored selection, overrides, reports, alignment, explicit
  point confidence, matcher selection, API, and failure retention.
