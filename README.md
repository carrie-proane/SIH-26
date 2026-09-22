# SIH26158 - Jay backend deliverable

Real capture intake and server execution are documented in
[`docs/real-dataset-validation.md`](docs/real-dataset-validation.md). The preparation command is
safe to run before COLMAP/GPU access and does not start reconstruction.

This repository implements the operator and backend scope for the trustworthy single-pass drone
video reconstruction prototype: immutable ingest, FastAPI orchestration, exact run states, COLMAP
SIFT execution, local-metric alignment utilities, declared artifact serving, persisted measurements,
geometry exports, and quality/known-distance reporting. Learned matching remains an unimplemented
experiment and is rejected rather than silently falling back or being misreported.

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

## Preflight before an expensive run

`doctor` inspects the exact installed command contracts, resources and optional input files without
starting reconstruction. It reports sparse readiness separately from optional dense readiness, so a
missing CUDA/OpenMVS path never disguises an otherwise valid sparse environment.

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli doctor \
  --video /path/capture.mp4 \
  --telemetry /path/capture.srt \
  --dense-provider auto \
  --minimum-free-disk-gb 50 \
  --output environment_report.json
```

Use `--require-gpu` or `--require-dense` on a server where those capabilities are mandatory. The
command exits non-zero only for hard blockers and writes `READY`, `READY_WITH_WARNINGS`, or
`BLOCKED`, plus the precise checks and installed versions, to the JSON report.

Each real run persists the same decision evidence as the declared `server_capabilities.json`
artifact. GPU requests fall back to CPU sparse execution when CUDA is unavailable, and that warning
is retained. Optional dense execution uses detected CUDA COLMAP or OpenMVS capability; unavailable
dense tooling does not fail valid sparse evidence.

Long external stages are controlled by `sparse_timeout_s` (default 7,200 seconds),
`dense_timeout_s` (default 21,600 seconds), and `command_heartbeat_s` (default 10 seconds). Cancel
an active run with `POST /api/runs/{run_id}/cancel`; declared artifacts from completed stages are
preserved and the run can later resume from its last checksum-valid checkpoint.

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

The normal upload/CLI route extracts candidates with decoded timestamps and evaluates absolute and
adaptive sharpness, exposure, structural redundancy, corner support, spatial feature coverage and
adjacent optical-flow displacement. It selects a temporally distributed, overlapping subset and
sends only selected images to COLMAP. COLMAP then performs guided matching, sparse mapping, final
global bundle adjustment and conservative point filtering. Every real run writes
`sparse/geometry_diagnostics.json` with track-length, triangulation, reprojection and camera-path
gates. `--preprocessing-run` remains an optional advanced override for debugging an existing
handoff. Real runs never substitute synthetic geometry when a dependency or input is missing.

Use the accuracy profile for small/medium captures where all-pairs matching is affordable:

```bash
PYTHONPATH=src python -m sih26158.cli run \
  --video /path/pass.mp4 \
  --telemetry /path/pass.srt \
  --profile accurate \
  --matching-strategy AUTO \
  --camera-model SIMPLE_RADIAL
```

`accurate` resolves `AUTO` to exhaustive guided matching, providing loop closure without a
vocabulary-tree download. Sequential mode supports an explicitly supplied local tree through
`--vocab-tree`. A separately calibrated camera can be supplied with `--camera-params` and held
fixed with `--fix-intrinsics`; do not fix guessed values.

Camera calibration is also bounded and auditable. With `--camera-model-policy AUTO`, the pipeline
compares `SIMPLE_RADIAL`, `RADIAL` and `OPENCV` in isolated workspaces, rejects malformed or
physically implausible focal/distortion solutions, and ranks valid attempts by registered images,
median reprojection error and lexical path. If the initial winner misses the sparse gates, exactly
one recovery attempt uses a lower SIFT peak threshold, more features and exhaustive matching.
`sparse/camera_model_selection.json`, `sparse/model_selection.json` and
`sparse/sparse_commands.json` retain every outcome and command. Trusted supplied calibration
automatically switches the policy to `FIXED`.

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
The gate also rejects tracked run directories, reconstruction inputs, generated frontend output and
files larger than 25 MiB. GitHub Actions runs the same gate from a clean Python/Node installation.

The official SIH desired-output targets are tracked without overclaiming in
`docs/requirements-matrix.md`. The reproducible local, CPU/GPU preflight, ten-minute benchmark,
cancellation/resume, report inspection, SSH tunnel, and scheduler-intake workflow is in
`docs/server-validation.md`.

## Geometry exports

Completed reports convert existing declared geometry without rerunning reconstruction. Observed
and dense point clouds can produce LAS; existing meshes can produce OBJ/MTL/textures and GLB. Use
the API contract rather than filesystem guesses:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/readiness
curl -f -X POST http://127.0.0.1:8000/api/runs/RUN_ID/exports
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/exports
```

The POST backfills/revalidates exports for a terminal run using only its already-declared geometry;
it does not execute reconstruction.

Every downloadable file is declared and hash-addressed in `export_readiness.json`; each package has
an `export_manifest.json` with source hash, provenance, options, coordinate convention and reopen
validation. OBJ/LAS remain local ENU metres. GLB applies the documented reversible
`(east, north, up) -> (east, up, -north)` rotation. LAS deliberately declares no EPSG for local ENU.
A point-only run reports OBJ/GLB unavailable instead of inventing a surface, and successful export
never changes measurement eligibility. See `docs/api.md` for exact frontend requests and download
examples.

## Restart-safe execution

Every run has an internal `.pipeline_checkpoint.json` and an advisory `.execution.lock`. A stage is
checkpointed only after all of its artifacts have been declared and hashed. On resume, the pipeline
validates output hashes plus stage input, effective-configuration, and relevant executable
fingerprints in dependency order; the first mismatch reruns only that stage and descendants. A
separate cross-process heavy-job slot defaults to one active reconstruction across runs.

The API automatically requeues interrupted `QUEUED`, `INGESTING`, `PREPROCESSING`,
`RECONSTRUCTING`, and `REPORTING` records on startup. An honestly failed run can be retried with:

```bash
curl -X POST http://127.0.0.1:8000/api/runs/RUN_ID/resume
```

Completed runs cannot be resumed or overwritten. The checkpoint and lock are internal control
files and are never declared or served as reconstruction evidence.

## Important files

- `docs/jay-seven-day-evidence.md` - contract-to-evidence ledger.
- `docs/source-review.md` - full-document decisions and resolved scope differences.
- `docs/architecture.md` - ownership boundary and invariants.
- `docs/api.md` - endpoint and preprocessing handoff contract.
- `docs/matcher-benchmark.md` - SIFT/SuperPoint+LightGlue promotion rule.
- `examples/viewer-manifest.json` - frontend payload fixture.

## Operator frontend

The React/TypeScript/Three.js workspace lives in `frontend/`. It provides server project/run
navigation, refresh recovery, supported run configuration, early sparse access, source-frame and
confidence inspection, backend-persisted measurement, linked reruns, cancellation/resume, and
declared export/report downloads.

```bash
make ui-install
make ui
```

For deterministic browser QA without the API, open `http://127.0.0.1:5173/?fixture=1`. It is
prominently labelled as a synthetic UI fixture and does not count as reconstruction evidence.

The older Arnav delivery ledgers are historical first-round notes; current ownership and contracts
are authoritative in `docs/architecture.md` and `docs/api.md`.

## Honest limitations

- Only directly observed multi-view geometry is measurable.
- Ordinary GNSS is a soft prior; absolute horizontal and vertical errors require independent checks.
- AI-assisted geometry is never accepted as verified measurement.
- Dense reconstruction and textured meshes are optional visual outputs after sparse/metric gates;
  their availability depends on CUDA COLMAP or an external OpenMVS installation. OBJ/GLB export can
  only convert a mesh that already exists; Gaussian Splatting and orthomosaics are not implemented.
  Telemetry offset estimation is a bounded per-run search, but ordinary GNSS remains a soft prior.
- OBJ/GLB/LAS correctness is verified on small deterministic fixtures only. Representative
  real-scene, large-scene, memory/performance and college-server validation remain pending.
- YOLO segmentation, SuperPoint/LightGlue and Depth Anything are optional experiments. Missing
  segmentation weights fall back only in `AUTO`; `REQUIRED` and `PRIMARY_SUBJECT` stop honestly.
