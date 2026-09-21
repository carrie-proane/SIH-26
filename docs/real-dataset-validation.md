# Real-dataset preparation and evaluation

This workflow prepares approved capture bundles now and runs reconstruction only after an explicit
server-side opt-in. It extends the existing run, measurement, export and benchmark reports; it is
not a second benchmark framework.

## Team intake checklist

For every dataset, obtain and record:

- the dataset ID and role (`DEVELOPMENT`, `VALIDATION`, or `HELD_OUT`);
- raw video and matching SRT/CSV telemetry, kept outside Git, plus SHA-256 for each;
- capture device make/model, recording mode, rotation/crop history and camera calibration source;
- whether calibration parameters refer to source-video or processed-frame pixel coordinates;
- telemetry time basis and altitude reference (ellipsoidal, orthometric, relative-to-launch, or
  explicitly unknown);
- benchmark `RunConfig` file and SHA-256;
- optional tape/laser relative distances and/or independently surveyed checkpoints, including
  frame, units, axis convention, vertical datum, method, stated accuracy, evidence, and whether
  each is a scale control or held-out evaluation item;
- for any coordinate transform, its 4x4 matrix, direction, explanation, and every checkpoint ID
  used to fit it;
- server OS/version, scheduler/partition/QOS and wall-time limits;
- allocated CPU count, RAM, GPU model/count/VRAM and the scheduler's GPU resource syntax;
- local scratch/storage capacity and retention, module/container policy, network policy, and whether
  the team has permission to install user-space packages or must request administrator installation.

Also request the organizer's definition of the stated `<= 1 m` target: horizontal or 3D statistic,
sampling protocol, minimum checkpoint count, accepted reference accuracy, and treatment of vertical
datum. Until clarified, threshold booleans are labelled provisional and never establish organizer
compliance.

Copy [the real manifest template](../datasets/templates/real-dataset.manifest.json) beside the
approved bundle. The template's zero hashes and empty fields are placeholders—not observations.
Compute real hashes, for example:

```bash
shasum -a 256 raw/capture.mp4 raw/telemetry.srt benchmark.json
```

Raw video, telemetry, survey evidence, workspaces and exports belong in ignored external bundle or
server storage, not Git. Relative paths may not escape the manifest directory.

## Preparation only: no reconstruction binaries required

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli prepare-dataset \
  --manifest /approved/datasets/ten-minute/dataset.manifest.json \
  --output /approved/reports/ten-minute-preparation.json
```

This verifies asset existence/hashes, decodable video metadata, telemetry parsing/time coverage,
calibration compatibility, coordinate/reference declarations and the benchmark configuration. It
checks `ffprobe`, but deliberately does not inspect or execute COLMAP, OpenMVS, CUDA or a GPU.
`BLOCKED` means the bundle is unsafe to execute. `READY_WITH_WARNINGS` may be executable; for
example, missing held-out references produces `UNVERIFIED_NO_HELD_OUT_REFERENCE`, not a
reconstruction block.

Inspect preparation before moving data to the server:

```bash
jq '{status, blockers, warnings, accuracy_claim_status, checks}' \
  /approved/reports/ten-minute-preparation.json
```

## Server sequence

Run these phases in order. Each execution manifest must hash its exact configuration. Use distinct
approved bundles/configurations for the tiny smoke, intermediate and final ten-minute run; never
edit a completed run in place.

1. Validate the bundle without reconstruction:

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli prepare-dataset \
  --manifest /approved/datasets/tiny-smoke/dataset.manifest.json \
  --output /approved/reports/tiny-smoke-preparation.json
```

2. Preflight the allocated host and exact inputs:

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli doctor \
  --video /approved/datasets/tiny-smoke/raw/capture.mp4 \
  --telemetry /approved/datasets/tiny-smoke/raw/telemetry.srt \
  --minimum-free-disk-gb 50 \
  --output /approved/reports/server-preflight.json
```

3. Run the tiny real CPU smoke only after both reports are reviewed:

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli run-dataset \
  --manifest /approved/datasets/tiny-smoke/dataset.manifest.json \
  --data-root /approved/local/storage/sih-projects \
  --data-classification INTERNAL --execute
```

4. Run an intermediate real capture with bounded frame/image settings in its hashed configuration:

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli prepare-dataset \
  --manifest /approved/datasets/intermediate/dataset.manifest.json \
  --output /approved/reports/intermediate-preparation.json
PYTHONPATH=src .venv/bin/python -m sih26158.cli run-dataset \
  --manifest /approved/datasets/intermediate/dataset.manifest.json \
  --data-root /approved/local/storage/sih-projects \
  --data-classification INTERNAL --execute
```

5. Run the final 595–605 second workload in a fresh run directory. Do not resume or reuse stage
checkpoints for the official runtime gate:

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli prepare-dataset \
  --manifest /approved/datasets/ten-minute/dataset.manifest.json \
  --output /approved/reports/ten-minute-preparation.json
PYTHONPATH=src .venv/bin/python -m sih26158.cli run-dataset \
  --manifest /approved/datasets/ten-minute/dataset.manifest.json \
  --data-root /approved/local/storage/sih-projects \
  --data-classification INTERNAL --execute
```

Omitting `--execute` refuses to start. The run snapshots the validated dataset manifest and
preparation report. The benchmark report binds dataset/input hashes, repository revision and dirty
state, effective config/dependencies, hardware/resource choices, fresh/resumed classification,
stage/reconstruction/export timing, sparse/dense/export outcomes and the available process peak-RSS
observation. It never extrapolates a short run to ten minutes.

Inspect a run using only its declared artifacts:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/artifact-index
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/quality_report.json -o quality.json
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/benchmark_report.json -o benchmark.json
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/export_readiness.json -o exports.json
jq '{gate: .official_runtime_gate, cache: .cache_classification, timing: .timing_contract, outcomes}' benchmark.json
jq '{available_validated_formats, partial_success, formats}' exports.json
```

## Independent evaluation

Relative-distance measurements remain backend-computed and bound to the declared PLY hash through
the measurement API. Set the measurement request's `reference_id` to the matching manifest
relative-distance ID; missing IDs and conflicting values/roles are excluded. Positional evaluation
uses a separate correspondence file containing
reconstructed coordinates for manifest checkpoint IDs:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/measurements -o measurements.json
PYTHONPATH=src .venv/bin/python -m sih26158.cli evaluate-dataset \
  --manifest /approved/datasets/ten-minute/dataset.manifest.json \
  --measurements measurements.json \
  --checkpoints /approved/datasets/ten-minute/reconstructed-checkpoints.json \
  --output dataset_evaluation_report.json
```

Scale controls are excluded from independent statistics. A transform fitted with any held-out
checkpoint is rejected. Same-frame metric coordinates can be compared directly; latitude/longitude
degrees, local ENU and reconstruction-local coordinates cannot be mixed without a declared
reference-to-reconstruction transform. Unknown or mismatched altitude references leave vertical
and 3D results unavailable while compatible horizontal evaluation remains separate.

The report provides signed X/Y/Z residuals plus horizontal, vertical and 3D RMSE, median, p95 and
maximum error, count and units. Shape-aligned evaluation is explicitly separate and is not silently
performed. Missing reference data yields `UNVERIFIED`, not a favorable result.

## Comparable benchmark reports

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli compare-benchmarks \
  --report /approved/reports/run-a/benchmark_report.json \
  --report /approved/reports/run-b/benchmark_report.json \
  --output benchmark_comparison.json
```

Only identical video/telemetry hashes are comparable. Configuration differences are listed. A
fresh run and a resumed/cached run are explicitly incompatible for runtime comparison. Registration
percentage remains temporal/connection evidence, not surface completeness, and a larger point/face
count is not accuracy evidence.

## Remaining resource risks

PLY ingestion and GLB assembly currently use memory-resident geometry arrays; GLB may additionally
hold embedded texture bytes. The writers validate and publish atomically, but large-scene peak
memory and throughput have not been established without the target server and representative real
geometry. LAS writing is chunked where practical, yet target-server reader interoperability and
large-cloud behavior still require validation. Do not label fixture tests as real reconstruction,
accuracy, completeness, or large-scene performance evidence.
