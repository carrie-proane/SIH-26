# Reproducible local and first-server validation

Services bind to loopback by default. Do not expose the unauthenticated API publicly.

For versioned capture intake, prep-only validation, the ordered tiny/intermediate/fresh-full run
sequence, independent positional checkpoints and comparable-run reports, follow
[`real-dataset-validation.md`](real-dataset-validation.md). Preparation intentionally works before
COLMAP/GPU access is available.

## Environment and offline preparation

Use Python 3.11+, the pinned ranges in `pyproject.toml`, and the exact frontend lockfile. Prepare
FFmpeg/ffprobe, COLMAP, and optional OpenMVS packages before an offline server allocation. Do not
download weights during a run. Estimate storage from the source video plus candidates, sparse
attempt workspaces, and (when enabled) several times the selected-image size for dense products.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
npm --prefix frontend ci
make verify
```

## Read-only preflight

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli doctor \
  --video /approved/input.mp4 \
  --telemetry /approved/input.srt \
  --minimum-free-disk-gb 50 \
  --output environment_report.json
```

This does not reconstruct. It separates GPU visibility, the installed COLMAP build's CUDA support,
sparse command flags, dense providers, CPU affinity/scheduler allocation, memory, disk, versions,
and input readability. `CUDA_VISIBLE_DEVICES` and Slurm CPU allocation are recorded and respected.
GPU diagnostics query only the CUDA-visible device selection, and explicit worker counts are
clamped to the scheduler/affinity limit.

## Opt-in small actual execution probe

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli execution-probe \
  --output execution_probe_cpu.json

PYTHONPATH=src .venv/bin/python -m sih26158.cli execution-probe \
  --use-gpu --output execution_probe_gpu.json
```

The probe executes bounded COLMAP feature extraction and matching on generated images. It is not a
reconstruction, accuracy, completeness, or official runtime result.

## Start the API

```bash
export SIH_DATA_ROOT=/approved/local/storage/sih-projects
export SIH_HEAVY_JOB_LIMIT=1
PYTHONPATH=src .venv/bin/python -m uvicorn sih26158.app:app \
  --host 127.0.0.1 --port 8000
```

No separate worker is required. The API runner uses per-run filesystem locks and a separate
cross-run heavy-job slot. Kernel locks recover after a process crash; a surviving managed child PID
marker conservatively prevents a duplicate reconstruction.

For approved remote access, tunnel rather than binding publicly:

```bash
ssh -L 8000:127.0.0.1:8000 USER@SERVER
```

## Tiny real CPU integration

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli run \
  --video /approved/short.mp4 --telemetry /approved/short.srt \
  --video-origin REAL --telemetry-origin REAL \
  --profile smoke --matching-strategy SEQUENTIAL \
  --max-candidate-frames 40 --max-selected-frames 24 \
  --processing-max-image-dimension 1280 --worker-threads 4 \
  --sparse-timeout 1800
```

Short-clip timings are smoke evidence only and must not be presented as the ten-minute result.

## Representative ten-minute benchmark

Run a full, user-supplied 600-second capture without prior checkpoints. Preserve the emitted run ID.

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli run \
  --video /approved/ten-minute.mp4 --telemetry /approved/ten-minute.srt \
  --video-origin REAL --telemetry-origin REAL \
  --profile accurate --matching-strategy SEQUENTIAL --sequential-overlap 15 \
  --use-gpu --dense --dense-provider auto \
  --max-candidate-frames 480 --max-selected-frames 180 \
  --processing-max-image-dimension 3840 --worker-threads 8 \
  --sparse-timeout 14400 --dense-timeout 28800 --coverage-interval 60
```

`benchmark_report.json` declares timing boundaries, input duration/resolution, cold/warm status,
stage times, selected frames, backends, source revision/dirty state, tool versions, and artifact
hashes. Only a non-synthetic fresh 595-605 second run can pass its strict `<900 s` gate.
Short-clip timing is never extrapolated. `run-dataset --execute` is the preferred server command
when an approved versioned dataset bundle is available because it snapshots the dataset/preparation
contract and binds the manifest hash into the run configuration.

## Cancellation and resume

```bash
curl -X POST http://127.0.0.1:8000/api/runs/RUN_ID/cancel
curl http://127.0.0.1:8000/api/runs/RUN_ID
curl -X POST http://127.0.0.1:8000/api/runs/RUN_ID/resume
curl http://127.0.0.1:8000/api/runs/RUN_ID/readiness
```

Inspect reports through the declared artifact endpoints:

```bash
curl http://127.0.0.1:8000/api/runs/RUN_ID/artifact-index
curl http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/quality_report.json
curl http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/server_capabilities.json
curl http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/benchmark_report.json
curl http://127.0.0.1:8000/api/runs/RUN_ID/artifacts/export_readiness.json
```

## Scheduler integration information required

Before creating a Slurm script, obtain the institute partition/QOS, wall-time limit, CPU and memory
syntax, GPU generic-resource syntax, local scratch path, module/container policy, network policy, and
job-step rules. Do not bypass allocation or assume a GPU/VRAM/driver. Set worker threads and
`SIH_HEAVY_JOB_LIMIT` within the assigned resources; leave `CUDA_VISIBLE_DEVICES` unchanged.
