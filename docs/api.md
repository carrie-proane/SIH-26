# API handoff

## Create a project

```bash
curl -X POST http://127.0.0.1:8000/api/projects \
  -F name='Campus facade' \
  -F video=@pass.mp4 \
  -F telemetry=@pass.csv
```

The response is the immutable project manifest, including original filenames, sizes, MIME types,
and SHA-256 checksums.

## Start a run

```bash
curl -X POST http://127.0.0.1:8000/api/projects/PROJECT_ID/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "config_version": "1.0",
    "profile": "preview",
    "matcher": "SIFT",
    "execution_mode": "COLMAP",
    "camera_model": "SIMPLE_RADIAL",
    "sequential_overlap": 10,
    "use_gpu": false
  }'
```

Poll `GET /api/runs/RUN_ID`. The response contains `stage`, `status`, `progress`, an actionable
`failure_reason`, the event history, and declared artifacts.

## Fetch artifacts

Use `GET /api/runs/RUN_ID/artifact-index`, then fetch only a returned URL. The API does not expose
directory listing and will not guess a filesystem path.

## Open the operator viewer

After the run declares a cloud, camera poses, selected frames and quality report:

```bash
curl http://127.0.0.1:8000/api/runs/RUN_ID/viewer-manifest
```

This is Arnav's stable frontend payload. Incomplete runs return HTTP 409 with the exact missing
artifact classes. The renderer uses declared artifact URLs only and never fabricates success data.

## Automatic preprocessing and optional handoff override

The normal upload route needs no server-side path. It creates decoded candidate timestamps, real
blur/exposure/redundancy scores, a contact sheet, a temporally distributed selection, normalized
telemetry and declared source-frame URLs. Only selected images are copied into COLMAP's `frames/`
input directory.

Advanced clients may send `force_include_frame_indices` and `force_exclude_frame_indices`. Indices
must exist in the scored set, cannot overlap or repeat, and must preserve at least three selected
images. Each decision is recorded in `keyframes.json`.

`preprocessing_run` remains an optional debugging override. When supplied, it must contain:

```text
handoff/
  keyframes.json
  frame_scores.csv
  normalized_telemetry.csv
  normalized_telemetry.meta.json
  frames/
    frame_000001.jpg
    ...
```

Frames must be the exact selected originals or matcher-resolution copies declared by
`keyframes.json`. Each selected keyframe must include `image_name` (or `filename`) and
`timestamp_s`. Normalized telemetry must follow
`data/schemas/normalized_telemetry.schema.md` exactly: `timestamp_s`, `lat`, `lon`, `alt_m`,
`alt_source`, `fix_quality`, and `source_row`, plus the metadata sidecar. The backend embeds the
sidecar and warnings in `ingest_report.json`. Invalid overrides are explained and safely fall back
to automatic processing of the immutable upload pair.

## Resume an interrupted or failed run

`POST /api/runs/{run_id}/resume` requeues a non-completed run. The runner validates its internal
stage checkpoint and resumes from the last checksum-valid boundary. It returns `409` when the run
is already active or completed, and `404` for an unknown run. Resuming never makes internal lock or
checkpoint files downloadable artifacts.

## Cancel an active run

```bash
curl -X POST http://127.0.0.1:8000/api/runs/RUN_ID/cancel
```

Cancellation creates an internal cross-process marker. Managed COLMAP and OpenMVS process groups
are terminated, while a Python-only stage stops at its next safe stage boundary. The terminal run
status is `CANCELLED`; declared artifacts from completed stages remain available. The marker is
never registered or downloadable. A cancelled run may later use the resume endpoint, which clears
the marker and revalidates its stage checkpoints.

## Capability snapshot and execution controls

Every run declares `server_capabilities.json`. It contains the exact tool/resource preflight used
for that run, requested and effective sparse GPU mode, available dense providers, selected dense
provider, and CPU-fallback warnings. Effective choices are also exposed on the run record as
`effective_sparse_gpu` and `selected_dense_provider`.

Run configuration accepts `sparse_timeout_s`, `dense_timeout_s`, and `command_heartbeat_s`.
Timeouts apply across the corresponding managed external-command stage rather than restarting for
each command.
