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
    "use_gpu": false,
    "preprocessing_run": "/absolute/path/to/yosha-handoff"
  }'
```

Poll `GET /api/runs/RUN_ID`. The response contains `stage`, `status`, `progress`, an actionable
`failure_reason`, the event history, and declared artifacts.

## Fetch artifacts

Use `GET /api/runs/RUN_ID/artifact-index`, then fetch only a returned URL. The API does not expose
directory listing and will not guess a filesystem path.

## Preprocessing handoff contract

The configured directory must contain:

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
`alt_source`, `fix_quality`, and `source_row`, plus the metadata sidecar. Jay embeds the sidecar and
its warnings in `ingest_report.json`. Masks can be added beside this handoff once Yosha's interface
is merged.
