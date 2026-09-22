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

The operator catalog is server-authoritative and supports refresh/deep-link recovery:

```bash
curl -f http://127.0.0.1:8000/api/projects
curl -f http://127.0.0.1:8000/api/projects/PROJECT_ID/runs
```

The first response contains `projects`; the second contains newest-first `runs`. A run's
`derived_from_run_id` distinguishes a linked rerun from an original run. The browser stores only
the selected IDs in `?project=...&run=...`, then refetches these endpoints.

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

Poll `GET /api/runs/RUN_ID`. The response contains `stage`, `status`, backend-reported `progress`, an actionable
`failure_reason`, the event history, and declared artifacts.

Only `SIFT` is currently executable. `SUPERPOINT_LIGHTGLUE` remains a proposed experiment and is
rejected with HTTP 422; no SIFT execution is relabelled as learned matching. Additive run fields
`requested_matcher` and `executed_matcher` keep configuration and actual execution distinct.

The following run fields are additive and safe for older clients to ignore:
`processing_started_at`, `processing_completed_at`, `stage_timings_s`, and
`derived_from_run_id`. For example:

```json
{
  "processing_started_at": "2026-09-20T12:00:00+00:00",
  "processing_completed_at": "2026-09-20T12:07:31+00:00",
  "stage_timings_s": {"INGEST": 0.8, "PREPROCESS": 42.1, "SPARSE": 341.0, "REPORT": 0.4},
  "derived_from_run_id": null
}
```

## Fetch artifacts

Use `GET /api/runs/RUN_ID/artifact-index`, then fetch only a returned URL. The API does not expose
directory listing and will not guess a filesystem path.

## Open the operator viewer

As soon as the run declares a cloud, camera poses and selected frames:

```bash
curl http://127.0.0.1:8000/api/runs/RUN_ID/viewer-manifest
```

This is the operator frontend payload. `preview_mode: SPARSE_EARLY` and a null
`quality_report_url` explicitly mean that observed sparse evidence is viewable while final quality
reporting is pending. Incomplete runs return HTTP 409 only when a minimum viewer artifact class is
missing. The renderer uses declared artifact URLs only and never fabricates success data. Optional
`completion.status: NOT_RUN` keeps future inferred geometry separate without changing the existing
confidence enum.

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

Additive resource fields are `max_candidate_frames`, `max_selected_frames`,
`processing_max_image_dimension`, `worker_threads`, and `coverage_interval_s`. An explicit thread
request is clamped to the process CPU affinity and `SLURM_CPUS_PER_TASK`; the capability and
benchmark reports expose both requested and effective values. Supplied intrinsics
also declare `camera_params_reference` as `PROCESSED_FRAMES` or `SOURCE_VIDEO`; source-video
intrinsics are rejected when an unmodelled rotate/resize/crop would make them incompatible.

## Sparse readiness before optional dense completion

`GET /api/runs/{run_id}/readiness` reports `sparse_preview_ready`, missing sparse artifacts, dense
status, the only measurement geometry, and the export report/status once reporting has completed.
It is safe to poll during reconstruction and does not promote dense or inferred geometry to
measurement evidence.

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/readiness
```

```json
{
  "schema_version": "1.0",
  "run_id": "run_example",
  "stage": "RECONSTRUCTING",
  "status": "RECONSTRUCTING",
  "sparse_preview_ready": true,
  "sparse_preview_missing": [],
  "quality_report_ready": false,
  "dense_status": "NOT_READY_OR_NOT_REQUESTED",
  "dense_visual_artifacts": [],
  "measurement_geometry": "sparse/sparse_local.ply",
  "inferred_geometry_measurement_eligible": false,
  "export_report_ready": false,
  "export_report_url": null,
  "export_status": {}
}
```

`sparse_preview_ready: true` means the observed local PLY, camera path and selected-frame contract
are declared. It does not mean dense reconstruction, mesh export, survey-grade geolocation, or all
requested formats are ready. `dense_status: FAILED_OR_UNAVAILABLE` can coexist with a valid sparse
preview and LAS export. Once `export_report_ready` becomes true, use `export_report_url`; do not
guess export paths.

Each keyframe may also expose additive `reconstruction_status` and
`reconstruction_exclusion_reason` fields so the UI can distinguish unselected frames, registered
frames in the delivered component, and selected frames that COLMAP did not register.

## Immutable configuration changes

Completed runs cannot be edited. `POST /api/runs/{run_id}/rerun` accepts a complete `RunConfig`,
creates a new run for the same project, and sets `derived_from_run_id` to the completed evidence run.
The response is HTTP 202 and contains a new `run_id`; the source run and its artifacts remain
unchanged. This accepted request uses API defaults for fields not shown:

```bash
curl -f -X POST http://127.0.0.1:8000/api/runs/RUN_ID/rerun \
  -H 'Content-Type: application/json' \
  -d '{
    "config_version": "1.0",
    "profile": "preview",
    "matcher": "SIFT",
    "execution_mode": "COLMAP",
    "camera_model": "SIMPLE_RADIAL",
    "sequential_overlap": 10,
    "use_gpu": false,
    "enable_dense_reconstruction": true,
    "dense_provider": "auto"
  }'
```

The endpoint returns 409 unless the source run is completed; it never mutates or resumes the source
run.

## Geometry export availability and download

Newly reported runs create exports automatically. For a completed, failed, or cancelled run that
predates the exporter (or to revalidate its current declared artifacts), create/reuse export
packages without reconstruction:

```bash
curl -f -X POST http://127.0.0.1:8000/api/runs/RUN_ID/exports -o exports.json
```

The response is HTTP 201. The operation never invokes COLMAP/OpenMVS, accepts only terminal runs,
uses source/dependency hashes plus options/coordinate metadata for cache reuse, and leaves the run
status and measurement eligibility unchanged. An active run returns 409.

After `export_report_ready` is true, fetch the same declared export contract read-only:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/exports -o exports.json
jq '.formats | with_entries(.value |= {status, reason, failures, exports})' exports.json
```

Each format has an independent status. `AVAILABLE_VALIDATED` means at least one export package was
written atomically and reopened successfully. It may still contain `failures` for other source
artifacts. `UNAVAILABLE`, `INVALID`, and `INVALID_OR_UNSUPPORTED` include an actionable `reason`.
The top-level `partial_success` is expected when, for example, an observed point cloud supports PLY
and LAS but no mesh exists for OBJ/GLB, or when out-of-scope GeoTIFF/FBX remain unavailable.

Selected fields from a successful mesh-capable response are:

```json
{
  "schema_version": "2.0",
  "run_id": "run_example",
  "exporter": {"name": "sih26158.geometry_exports", "version": "1.0"},
  "available_validated_formats": ["PLY", "OBJ", "GLB_GLTF", "LAS"],
  "partial_success": true,
  "formats": {
    "OBJ": {
      "status": "AVAILABLE_VALIDATED",
      "failures": [],
      "exports": [{
        "format": "OBJ",
        "variant": "visual-mesh",
        "geometry_provenance": "DERIVED_OBSERVED_VISUAL",
        "measurement_eligible": false,
        "measurement_api_supported": false,
        "files": [
          {"relative_path": "exports/obj/visual-mesh/HASH/model.obj", "url": "/api/runs/run_example/artifacts/exports/obj/visual-mesh/HASH/model.obj"},
          {"relative_path": "exports/obj/visual-mesh/HASH/material.mtl", "url": "/api/runs/run_example/artifacts/exports/obj/visual-mesh/HASH/material.mtl"},
          {"relative_path": "exports/obj/visual-mesh/HASH/textures/atlas.png", "url": "/api/runs/run_example/artifacts/exports/obj/visual-mesh/HASH/textures/atlas.png"}
        ],
        "manifest_url": "/api/runs/run_example/artifacts/exports/obj/visual-mesh/HASH/export_manifest.json"
      }]
    },
    "GLB_GLTF": {
      "status": "AVAILABLE_VALIDATED",
      "exports": [{
        "format": "GLB",
        "geometry_provenance": "DERIVED_OBSERVED_VISUAL",
        "measurement_eligible": false,
        "files": [
          {"relative_path": "exports/glb/visual-mesh/HASH/model.glb", "url": "/api/runs/run_example/artifacts/exports/glb/visual-mesh/HASH/model.glb"},
          {"relative_path": "exports/glb/visual-mesh/HASH/export_manifest.json", "url": "/api/runs/run_example/artifacts/exports/glb/visual-mesh/HASH/export_manifest.json"}
        ]
      }]
    },
    "LAS": {
      "status": "AVAILABLE_VALIDATED",
      "exports": [{
        "format": "LAS",
        "geometry_provenance": "OBSERVED",
        "measurement_eligible": true,
        "measurement_api_supported": false,
        "files": [
          {"relative_path": "exports/las/observed-sparse/HASH/cloud.las", "url": "/api/runs/run_example/artifacts/exports/las/observed-sparse/HASH/cloud.las"},
          {"relative_path": "exports/las/observed-sparse/HASH/provenance.json", "url": "/api/runs/run_example/artifacts/exports/las/observed-sparse/HASH/provenance.json"},
          {"relative_path": "exports/las/observed-sparse/HASH/export_manifest.json", "url": "/api/runs/run_example/artifacts/exports/las/observed-sparse/HASH/export_manifest.json"}
        ]
      }]
    },
    "GEOTIFF": {"status": "UNAVAILABLE", "reason": "Out of scope: no defensible georeferenced DSM or orthomosaic raster product exists for this run..."}
  }
}
```

`GET /api/runs/RUN_ID/exports` also adds a declared `bundle_url` to each available OBJ package.
Use it to obtain the OBJ, MTL, textures and provenance manifest together without reconstructing a
server filesystem path:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/exports -o exports.json
OBJ_BUNDLE_URL=$(jq -r '.formats.OBJ.exports[0].bundle_url' exports.json)
curl -f "http://127.0.0.1:8000${OBJ_BUNDLE_URL}" -o model-obj-package.zip
```

PLY is downloaded through `viewer-manifest.cloud.url`; LAS/OBJ/GLB files, sidecars and manifests
come only from their declared `url` fields. GeoTIFF and FBX have no download URL while unavailable.

The real response also includes options, coordinate contracts, validation, checksums, sizes and
media types. Every item in `files` is in the run's declared artifact index. Download a
self-contained GLB or LAS like this:

```bash
GLB_URL=$(jq -r '.formats.GLB_GLTF.exports[0].files[] | select(.relative_path | endswith("model.glb")) | .url' exports.json)
LAS_URL=$(jq -r '.formats.LAS.exports[0].files[] | select(.relative_path | endswith("cloud.las")) | .url' exports.json)
curl -f "http://127.0.0.1:8000${GLB_URL}" -o model.glb
curl -f "http://127.0.0.1:8000${LAS_URL}" -o cloud.las
```

OBJ is a package, not one standalone file. Download every listed file while retaining the relative
layout between `model.obj`, `material.mtl`, and `textures/...`; `map_Kd` uses that relative path.
For example:

```bash
mkdir -p export-download
jq -r '.formats.OBJ.exports[0].files[] | [.url, .relative_path] | @tsv' exports.json |
while IFS=$'\t' read -r url relative; do
  mkdir -p "export-download/$(dirname "$relative")"
  curl -f "http://127.0.0.1:8000${url}" -o "export-download/${relative}"
done
```

The package `export_manifest.json` records the source artifact/hash, exporter/version, options,
coordinate transform, provenance, validation, and generated-file hashes. LAS additionally carries
a `SIH26158` VLR that hash-binds `provenance.json`.

Coordinate rules are explicit and reversible:

- OBJ and LAS keep local ENU metres unchanged: X=east, Y=north, Z=up.
- GLB applies `(east, north, up) -> (east, up, -north)`, so glTF X=east, Y=up and Z=south. Both
  4x4 transforms are recorded in the manifest.
- LAS deliberately has no EPSG code because local ENU is not a projected CRS. Origin WGS84,
  altitude-reference declarations, units and local-transform hash are carried as metadata.
- None of the formats improves or independently validates scale, absolute geolocation, or vertical
  datum. Large scenes have not yet been validated on the college server.

Only declared meshes with faces produce OBJ/GLB. A point-only run reports mesh export unavailable
and is never silently triangulated. An explicitly untextured mesh may produce a valid untextured
OBJ/GLB. Observed, derived-observed visual, and inferred sources remain distinct in each export.
OBJ text and LAS point records are streamed while writing, but the current PLY reader and GLB
encoder still materialize geometry in memory. Multi-atlas PLY texturing and LAS classification
outside the standard 0-31 range fail explicitly rather than dropping data.

## Traceable multiple measurements

`POST /api/runs/{run_id}/measurements` accepts a declared geometry path and SHA-256, two finite
endpoints (with optional point/face IDs), coordinate frame, units, measurement kind, and optional
reference metadata. The backend verifies ownership/hash/reference bounds, computes distance, and
derives provenance/eligibility; there is no client `verified` flag.

```bash
curl -f -X POST http://127.0.0.1:8000/api/runs/RUN_ID/measurements \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "geometry_artifact_path": "sparse/sparse_local.ply",
  "geometry_artifact_sha256": "64_HEX_CHARACTERS",
  "start": {"coordinates": [0, 0, 0], "point_id": 10},
  "end": {"coordinates": [3, 4, 0], "point_id": 25},
  "coordinate_frame": "LOCAL_ENU_METRES",
  "units": "m",
  "measurement_kind": "RELATIVE_DIMENSION",
  "reference_id": "facade-width-01",
  "reference_value_m": 5.1,
  "reference_method": "laser rangefinder",
  "reference_evidence": "field-note-07",
  "reference_role": "HELD_OUT_EVALUATION"
}
JSON
```

The request body is:

```json
{
  "geometry_artifact_path": "sparse/sparse_local.ply",
  "geometry_artifact_sha256": "64_HEX_CHARACTERS",
  "start": {"coordinates": [0, 0, 0], "point_id": 10},
  "end": {"coordinates": [3, 4, 0], "point_id": 25},
  "coordinate_frame": "LOCAL_ENU_METRES",
  "units": "m",
  "measurement_kind": "RELATIVE_DIMENSION",
  "reference_id": "facade-width-01",
  "reference_value_m": 5.1,
  "reference_method": "laser rangefinder",
  "reference_evidence": "field-note-07",
  "reference_role": "HELD_OUT_EVALUATION"
}
```

`GET /api/runs/{run_id}/measurements` returns every record plus sample count, median, RMSE, p95,
maximum, and relative errors. `SCALE_CONTROL` references are excluded from independent evaluation.
For dataset-level evaluation, `reference_id` must match the hash-bound dataset manifest entry; an
unbound measurement or a conflicting value/role is reported and excluded from dataset statistics.
Fetch it with:

```bash
curl -f http://127.0.0.1:8000/api/runs/RUN_ID/measurements
```

Distances use the submitted `measurement_kind`: Euclidean 3D, XY horizontal, absolute Z vertical,
or Euclidean relative dimension. Coordinates must be `LOCAL_ENU_METRES` and units `m` for eligible
evidence. The current measurement endpoint accepts declared PLY only, so successful OBJ, GLB or LAS
export never makes a visual or inferred artifact measurement-eligible. Only a real, observed
`sparse/sparse_local.ply` with its exact declared SHA-256 can be eligible; the response always
includes the backend-computed distance, provenance, eligibility boolean and reason.

The legacy `known_distance_m`/`measured_distance_m` pair remains in the quality report and is now
explicitly labelled a manual, role-ambiguous input rather than independent accuracy evidence.
