# Surface-completion runtime contract

## Purpose

`sih26158.reconstruction.surface_completion` connects the evidence pipeline to a separately trained
AI inference runtime. Isolation keeps heavyweight ML dependencies out of the FastAPI worker and
makes the boundary testable. Nothing is downloaded automatically.

## Configuration

- Set `SIH_SURFACE_COMPLETION_BIN` to an executable path or command on `PATH`.
- Set `surface_completion_model_path` in `RunConfig` to local weights.
- Set `enable_surface_completion=true`.
- Choose `surface_completion_samples` from 1 to 8; three is the default because disagreement across
  plausible samples is useful uncertainty evidence.

The executable is called as:

```text
<binary> --request <run_dir>/completion_request.json --output-dir <run_dir>/completion
```

## Request

The versioned JSON request declares the observed geometry, geometry kind, camera poses, selected
frames, local coordinate frame, weights path, sample count, required output paths, and scientific
rules. All run-artifact input paths are relative to the run directory.

## Required output

- `completion/completed_mesh.ply` - non-empty PLY beginning with the normal `ply` header.
- `completion/uncertainty.json` - object with `schema_version`,
  `semantics=HIGHER_IS_MORE_UNCERTAIN`, and a `summary` object.

A production uncertainty file should additionally map uncertainty to mesh vertices/faces or voxel
cells, identify observed versus generated regions, summarize hypothesis disagreement, record model
and training-data versions, and state calibration error on the validation set.

## Failure behavior

Missing executable/weights produces `completion_report.json` with `BLOCKED`. Runtime or validation
failure produces `FAILED`. Neither outcome changes a valid sparse/dense reconstruction to `FAILED`.
Successful output produces `COMPLETED`, but `measurement_eligible` is always false and the browser
labels it `AI_ASSISTED_NOT_MEASURABLE`.
