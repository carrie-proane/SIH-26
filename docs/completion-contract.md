# Proposed completion and surface-support contract, version 1.0

Status: **PROPOSED_NOT_INTEGRATED**. Yosha supplies these algorithms and schemas. Jay owns
orchestration, configuration, resources, lifecycle, artifact publication, exporters and all frontend
work. No agreement with Jay is implied by this implementation. Existing run statuses and confidence
enums are unchanged. The modules do not modify `pipeline.py`, `viewer_manifest.py` or `exports.py`.

## Inputs and lifecycle

`completion.complete_planar_gaps(source_root, output_dir, source, parameters=..., reviews=...)`
requires a `MeshSource` with source run ID, declared PLY path/hash, observed-reconstruction provenance,
and an explicit, verified local ENU metre coordinate contract. The alignment evidence is itself a
hash-bound relative asset. All referenced paths must remain inside `source_root` after symlink
resolution. Missing meshes, checksum mismatches and sparse point clouds return `UNAVAILABLE`.
A declaration is an integration trust boundary: Jay must construct it from the artifact index and
validated alignment, never from arbitrary user assertions or a filename.

`output_dir` must be new and outside the immutable source run, including through symlinks. The
original mesh is loaded with the foundation's PLY reader, without processing, reordering or welding.
It is checked again before output. This creates a standalone derived bundle, not a published run.
Jay must choose a linked run or approved derived-artifact lifecycle before publication.

## Bounded planar algorithm

Directed mesh edges identify closed boundary loops. Non-manifold edges, inconsistent winding and
open/touching chains are unavailable. Every candidate is checked against recorded thresholds:

| Parameter | Default |
|---|---:|
| Maximum gap area | 0.25 m² |
| Maximum diameter / boundary edge | 1.0 m / 0.5 m |
| Maximum support-plane residual | 0.01 m |
| Maximum neighboring normal disagreement | 10° |
| Minimum adjacent supporting faces | 6 |
| Minimum adjacent support area / gap area | 2 |
| Maximum loop vertices / regions | 64 / 64 |
| Maximum source faces / bytes | 200,000 / 50,000,000 |

The prototype supports strictly convex planar loops. Concave, degenerate or self-crossing loops,
large openings, sharp corners, insufficient support, existing surface inside a boundary and nested
surface islands are rejected. These limits are conservative implementation defaults, not thresholds
validated against real reconstruction data. Work is bounded by mesh/region limits; hard process
cancellation remains a caller responsibility under Jay's existing executor.

Even a geometrically eligible loop is **not** evidence of an occlusion. Each accepted region needs
an explicit `GapReview` with `CONFIRMED_SMALL_GAP`, reviewer, explanation and hash-verified source
image references. `UNKNOWN`, `STRUCTURAL_OPENING`, missing reviews or missing image evidence never
produce faces. The module verifies evidence integrity, not image semantics. A human must establish
that the image corresponds to this boundary and rules out a doorway/window/scene edge. This human
assertion remains an assumption in provenance.

Boundary IDs hash the sorted source vertex IDs and are meaningful only with the source artifact
hash. Use an initial review-free call to obtain candidates, then a **new** output directory with
reviews. Triangulation reverses boundary winding and uses only original boundary coordinates.
Generated PLY uses the existing trimesh writer in deterministic binary order. Reopening must
preserve exact vertex coordinates and face order; unsupported coordinate precision fails closed.
No ENU-to-GLB transform, texture transfer or observed-face replacement happens here.

## Outputs

| Relative path | Contract |
|---|---|
| `completion/report.json` | Always: schema/status/reason, source, parameters, timing, counts, accepted/rejected regions, warnings, `measurement_eligible=false` |
| `completion/generated_mesh.ply` | Only on `COMPLETED`: generated faces only, original ENU coordinate convention |
| `completion/provenance.json` | Source run and hash, output hash, exhaustive zero-based generated face IDs and region mappings, source vertices/support faces/review images, assumptions, `geometry_source=AI_INFERRED`, `measurable=false` |
| `coverage/report.json` | Caller serializes `CoverageReport` with existing `atomic_json`; no automatic pipeline publication |

The score records plane residual relative to the configured threshold. It is a rule-based score,
not a calibrated probability. The viewer can map inferred geometry to the existing
`AI_ASSISTED_NOT_MEASURABLE` label. RGB values are never interpreted as confidence. Any later
simplification, reordering or conversion must preserve/remap face provenance or refuse export.

Machine-readable proposed schemas are in [schemas/](schemas/): completion source, gap review,
report, provenance and coverage report. The Pydantic models in `completion_contract.py` and
`coverage.py` are the schema sources.

## Recover observed support before infill

`coverage.analyze_surface_support` consumes the same verified observed mesh and `DepthView` records.
Each view declares its frame ID, same-source raw-video hash, rigid world-to-camera pose, rectified
intrinsics and hash-bound **camera-Z metric depth**. Jay must bind these to declared artifacts and
verify video/frame association. Relative monocular depth is rejected. Intrinsics and depth must
already share the processed pixel grid, crop and orientation; distorted imagery is unsupported.
The function projects face centroids and checks measured depth:

- A nearer measured depth is occlusion, not observed support.
- Missing/invalid depth is unknown; frustum inclusion alone gives no support.
- Depth inconsistent with a surface on the far side is reported separately.
- Greedy proposals favor additional same-video frames with new depth-consistent samples, avoiding
  redundant views. These are proposals for existing extraction/selection, not a second runner.

Reports retain available sample statistics, poses, intrinsics, hashes and thresholds. All surface
areas, percentages and the denominator remain `null` with a reason: centroid support on a partial
reconstruction is not a reference domain for visible-scene completeness. Inferred patches never
enter this calculation. Existing temporal coverage diagnostics remain untouched.

## Integration dependencies to freeze with Jay

1. PREPROCESS must hash `segmentation_fingerprint_inputs(model_path, settings)`, including resolved
   environment weights, checkpoint bytes, versions and every setting; bump the stage contract.
   The existing fingerprint still hashes only the explicit model path. Until integrated, use fresh
   runs for segmentation comparisons and never claim environment-weight cache invalidation works.
2. Plumb the segmentation settings/cancellation callback and existing scheduler allocation. CPU is
   the default. Accelerator use requires explicit caller authorization after resource allocation.
   Cooperative checks cannot interrupt a blocked model call; use the existing managed process
   executor for hard timeouts. Do not introduce another scheduler or runner.
3. Completion cache inputs: source run/artifact and alignment hashes, coordinate contract, every
   threshold, code/schema version, numpy/trimesh versions, source-image hashes and review decisions.
   Coverage adds video/frame association, depth hashes, poses/intrinsics, depth semantics,
   selected-frame flags, depth tolerance and proposal limits.
4. Add all returned segmentation review artifacts to the existing artifact list (the current call
   already registers returned paths). Add new completion/coverage schemas only after agreement.
   Include enabled inference/completion and publication work in end-to-end timing.
5. `RunConfig` currently accepts `SUPERPOINT_LIGHTGLUE` without an implemented execution path.
   Jay must reject it or report requested/executed matchers separately. This delivery uses SIFT.
6. FFprobe sees a -180° display-matrix rotation in DJI_0574 while preparation reports rotation null.
   Existing extraction correctly records/applies 180°; the preparation metadata needs reconciliation.
   The resulting preview also looks sideways and needs capture/orientation review before calibration.

Symmetry, terrain interpolation and learned depth fusion are deferred: no stable real observed mesh,
real gap references, symmetric building or asymmetric counterexample was supplied. Synthetic patch
checks establish behavior only. A real segmentation A/B and target-server performance are still
required before promoting completion into the runtime.
