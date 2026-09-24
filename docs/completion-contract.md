# Integrated bounded completion and surface-support contract, version 1.0

Status: **INTEGRATED_WITH_CONSERVATIVE_LIMITS**. The pipeline, API, viewer manifest, operator gap
review and exporters share this contract. Existing run statuses and confidence enums remain
unchanged. Fixture tests establish contract behavior only; they do not establish real completion
accuracy.

## Inputs and lifecycle

`completion.complete_planar_gaps(source_root, output_dir, source, parameters=..., reviews=...)`
requires a `MeshSource` with source run ID, declared PLY path/hash, observed-reconstruction provenance,
and an explicit, verified local ENU metre coordinate contract. The alignment evidence is itself a
hash-bound relative asset. All referenced paths must remain inside `source_root` after symlink
resolution. Missing meshes, checksum mismatches and sparse point clouds return `UNAVAILABLE`.
A declaration is an integration trust boundary: Jay must construct it from the artifact index and
validated alignment, never from arbitrary user assertions or a filename.

The original mesh is loaded without processing, reordering or welding and checked again before
output. Published completion attempts use source/request/dependency fingerprints. Every decision
has an immutable versioned status; the current status is a pointer, so a later refusal does not
erase an earlier successful attempt. Original geometry is never overwritten.

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
| `coverage_report.json` | Integrated topology candidates, boundary ENU coordinates, selected-frame evidence options and optional metric-depth support |

The score records plane residual relative to the configured threshold. It is a rule-based score,
not a calibrated probability. The viewer can map inferred geometry to the existing
`AI_ASSISTED_NOT_MEASURABLE` label. RGB values are never interpreted as confidence. Any later
simplification, reordering or conversion must preserve/remap face provenance or refuse export.
OBJ/GLB completed selection therefore publishes observed and inferred geometry as separate
packages, so an independent reader need not trust face-index stability after conversion.

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

## Integrated lifecycle and deferred work

- Completion cache inputs include source run/artifact and alignment hashes, coordinate contract,
  every threshold, code/schema and NumPy/trimesh versions, source-image hashes and review decisions.
- Post-run completion shares the global heavy-job limit, records separate operation timing and
  invalidates the current export-readiness pointer when the selected attempt changes. Previous
  export/completion evidence remains declared and traceable. An identical fingerprint is idempotent.
- Relative monocular depth is rejected. Metric support requires a declared
  `coverage_depth_views.json`, raw-video hash, processed-grid intrinsics, rigid poses and hash-bound
  camera-Z metre arrays. Topology remains available when this evidence is missing or rejected; no
  completeness percentage or reference denominator is invented.

Symmetry remains an explicit refusal, and terrain interpolation and learned depth fusion are deferred: no stable real observed mesh,
real gap references, symmetric building or asymmetric counterexample was supplied. Synthetic patch
checks establish behavior only. A real segmentation A/B and target-server performance are still
required before promoting completion into the runtime.
