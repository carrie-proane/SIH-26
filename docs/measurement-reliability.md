# Measurement reliability

Processing completion and evidence acceptance are separate. A failed evaluated gate takes
precedence over missing validation; otherwise missing validation gives `NOT_VALIDATED`.
Registration requires at least 80%; median reprojection error must be at most 1.5 pixels.

CSV altitude references come from an explicit `altitude_reference` column or unambiguous
headers (`rel_alt`, `height_above_takeoff`, `altitude_above_sealevel`,
`absolute_ellipsoidal`). Generic `alt_m` is unknown. The inference source is recorded.
Unknown references retain best-effort local height differences and an assumption warning,
with vertical alignment `NOT_VALIDATED`. Relative and MSL heights use local tangent
horizontal coordinates and height differences, never pretend to be ellipsoidal heights.
Only explicit ellipsoidal altitude uses full geodetic ENU conversion. A configured origin's
height must use the declared telemetry reference.

The conservative mixed-reference guard rejects abrupt transitions between altitude magnitudes
<=100 m and >=300 m, with a change >200 m and vertical speed >30 m/s. This is a discontinuity
heuristic, not a datum detector: smooth climbs remain valid; suspected jumps require checking
the source log. Parser rejection returns no records and a `MIXED_ALTITUDE_REFERENCE` warning.

Trajectory identifiability uses the two largest eigenvalues of the 3D covariance of
matched telemetry inliers. Their variance ratio must be <=100 (a 10:1 standard-deviation
ratio). This is a conservative conditioning heuristic, not an accuracy guarantee. Using
the two largest eigenvalues avoids rejecting planar orbits merely because height is
constant. A line, near-line, or stationary trajectory is degenerate. The best-effort
transform is retained, but evidence cannot pass. Residuals are camera-to-telemetry
consistency metrics; matching an uncertain telemetry prior does not validate true position.

Measurements need both local geometric support and global validation. `ALLOWED` requires
high support at both endpoints, `evidence_verdict=PASSED`, and a well-conditioned alignment.
`NOT_VALIDATED` is deliberately insufficient: missing independent scale validation cannot
be treated as passing. Unknown older reports also downgrade to `CAUTION`. Numeric visual
estimates remain visible; lower-confidence and non-measurable geometry keep their rules.

Known-distance validation uses `known_distance_m`, `known_distance_reference_source`, and
`known_distance_endpoint_a` / `_b` (`point_id`, optional `description`). IDs are zero-based
PLY vertex indices, matching the confidence artifact, from this run's metric sparse cloud.
The report records endpoints, resolved coordinates, run ID, cloud checksum, and computation
time. Typed `measured_distance_m` remains accepted for compatibility but is recorded only as
`reported_input_measured_m`; it never supplies the validation result. Missing endpoints,
unsupported geometry, or missing provenance cannot pass. Error above 10% fails evidence.
This reference does not set the telemetry-fitted scale, so it is independent of scale fitting.

Frame scoring streams adjacent previews, bounded to a 1920-pixel longest dimension, while
retaining only scalar scores for global normalization. Weights, SSIM formula, score
normalization, greedy ranking, temporal spacing, and overrides are unchanged. Selected frames
are decoded again at native resolution and original orientation; rejected candidates retain
only previews. The second decode is a deliberate memory/time tradeoff.

Real-video acceptance (`tests/test_frames.py`, local `DJI_0574.MP4`, 3840x2160, 32.27 seconds)
uses 33 candidates sampled every 30 frames and target 20, retaining 17 after temporal spacing.
The tolerance is >=90% **exact** selected-frame overlap and >=20% lower peak RSS, measured in
fresh processes. At 1920 pixels, all 17 selections matched; peak RSS fell from 1,825,136 KiB
to 460,868 KiB (74.7%). Total time was 35.2s versus 40.1s, including restoring native frames.
A 1280-pixel trial reached only 76.5% exact overlap and was rejected. This validates the chosen
bound on this representative clip, not equivalence on every video. The acceptance test skips
explicitly if the private local video is absent. Reproduce with:

```sh
.venv/bin/python scripts/benchmark_frame_previews.py --video DJI_0574.MP4 --output /tmp/preview-report.json
```

The measured report is in `evidence/measurement-reliability/frame-preview-benchmark.json`.

Automatic preprocessing profiles now control the retained-frame target: `smoke` 60,
`diagnostic` 75, `preview` 100 (unchanged default), `balanced` 110, and `accurate` 120.
These are selection budgets, capped by candidate availability and subject to the existing
spacing and explicit overrides. The profile and effective target are recorded in telemetry
preprocessing metadata. A validated supplied preprocessing handoff retains its own selection;
the synthetic demo retains its explicitly labeled fixture. Profiles make no change to matcher
choice, scoring weights, sparse-model selection, or dense/measurement separation.

Verification history (the full backend, UI-unit, and browser suites were run after each
numbered fix; this table records phase completion):

| Milestone | Backend tests | UI tests | Browser tests |
| --- | ---: | ---: | ---: |
| Baseline | 73 | 15 | 2 |
| Phase 1 | 81 | 16 | 2 |
| Phase 2 | 89 | 18 | 2 |
| Phase 3 | 93 | 18 | 2 |
| Phase 4 | 97 | 18 | 2 |
| Phase 5 | 98 | 18 | 2 |

Final lint and the frontend production build pass. Backend tests ran outside the execution
sandbox because its restrictions stalled TestClient. Browser tests used a temporary config
changing the existing macOS `/private/tmp` data path to `/tmp`; the original browser config
was preserved. The existing Starlette deprecation and frontend chunk-size warnings remain.
The requested `contract_1_.md` was absent; the repository's four-section
`docs/scientific-integrity-contract.md` and the audit's explicit preservation rules were used.
