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
