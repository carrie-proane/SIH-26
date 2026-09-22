# Synthetic planar removed-patch fixture

`planar_ring.ply` is a generated, untextured triangle mesh with 16 vertices and 16 faces.
It is an eight-sector annulus on Z=0: inner radius 0.2, outer radius 1.0, nominal local metres.
The missing inner octagon is a known removed patch. The outer loop represents a scene boundary.
Vertex IDs 0–7 are the inner loop; 8–15 the outer loop. Coordinates and winding are deterministic.

Tests create explicitly synthetic alignment and source-image placeholders around this fixture.
Those placeholders are not real observation or operator-review evidence. The fixture proves
rejection/generation/provenance behavior only, not real-scene quality, completeness or accuracy.
