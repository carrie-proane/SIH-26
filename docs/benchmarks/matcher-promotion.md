# SIFT vs SuperPoint + LightGlue decision

SIFT is the reproducible baseline and remains functional without learned models. For the same
selected-frame set, save one metrics JSON per matcher with:

```json
{
  "matcher": "SIFT",
  "eligible_frames": 100,
  "registered_frames": 91,
  "median_reprojection_error_px": 0.92,
  "p95_reprojection_error_px": 1.61,
  "runtime_s": 48.2
}
```

Then run:

```bash
sih26158 benchmark-matchers \
  --sift evidence/sift.json \
  --learned evidence/superpoint-lightglue.json \
  --output evidence/matcher_benchmark.json
```

The learned matcher is selected only if it improves registered-frame rate, or reduces median
reprojection error without reducing registration. The report records both inputs and the decision.
No learned benchmark is claimed until it is run on the same real primary sample.

Model evidence still required for a real run: hloc/SuperPoint/LightGlue revisions, checkpoint hashes,
licences, device, runtime, and source/output checksums.

