# Jay backend architecture

The backend implements the contract's narrow first-round path:

```text
POST video + telemetry
        |
        v
immutable project/input + manifest.json + SHA-256
        |
        v
QUEUED -> INGESTING -> PREPROCESSING -> RECONSTRUCTING -> REPORTING
        |                                                    |
        +---------------------> FAILED <---------------------+
                                                             |
                                                             v
                                                         COMPLETED
                                                             |
                                                             v
declared artifact index -> quality report / PLY / camera poses / viewer payload
```

## Ownership boundary

Jay owns the project/run API, run state, ffprobe report, COLMAP invocation, alignment utilities,
matcher decision, quality report, known-distance evidence, and end-to-end command. Yosha's
preprocessor hands off `keyframes.json`, `frame_scores.csv`, `normalized_telemetry.csv`, its
`.meta.json` sidecar, and `frames/`. Arnav owns the final
`viewer-manifest` endpoint and browser implementation; `examples/viewer-manifest.json` freezes the
payload Jay publishes for UI development.

## Invariants

- Raw inputs are copied once, checksummed, and never edited.
- Completed artifacts are accessed only through the run's declared index.
- Paths containing `..`, absolute paths, and undeclared files are rejected.
- The only states are `QUEUED`, `INGESTING`, `PREPROCESSING`, `RECONSTRUCTING`, `REPORTING`,
  `COMPLETED`, and `FAILED`.
- A failed run retains its manifest, events, logs, and every prior declared artifact.
- `SYNTHETIC_DEMO` is labeled in every artifact and cannot be mistaken for genuine geometry.
- A real run never falls back to synthetic geometry.

## Local coordinate alignment

`geo.py` converts WGS84 latitude/longitude/ellipsoidal height to ECEF and then local ENU metres.
It implements a 7-DoF Umeyama similarity fit and robust MAD outlier rejection. Absolute horizontal,
vertical, relative, and scale errors remain separate report concepts.
