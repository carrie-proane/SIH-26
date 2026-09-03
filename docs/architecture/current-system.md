# Current backend architecture

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

## Module boundary

The API, run state, ffprobe report, COLMAP invocation, alignment utilities, matcher decision,
quality report, known-distance evidence, and end-to-end command live under the installable
`sih26158` package. Frame and telemetry modules are integrated into the ordinary upload route; a
compatible external handoff remains an optional debugging override. The reporting layer owns the
final `viewer-manifest` endpoint contract, while the React application owns browser rendering;
`examples/viewer-manifest.json` freezes the payload used for UI development.

## Invariants

- Raw inputs are copied once, checksummed, and never edited.
- Completed artifacts are accessed only through the run's declared index.
- Paths containing `..`, absolute paths, and undeclared files are rejected.
- The only states are `QUEUED`, `INGESTING`, `PREPROCESSING`, `RECONSTRUCTING`, `REPORTING`,
  `COMPLETED`, and `FAILED`.
- A failed run retains its manifest, events, logs, and every prior declared artifact.
- `SYNTHETIC_DEMO` is labeled in every artifact and cannot be mistaken for genuine geometry.
- A real run never falls back to synthetic geometry.
- Candidate images stay under `preprocessing/candidates/`; only selected images enter `frames/`.
- PLY RGB is photographic display data. Explicit confidence is exported with the PLY in the same
  deterministic point order and validated by vertex count.

## Local coordinate alignment

`geo.py` converts WGS84 latitude/longitude/ellipsoidal height to ECEF and then local ENU metres.
It implements a 7-DoF Umeyama similarity fit and robust MAD outlier rejection. Absolute horizontal,
vertical, relative, and scale errors remain separate report concepts.
