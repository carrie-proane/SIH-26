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

Every stage boundary is recorded in the run's internal `.pipeline_checkpoint.json`. A stage is
resumable only when every checkpointed artifact still exists and matches its recorded SHA-256.
The internal `.execution.lock` gives each run a single filesystem-level executor, including across
multiple API processes. Neither internal file is registered or downloadable as evidence.

## Ownership boundary

Jay owns the project/run API, run state, ffprobe report, COLMAP invocation, alignment utilities,
matcher decision, quality report, known-distance evidence, and end-to-end command. Yosha's frame
and telemetry modules are integrated into the ordinary upload route; a compatible external handoff
remains an optional debugging override. Arnav owns the final
`viewer-manifest` endpoint and browser implementation; `examples/viewer-manifest.json` freezes the
payload Jay publishes for UI development.

## Invariants

- Raw inputs are copied once, checksummed, and never edited.
- Completed artifacts are accessed only through the run's declared index.
- Paths containing `..`, absolute paths, and undeclared files are rejected.
- The only states are `QUEUED`, `INGESTING`, `PREPROCESSING`, `RECONSTRUCTING`, `REPORTING`,
  `COMPLETED`, `FAILED`, and `CANCELLED`.
- A failed run retains its manifest, events, logs, and every prior declared artifact.
- Startup recovery requeues interrupted `QUEUED`, `INGESTING`, `PREPROCESSING`, `RECONSTRUCTING`,
  and `REPORTING` runs from their last checksum-valid stage boundary.
- `POST /api/runs/{run_id}/resume` is idempotently deduplicated; active and completed runs cannot
  be submitted again.
- An interrupted COLMAP attempt restarts only its private scratch workspace so a partial SQLite
  database cannot contaminate the retry. Previously declared evidence remains untouched.
- Long COLMAP/OpenMVS commands execute in isolated process groups with a shared stage deadline,
  periodic manifest heartbeat, and cross-process cancellation marker. Cancellation terminates the
  process group and records `CANCELLED` without relabeling it as reconstruction failure.
- `server_capabilities.json` freezes the host preflight used to choose CPU/GPU sparse execution and
  the optional dense provider. A requested but unavailable CUDA path falls back to CPU sparse with
  an explicit warning; dense unavailability never fabricates visual artifacts.
- `SYNTHETIC_DEMO` is labeled in every artifact and cannot be mistaken for genuine geometry.
- A real run never falls back to synthetic geometry.
- Candidate images stay under `preprocessing/candidates/`; only selected images enter `frames/`.
- PLY RGB is photographic display data. Explicit confidence is exported with the PLY in the same
  deterministic point order and validated by vertex count.

## Local coordinate alignment

`geo.py` converts WGS84 latitude/longitude/ellipsoidal height to ECEF and then local ENU metres.
It implements a 7-DoF Umeyama similarity fit and robust MAD outlier rejection. Absolute horizontal,
vertical, relative, and scale errors remain separate report concepts.
