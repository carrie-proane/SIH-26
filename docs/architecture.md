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
declared artifact index -> quality report / PLY / camera poses / export packages / viewer payload
```

Every stage boundary is recorded in the run's internal `.pipeline_checkpoint.json`. A stage is
resumable only when every checkpointed artifact still exists and matches its recorded SHA-256 and
the stage's input, effective-configuration, and relevant executable fingerprints still match.
The internal `.execution.lock` gives each run a single filesystem-level executor, including across
multiple API processes. Neither internal file is registered or downloadable as evidence.

## Ownership boundary

Jay owns the operator frontend, project/run API, run state, backend integration, measurement policy,
artifact publication, server execution and final evaluation. Yosha owns dataset/reference
preparation, segmentation/model improvements, coverage diagnostics and constrained/symmetry
completion algorithms. Existing frame, telemetry and mask contracts remain integrated; a compatible
external handoff remains an optional debugging override. Future completion output must be additive,
separate inferred geometry and remain excluded from verified measurement.

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
- Separate `.resource_locks/heavy-N.lock` slots limit expensive work across different runs; the
  default is one slot and `SIH_HEAVY_JOB_LIMIT` may raise it within scheduler allocations.
- A live `.active_process.json` PID marker blocks crash recovery while a managed child survives;
  heartbeat expiry alone never authorizes duplicate execution.
- Long COLMAP/OpenMVS commands execute in isolated process groups with a shared stage deadline,
  periodic manifest heartbeat, and cross-process cancellation marker. Cancellation terminates the
  process group and records `CANCELLED` without relabeling it as reconstruction failure.
- `server_capabilities.json` freezes the host preflight used to choose CPU/GPU sparse execution and
  the optional dense provider. A requested but unavailable CUDA path falls back to CPU sparse with
  an explicit warning; dense unavailability never fabricates visual artifacts.
- `SYNTHETIC_DEMO` is labeled in every artifact and cannot be mistaken for genuine geometry.
- A real run never falls back to synthetic geometry.
- Candidate images stay under `preprocessing/candidates/`; only selected images enter `frames/`.
- Candidate retention is uniformly bounded across the whole video. Reports bin candidate,
  selected, and registered frames by time and disclose long gaps and excluded components.
- PLY RGB is photographic display data. Explicit confidence is exported with the PLY in the same
  deterministic point order and validated by vertex count.
- Geometry exporters consume only checksum-verified declared artifacts. Each OBJ, GLB, or LAS
  package is built in a staging directory, reopened, hashed, and atomically renamed before its
  files are added to the declared artifact index.
- Export cache identity includes exporter version, source/dependency hashes, options, and the full
  coordinate contract. A changed origin or local transform cannot reuse stale metadata.
- Export success does not change the source provenance or measurement eligibility. Sparse points
  are never triangulated into an observed mesh, and unsupported attributes are disclosed rather
  than silently relabelled or dropped.

## Local coordinate alignment

`geo.py` converts WGS84 latitude/longitude/ellipsoidal height to ECEF and then local ENU metres.
It implements a 7-DoF Umeyama similarity fit and robust MAD outlier rejection. Absolute horizontal,
vertical, relative, and scale errors remain separate report concepts.

The geometry export contract preserves this distinction:

| Format | Output axes and units | Transform from local ENU |
|---|---|---|
| OBJ | X east, Y north, Z up; metres | Identity |
| LAS 1.2 point format 2 | X east, Y north, Z up; metres | Identity; per-axis scale/offset quantization |
| GLB/glTF 2.0 | X east, Y up, Z south; metres | `(x, y, z) -> (x, z, -y)` |

For homogeneous column vectors, the ENU-to-glTF transform is:

```text
[ 1  0  0  0 ]
[ 0  0  1  0 ]
[ 0 -1  0  0 ]
[ 0  0  0  1 ]
```

Its inverse is recorded alongside every GLB. The transform is a proper rotation, so lengths and
handedness are preserved. Export validation checks finite coordinates, counts and bounds; fixture
tests reopen the formats and transform GLB bounds back to ENU within declared tolerances.

Local ENU has no fabricated EPSG assignment. `local_transform.json` supplies the WGS84 origin and
altitude-reference declaration when present, but ordinary GNSS/local alignment does not establish
a survey-grade vertical datum or independently verified absolute geolocation.
