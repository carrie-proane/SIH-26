# Complete source review and implementation decisions

## Contract

The seven-day contract is the controlling scope for this repository. Jay owns:

- FastAPI project/run orchestration and exact run states.
- Immutable manifests, checksums, ffprobe ingest reporting, and declared artifact serving.
- COLMAP invocation, sparse metrics, camera poses, local metric coordinates, and known-distance QA.
- SIFT versus SuperPoint+LightGlue evidence and the keep/cut decision.
- End-to-end command, failure diagnostics, run manifest, frozen dependencies, and backend walkthrough.

Yosha owns telemetry parsing, interpolation, frame extraction/scoring/selection, masks, and the
preprocessing fixtures. Arnav owns the browser UI and final viewer-manifest renderer. This backend
therefore validates those handoffs but does not silently absorb or impersonate their work.

## Ultra-detailed playbook

All 63 pages were reviewed, including feasibility boundaries; people/hardware/software/data needs;
the 14-stage reconstruction path; confidence/provenance policy; target metrics; datasets and capture
SOP; backend/frontend product designs; repository and execution plans; first-48-hour spike; tests;
security/licensing/deployment; risks; evaluator story; mentor questions; budget; definition of done;
and sources.

The implementation carries forward the playbook's main invariants:

- observed geometry and inferred/unseen content stay distinct;
- raw inputs and completed runs are immutable;
- GPS is a soft prior and metric math happens in a local coordinate frame;
- stages have declared inputs/outputs and failures preserve completed evidence;
- confidence uses named, explainable labels rather than invented percentages;
- real reconstruction never falls back to fabricated geometry;
- logs, parameters, versions, hashes, camera path, metrics, and limitations survive the run;
- local/offline operation is the critical path.

## Resolved differences

The playbook describes a broader 27-day/finalist system with PostgreSQL/PostGIS, a durable job queue,
Cesium/3D Tiles, dense reconstruction, multiple export formats, and optional distributed GPU workers.
The later seven-day contract deliberately narrows this to FastAPI, local artifacts, optional SQLite,
COLMAP sparse-first reconstruction, a stable small viewer payload, and stretch dense/mesh work.

For this build, the seven-day contract wins wherever the documents differ. In particular:

- no Postgres, Redis/Valkey, Celery, Kubernetes, cloud dependency, or custom distributed scheduler;
- no dense, GLB, orthomosaic, or learned-depth work before sparse gates pass; bounded per-run
  automatic time-offset calibration is implemented as an integrity fix;
- Three.js/Potree-compatible PLY and a compact viewer manifest instead of mandatory Cesium/3D Tiles;
- the contract's seven exact states instead of the playbook's larger long-term state machine;
- the contract's five confidence labels, including `AI_ASSISTED_NOT_MEASURABLE`.

## Evidence boundary

FFmpeg and COLMAP are available locally, and approved DJI video/SRT inputs have been exercised.
However, no independently surveyed known distance or scientifically valid learned-model evidence is
available. The repository therefore does not claim the real scale, learned-matcher, segmentation, or
depth gates. Synthetic orchestration remains explicitly separate from genuine evidence.
