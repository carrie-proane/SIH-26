# Arnav seven-day delivery evidence

This ledger maps Arnav's contract ownership to reviewable local files. It does not mark empirical
work complete when the required real inputs do not exist.

## Day 1 — baseline

- React + TypeScript + Vite operator shell: `frontend/`.
- Three.js photographic-RGB PLY viewer, explicit-confidence filters and camera flight path:
  `frontend/src/components/PointCloudViewer.tsx`.
- Deterministic, explicitly synthetic offline fixture: `frontend/public/demo/`.
- Chrome-rendered UI evidence: `evidence/arnav/operator-ui-fixture.png`.

## Day 2 — exact manifest integration

- Browser types follow Jay's `examples/viewer-manifest.json` rather than an invented payload:
  `frontend/src/types.ts`.
- Manifest/artifact loader accepts aligned real pose columns and the synthetic fixture columns:
  `frontend/src/api.ts`.
- Exact confidence legend toggles, selected-frame list/details, synthetic warning and source-preview
  unavailable state: `frontend/src/components/Workspace.tsx`.

## Day 3 — upload and progress

- Video + SRT/CSV upload, immutable project creation and real COLMAP run submission:
  `frontend/src/components/SetupScreen.tsx` and `frontend/src/App.tsx`.
- Normal uploads use automatic scored preprocessing. The external `preprocessing_run` path and
  include/exclude frame overrides are contained in an Advanced section.
- Exact backend stages, progress, event log, actionable failure and retained-artifact count:
  `frontend/src/components/ProgressScreen.tsx`.

## Day 4 — Depth Anything proof

- Reproducible Depth Anything V2 Small runner with frozen revision, licence, checksums, runtimes and
  non-measurable evidence label: `scripts/depth_anything_overlay.py`.
- Setup/model/fallback note: `experiments/depth-anything/README.md`.
- Current empirical status: `examples/depth-anything-evidence.blocked.json`.

**Honest gate:** not run. No real selected-frame images exist in the repository. The runner exits
without fabricating evidence. The UI keeps the AI toggle disabled until a declared overlay URL is
present and never feeds it to measurement.

## Day 5 — vertical slice

- Contract-owned endpoint `GET /api/runs/:runId/viewer-manifest`:
  `src/sih26158/viewer_manifest.py` plus `tests/test_viewer_manifest.py`.
- Endpoint publishes only declared cloud, pose, keyframe and quality artifacts; incomplete runs get
  HTTP 409.
- Live API smoke-fixture route exercises project upload → run polling → manifest → 3D workspace.
- Two-point 3D measurement enforces confidence behavior: high allowed, medium cautioned, low requires
  confirmation, AI/unseen blocked.
- Quality-report inspector shows registration, reprojection, independent distance, warnings and
  limitations without hiding failures.

## Day 6 — evidence and polish

- Responsive operational dark UI, desktop/tablet/mobile layouts, empty states, missing-preview state,
  cloud-load errors, API errors and failed-run diagnostics: `frontend/src/styles.css`.
- Vitest integration coverage: `frontend/src/App.test.tsx` and `frontend/src/csv.test.ts`.
- Real-browser route: `frontend/e2e/operator-workspace.spec.ts`.
- The browser suite covers both the offline manifest and a live local API upload → synthetic run →
  viewer-manifest → WebGL route.

## Day 7 — freeze and demo

- Frozen frontend dependency ranges and clean commands: `frontend/package.json` and
  `frontend/README.md`.
- Three-minute judge route and fallback behavior: `docs/arnav-demo-script.md`.
- Production build and tests are part of `make ui-check`.

**Still requires real team input:** A backup video cannot truthfully be recorded until the real
primary run, source frames, masks and measured geometry exist. The offline UI fixture is a technical
fallback, not a substitute for that evidence.
