# Data, segmentation and bounded-completion handoff

> Historical delivery note: the “not wired” statements below describe Yosha's original branch at
> handoff time. The current `jay/yosha-integration` state supersedes them; see
> `completion-contract.md`, `architecture.md` and `requirements-matrix.md` for integrated status.

Working branch: `yosha/data-ai-completion`.
Baseline fetched from `origin/jay/real-dataset-workflow`:
`8080847469a8fe0f20d816ec5629fdb300e53063` (exactly the pinned handoff).
The starting checkout was clean. No other feature branch was merged. No commit, push, PR or message
to Jay was sent; the local changes are reviewable as a patch.

## Delivered

- [DJI_0574 dataset card](../datasets/cards/dji_0574/README.md), exact manifest/config snapshots,
  [actual preparation report](../examples/yosha/dji_0574-preparation.json), source-preview summary
  and truthful segmentation-unavailable report. Real input hashes verified; zero preparation blockers.
- Existing segmentation seam extended with resolved provider/model metadata, checkpoint hashes,
  explicit bounded inference settings, cancellation boundaries, exact operational-mask polarity,
  cleanup and source/mask/overlay review artifacts. Manual-review fields remain unreviewed until
  someone inspects real model output. See [segmentation contract](optional-segmentation.md).
- Standalone bounded planar infill and depth-consistent additional-frame proposals, with proposed
  additive schemas, separate inferred faces, exact coordinate/order round trips and immutable source
  checks. See [completion contract and integration checklist](completion-contract.md).
- Depth overlay helper now requires local offline weights, records actual model files/device and
  preserves floating-point relative depth separately from its 8-bit visualization. This is a safety
  improvement to an optional experiment, not a depth-fusion pipeline or real execution result.
- A small [synthetic planar fixture](../tests/fixtures/completion/README.md) and
  [removed-patch report](../examples/yosha/synthetic-removed-patch-evidence.json), explicitly separated
  from real capture evidence. The ideal flat fixture produced six inferred faces and zero plane
  residual; this does not estimate real gap error.
- Focused tests for failure behavior, checksums, mask polarity, unsupported classes, small/large gaps,
  openings, corner/support rejection, source immutability, inferred-face IDs, coordinates, occlusion,
  additional-view selection and offline loading. One existing dense test now mocks its help probe,
  so its command-contract assertion does not require an installed `TextureMesh` binary.

## Real evidence and missing prerequisites

The approved real bundle is outside Git at:
`/home/scaiety/Documents/Cllg/hackathon/SIH-26-datasets/dji_0574-development`.
The original supplied files in the repository root remain unchanged and ignored. Raw footage,
telemetry, previews and model weights are not part of the patch.

Preparation: `READY_WITH_WARNINGS`, no blockers, accuracy
`UNVERIFIED_NO_HELD_OUT_REFERENCE`. Calibration, altitude datum and independent references are
missing; explanations are in the dataset card. FFprobe/display rotation differs from the preparation
report, and the viewed source contact sheet looks sideways after the extractor's metadata correction.
Keep this visible as an orientation/calibration review item.

The persistent source preview is at
`/home/scaiety/Documents/Cllg/hackathon/SIH-26-datasets/preview-not-benchmark/source_contact_sheet.jpg`.
A fresh source-only preview reused existing extraction, scoring and selection: 12 candidates,
6 selected, maximum dimension 960, 180° metadata rotation, no extraction warnings. This is a QA
preview with settings different from the hashed benchmark config; it is not a reconstruction run,
cache reuse result, segmentation overlay or runtime benchmark.

The optional segmentation invocation requested local YOLO but executed **no provider**, returning
`UNAVAILABLE_FALLBACK`: no local weights were supplied. No synthetic masks were substituted on real
images. Real mask QA, missed-object/overmask review and masked/unmasked A/B remain **BLOCKED** on
approved weights and fresh reconstruction execution. COLMAP/OpenMVS reconstruction was not run,
as the first-delivery instructions require preparation without reconstruction.

No declared observed dense mesh, metric depth/poses, real gap references or symmetry evidence was
supplied. Completion and coverage are fixture-tested standalone prototypes; real execution is
**UNAVAILABLE**. They are not wired to the pipeline, API, viewer or exporters. Symmetry, terrain and
learned fusion remain deferred behind stable observed geometry. No test establishes real accuracy,
visible-scene completeness or the 15-minute target. Intermediate, ten-minute and HELD_OUT captures,
independent measurements/surveys and server evidence are still needed. The accepted matcher in the
bundle is SIFT; unsupported learned-matcher execution remains Jay's integration issue.

## Configuration and integration changes

No `RunConfig` fields or shared runtime enums were changed. New module-level settings are explicit
CPU device, image size, confidence, max detections/frames, cooperative timeout and review sample
count. The resolved fingerprint helper is supplied for Jay to integrate; the existing pipeline still
misses environment-resolved model changes. Use fresh runs until that is fixed.

The proposed completion source/review/parameter schemas and every cache dependency are documented
in the completion contract. Jay must freeze them, wire resource controls/cancellation/fingerprints,
define linked-run or derived-artifact publication, preserve inferred provenance through exports,
map to the existing non-measurable confidence label, and include enabled work in total runtime.
No external coordination or contract approval is claimed.

## Reproduce the real preparation

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli prepare-dataset \
  --manifest /home/scaiety/Documents/Cllg/hackathon/SIH-26-datasets/dji_0574-development/dataset.manifest.json \
  --output /tmp/dji_0574-preparation.json
```

Use the bundle's own manifest, not the documentation snapshot directory. FFprobe is sufficient;
preparation does not require a GPU, COLMAP or OpenMVS. Additional reference records must use the
existing `DatasetManifest` schema; photo descriptions should identify endpoints, instrument,
stated accuracy and units. Keep controls and held-out evaluation references separate.

## Verification

Final verification completed on 2026-09-22 (Asia/Kolkata); these are actual current results,
not the historical handoff totals. Initial baseline collection was blocked by a missing declared `trimesh`
dependency. Installed the branch's existing `.[dev]` dependencies in `.venv`. Sandboxed TestClient
startup hung before the application ran; the API suite runs with local socket access outside that
sandbox. The first complete backend run produced 190 passes and one pre-existing mock omission
(`TextureMesh -h`); the omission was fixed without changing the dense provider.


| Command/check actually run | Final outcome |
|---|---|
| `make verify` (outside sandbox with isolated local test API) | PASS, exit 0 |
| Backend pytest through `make verify` | 193 passed in 23.41 s |
| Frontend Vitest | 23 passed across 8 files |
| TypeScript + Vite production build | PASS |
| Playwright Chrome browser tests | 2 passed in 7.6 s |
| `make lint` | PASS |
| `make hygiene` / verification hygiene target | PASS for 155 tracked baseline files |
| Same hygiene rules including new untracked patch files | PASS for 178 files |
| Ruff on optional depth script (outside Makefile lint scope) | PASS |
| `git diff --check` | PASS |
| Real `prepare-dataset` command | READY_WITH_WARNINGS; zero blockers; no reconstruction |

The existing Playwright configuration uses `/private/tmp`; on this Linux host its API was prestarted
with `SIH_DATA_ROOT=/tmp/sih26158-yosha-e2e-projects`, and the existing `reuseExistingServer` behavior
was used. The temporary API was stopped after verification. No frontend/configuration edits were
needed. Harmless terminal `NO_COLOR`/`FORCE_COLOR` notices appeared during browser verification.

Files changed: existing segmentation implementation/tests; optional-depth script/docs; dataset and
requirement documentation; one isolated dense test mock. New files: completion/coverage modules and
contracts, JSON schemas, focused tests and small synthetic fixture, real dataset snapshots/cards,
actual preparation/unavailable reports and this handoff. Raw inputs and large outputs remain outside
the patch. Every remaining real-data/model/server/contract prerequisite above is unresolved rather
than counted as a passing metric.
