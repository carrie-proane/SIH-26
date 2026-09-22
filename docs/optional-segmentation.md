# Optional local segmentation contract

Owner for this round: Yosha (provider/mask QA), Jay (resources, fingerprints and orchestration).
The existing callable seam and `(artifacts, warnings)` return shape remain compatible. Enable through
existing run configuration; the provider itself defaults to `FULL_SCENE`. `OFF` does no inference,
`AUTO` reports unavailable/failed unmasked fallback, and `REQUIRED` or `PRIMARY_SUBJECT` failures
remain blockers. Required masking is never weakened. Empty selected-frame sets cannot claim success.

Supply an approved local `.pt` segmentation checkpoint through `segmentation_model_path` or
`SIH_SEGMENTATION_MODEL`. The report identifies the resolved source, absolute path, SHA-256,
requested/executed provider, package versions, parameters, actual inference device and elapsed time.
Injected callables are labelled `INJECTED_PROVIDER` regardless of a model-path argument. Their device
is unknown unless they expose `execution_metadata`; they are not evidence of YOLO execution.
Missing packages, unreadable/invalid checkpoints and wrong model tasks produce explicit fallback or
blocker reports. No model is fetched; automatic dependency installation is disabled before loading
Ultralytics. Use only approved local weights. No checkpoint or model execution is included here.

`SegmentationSettings` adds explicit device (CPU by default), inference size, confidence,
max detections, max frames, cooperative deadline and contact-sheet sample count. Accelerator use
requires caller authorization after scheduler allocation. The caller must pass its cancellation
callback; cancellation/timeouts propagate and remove partial masks. Checks run before and after
model calls. A stuck call needs Jay's existing managed-process isolation for a hard timeout; this
module does not create another runner. The current pipeline does not yet wire these new controls.

The PREPROCESS integration must consume `segmentation_fingerprint_inputs` so environment-resolved
weights and all new settings invalidate caches. That helper is tested but intentionally not wired
into Jay's pipeline here. Use fresh runs until the integration is complete.

## Mask and review artifacts

| Output | Convention |
|---|---|
| `masks/<stem>_dynamic.png` | Nonzero excludes pixels; retained original name for compatibility |
| `masks/reconstruction/<image_name>.png` | COLMAP: zero excluded, 255 retained, including original extension |
| `masks/reconstruction/<image_name>.mask.png` | OpenMVS: zero excluded, 255 retained |
| `segmentation_comparison.json` | Additive v2.1 execution and exclusion statistics; not a reconstruction A/B |
| `segmentation_contact_sheet.jpg` | Up to six selected source/mask/overlay triplets by default |
| `segmentation_review.json` | Image/mask hashes, excluded fractions and initially unreviewed human-QA fields |

Per-frame `dynamic_mask_fraction`, `mask_url` and `mask_semantics` remain compatible. The provider
reads processed pixels without applying EXIF orientation again. YOLO requests original-grid masks
with `retina_masks=True`; binary masks use nearest-neighbor resizing. Original frames stay unchanged.
Duplicate stems, malformed/non-finite masks and masks excluding at least 98% fail safely, cleaning
partial outputs and metadata. No detections is valid for `FULL_SCENE`; trivial primary-subject masks
are rejected. Switching to OFF also clears previous operational masks for the supplied frames.

Full-scene exclusion targets person, bicycle, car, motorcycle, bus and truck, plus sky only when the
model actually exposes sky. Animals are not excluded. A parked car is not proven dynamic by its
class. Model class availability is recorded after inference. The filename is historical terminology,
not a claim of motion detection. Scene-policy heuristics remain distinct from semantic masks.

Manually review missed objects and over-masked static surfaces before claiming usable segmentation.
Then run fresh masked/unmasked reconstructions with identical raw inputs, selected frames,
intrinsics and settings. Record requested/executed backend, registration, geometry inspection,
ghost evidence and runtime; mark backend changes as confounds. Pixel fractions and point counts are
not accuracy improvements. See [the handoff](yosha-data-ai-handoff.md) for actual evidence blockers.

Implementation reference: [Ultralytics prediction settings](https://docs.ultralytics.com/modes/predict/)
documents explicit `device`, inference limits and original-resolution `retina_masks`.
