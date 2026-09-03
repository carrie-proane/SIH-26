# Arnav integration review of Jay and Yosha work

**Reviewed commit:** `1d710e9`
**Review date:** 2026-08-25 (historical review; implementation status has since changed)
**Scope:** integration facts that affect Arnav's viewer and demo route

## Bottom line

> Historical note: automatic scored preprocessing, declared source images, explicit confidence
> artifacts, current COLMAP flags, model selection, and bounded offset calibration were integrated
> after this review. The empirical known-distance gate remains unpassed.

Jay and Yosha have built useful foundations, but the repository does **not** yet contain evidence
for the product promise. The current passing metrics come from an explicitly synthetic ten-camera
fixture. There is no committed real COLMAP run, real preprocessing handoff, real known-distance
validation, or browser-ready source-frame set.

Arnav's UI therefore supports both the live API and an unambiguously labelled offline fixture. It
does not portray fixture results as real reconstruction success.

## Yosha's work

### What is genuinely strong

- CSV and three DJI SRT dialects are normalized to a documented schema.
- Tests cover coordinate order, DJI's `longtitude` typo, relative-vs-absolute altitude, feet-to-metre
  conversion, zero-island fixes, timestamp rebasing/gaps, duplicate timestamps, and high-rate GPS
  jitter.
- Synthetic telemetry is deterministic and is explicitly prohibited from backing a verified
  measurement.
- The primary staircase capture is documented honestly, including rotation, short duration,
  lighting, repeated-feature and ground-truth weaknesses.

### What remains missing from Yosha's seven-day contract

- OpenCV extraction, adaptive scoring/selection, overrides and telemetry interpolation are now
  integrated. Optional YOLO segmentation has a lazy local-weight runner and mocked tests, but no
  approved model weights or real masked/unmasked empirical run exists.
- No backup capture bundle.
- The documented primary clip is a 16.16-second phone video without real GPS telemetry, not the
  controlled 30–60 second drone-video-plus-telemetry input promised for Day 7.
- Its headline long dimension is derived rather than an independently measured scale check.

The frontend handles absent thumbnails, masks and depth overlays as missing declared evidence. It
does not invent them.

## Jay's work

### What is genuinely strong

- FastAPI implements immutable project ingest, SHA-256 checksums, exact run states, artifact index,
  declared-artifact-only serving and retained failure evidence.
- The COLMAP command path fails closed: a real run never silently substitutes synthetic geometry.
- ENU conversion, robust similarity alignment, transformed PLY and aligned camera-pose export are
  implemented and tested.
- Quality and known-distance reports distinguish registration, reprojection, metric alignment and
  independent distance error.
- The code publishes an exact sample viewer schema; current test totals are reported by `make verify`
  rather than frozen in this historical review.

### What remains missing or unproven

- No real COLMAP sparse reconstruction is committed. The supplied PLY is labelled
  `SYNTHETIC_DEMO` and contains fixture points.
- No real primary-scene registration rate, reprojection result or independent known-distance result
  has passed the Day-2 gate.
- No real SIFT versus SuperPoint/LightGlue run exists; only the selection utility and checklist do.
- FFmpeg/ffprobe and COLMAP are now available locally. The remaining empirical blocker is an
  independently measured scene distance and approved optional-model evidence.
- `GET /api/runs/:runId/viewer-manifest` is implemented using declared artifacts only.

## Cross-team integration findings

1. The normal pipeline preprocesses the immutable upload pair automatically. An absolute
   `preprocessing_run` is an optional advanced override and invalid overrides fall back safely.
2. Real aligned camera poses may contain both `sfm_x/y/z` and `x_m/y_m/z_m`; synthetic poses contain
   only metric columns. Arnav's CSV adapter prefers metric columns and safely accepts the fixture.
3. Selected source frames are declared with browser-safe URLs. Masks and depth overlays remain
   disabled unless their own declared artifacts exist.
4. PLY RGB is photographic only. Confidence comes exclusively from a separately validated explicit
   point artifact generated in the same deterministic point order as the PLY.
5. Generated `tmp/`, `egg-info`, `__pycache__` and `.pyc` files are committed. That conflicts with the
   contract's repository-hygiene rule, but they were left untouched because this work is restricted
   to Arnav's ownership.

## Integration decision

The UI is ready for Jay's declared artifacts and safe under missing/failed inputs. The empirical
project is **not submission-ready** until Yosha provides a real preprocessing handoff and Jay runs a
real COLMAP reconstruction with an independent scale check.
