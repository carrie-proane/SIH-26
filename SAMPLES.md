# Sample bundles

**Owner:** Yosha · Day-1 deliverable

Raw video and telemetry are **not committed** (contract §8). This file is the registry: it records what each bundle is, where it lives, and what ground truth it carries. Small derived artifacts (`normalized_telemetry.csv`, `.meta.json`) may be committed as approved samples.

## Status

| Bundle | Scene | Status | Telemetry | Ground truth |
|---|---|---|---|---|
| `primary_staircase` | Hostel staircase, 9 steps + railing | ✅ **captured** | ⬜ none — needs synthetic | 3 measured, 3 derived |
| `backup` | TBD — smaller/simpler | ⬜ **not captured** | TBD | TBD |
| `synthetic_orbit` | generated, no video | ✅ generated | `SYNTHETIC_TELEMETRY` | orbit radius 25.0 m, exact |

### primary_staircase

- **Scene:** hostel staircase — single flight, 9 steps, metal railing, entrance door at top landing
- **Captured:** 2026-08-25 by Yosha, phone (Samsung, Android 14)
- **Video:** `20260825_154043.mp4`, 16.16 s, 1920×1080 @ 30 fps, h264 High, 17.23 Mbps, 483 frames
- **Orientation:** ⚠️ `-90°` in a Display Matrix side-data flag, **not baked into the stream**. Effective frame is portrait 1080×1920. Extraction must honour it or frames come out sideways.
- **sha256 (video):** `142e5fb9fa2d0d50ddf619aadf3875aed75f5fdc2677f54de53e3d317c1c90ce`
- **Telemetry:** none — phone capture. Needs a synthetic companion track (see below).
- **Ground truth:** `ground_truth.json` — 3 measured primitives, 3 derived baselines
- **Headline reference:** `M6`, flight slope length, **2.934 m** (derived)
- **Frames:** 81 extracted at 5 fps into `frames/`

**Reconstruction viability — checked, it passes.** Motion was tested for usable parallax by comparing homography vs fundamental inlier ratios across the clip:

| Frame pair | Homography | Fundamental |
|---|---|---|
| 010→020 | 0.40 | 0.70 |
| 030→045 | 0.16 | 0.44 |
| 060→080 | 0.06 | 0.22 |

Fundamental dominates throughout, so the camera genuinely translates rather than merely rotating — SfM has something to triangulate. Roughly 2,500–3,000 ORB features per frame.

**Known weaknesses:**

- 16.2 s against the contract's 30–60 s target, limiting retained keyframes.
- Portrait orientation narrows horizontal FOV, so less scene context per frame.
- Fundamental inlier ratio falls 0.70 → 0.44 → 0.22 through the clip — likely the identical steps filling more of the frame on ascent, creating matching ambiguity.
- Backlit window at the top landing; expect blown highlights in later frames.
- Every long baseline is **derived**, not independently measured. See `honesty_note` in `ground_truth.json`.

**Extraction command** (orientation handled — ffmpeg honours the Display Matrix by default; do not add a manual `transpose`):

```bash
ffmpeg -i 20260825_154043.mp4 -vf "fps=5" -q:v 2 frames/frame_%04d.jpg
# verify: extracted frames must be 1080x1920 (portrait), not 1920x1080
```

**Telemetry:** generate a synthetic companion track matched to the real 16.2 s duration, and register it as `SYNTHETIC_TELEMETRY`:

```bash
python scripts/make_synthetic_telemetry.py --pattern line \
  --center-lat 18.5204 --center-lon 73.8567 \
  --radius-m 1.5 --alt-m 1.6 --duration-s 16.2 --rate-hz 10 \
  --out-dir data/samples/primary_staircase/
```

A stairwell has no GPS anyway, so this track is scaffolding to exercise the pipeline — it must never back a verified measurement.

`synthetic_orbit` is a parser and alignment fixture, **not** a demo bundle. It has no video and cannot be reconstructed. It exists so Jay can validate local-frame alignment against a track whose true geometry is known exactly, and so the SRT path is exercised deterministically in CI.

Regenerate it with:

```bash
python scripts/make_synthetic_telemetry.py --pattern orbit \
  --center-lat 18.5204 --center-lon 73.8567 \
  --radius-m 25 --alt-m 30 --duration-s 45 --rate-hz 10 \
  --out-dir data/samples/synthetic_orbit/
```

## Capture protocol

The contract's Day-2 gate is ≥80% frame registration and a known dimension within 10%. That gate is won or lost at capture time, not in code. Follow this.

### Scene choice

Pick a building facade or courtyard with:

- **Strong, non-repeating texture** — brick, stone, signage, window frames. COLMAP needs distinguishable features.
- **No large glass or mirrored surfaces.** Reflections generate features that move with the camera and actively corrupt matching.
- **No blank painted walls.** A flat single-colour wall gives nothing to match.
- **Little foot and vehicle traffic.** Dynamic content is what Day-4's YOLO masks address, but less is better.
- **Even lighting.** Overcast is ideal. Harsh midday sun produces blown highlights and black shadow that both score badly on exposure.

Avoid: anything with repeating identical units (a long row of identical windows causes false matches), and anything you cannot physically reach to measure.

### Flight/capture path

- **Orbit or wide arc around the subject**, not a straight line past it. Reconstruction needs baseline — a straight pass gives poor triangulation geometry and registration collapses.
- **Constant altitude**, roughly 1.5–2x the subject's height.
- **Camera angled inward and slightly down**, subject kept in frame throughout.
- **Move slowly and continuously.** Motion blur is the top cause of frames failing the blur score.
- **30–60 seconds** (contract §1). Longer is explicitly out of scope.
- **60–80% overlap between consecutive viewpoints.** Err toward more.

### Ground truth — do not skip this

Before or after capture, physically measure **at least two** dimensions on the subject with a tape or laser measure. This is what Day-2's 10% check and the Day-7 demo's headline measurement run against. Without it the whole trust story has no anchor.

Record for each: what it is, the measured value, the unit, the instrument, and a photo of the measurement being taken.

Good candidates: a door height, a window width, the spacing between two clearly identifiable features on the facade. Pick things visible in many frames and unambiguous to click on in the viewer.

### If no drone is available

A phone camera works for Day 1 and 2. Hold it as high as possible, walk a slow steady arc, keep the subject centred, do not zoom, lock exposure and focus if the camera allows.

You will have no telemetry. Generate a companion track with `make_synthetic_telemetry.py --pattern arc`, matched to your real duration and approximate radius. **This bundle must be registered here with `SYNTHETIC_TELEMETRY`, labelled in the UI, and must never back a verified measurement claim** (contract §1: AI-assisted and unverified inputs never support a verified measurement).

The ground-truth tape measurement is still real and still valid — that comes from the physical world, not the telemetry.

## Registering a bundle

When a bundle is captured, add a section here:

```markdown
### primary

- **Scene:** <building / courtyard, location>
- **Captured:** <date, time, weather>
- **Device:** <drone model + firmware, or phone model>
- **Video:** <filename>, <duration>s, <resolution>, <fps>, <codec>
- **Telemetry:** <filename>, dialect `<detected>`, <rows> rows, <rate> Hz
- **Path:** <orbit / arc / other>, approx radius <m>, approx altitude <m>
- **Storage:** <shared drive link or path — NOT in git>
- **sha256 (video):** <hash>
- **sha256 (telemetry):** <hash>
- **Ground truth:**
  | What | Measured | Instrument |
  |---|---|---|
  | main door height | 2.14 m | laser |
  | window width | 1.20 m | tape |
- **Parser output:** `data/samples/primary/normalized_telemetry.csv`
- **Warnings:** <from meta.json>
- **Notes:** <anything odd — people walking through, lighting change mid-flight>
```

   ## Where bundles live
   
   **Decided (Day 1): shared Drive/Dropbox folder.** Not Git LFS.
   
   Raw video, photo sets and telemetry live in the shared folder. This registry
   records the link plus a `sha256` for every file, so Day 7's checksum
   verification has something to verify against and so we can prove the
   submitted artifacts came from the bundle we say they did.
   
   Committed to git: only `normalized_telemetry.csv`, its `.meta.json` sidecar,
   `*.truth.json`, and small test fixtures.
   
   **Folder link:** https://drive.google.com/file/d/1u7pOu6Nr4s8f71mPSHGE61yvJe4VRhNO/view?usp=sharing
