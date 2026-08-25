# Depth Anything V2 Small — visual-only experiment

**Owner:** Arnav
**Confidence label:** `AI_ASSISTED_NOT_MEASURABLE`
**Measurement:** permanently disabled

This experiment produces relative monocular-depth overlays for a small selected-frame set. It is
strictly outside the COLMAP, alignment, point-cloud, and distance-tool paths.

## Model and licence

- Model: `depth-anything/Depth-Anything-V2-Small-hf`
- Frozen model revision: `32d03942121d29edb49de4e2cc15831558af3f36`
- Licence: Apache-2.0 for the **Small** model. Base/Large/Giant use a different, non-commercial
  licence and are deliberately not configured here.
- Upstream: https://github.com/DepthAnything/Depth-Anything-V2
- Transformers model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

## Reproduce

Use 3–6 real retained frames from Yosha's preprocessing handoff:

```bash
python3 -m venv .venv-depth
source .venv-depth/bin/activate
python -m pip install -r experiments/depth-anything/requirements.txt
python scripts/depth_anything_overlay.py \
  --input-dir /path/to/yosha-handoff/frames \
  --output-dir evidence/arnav/depth-anything \
  --limit 6
```

The command records model revision, input/output checksums, device, runtime and the non-measurable
label in `depth_anything_evidence.json`. If no real selected frames exist, it exits without creating
fake evidence.

## Current evidence state

`examples/depth-anything-evidence.blocked.json` records the present state. The repository does not
contain Yosha's selected-frame images, so an actual before/after experiment has not been run. That
is a real blocker, not a completed Day-4 empirical result.

## Frontend fallback

The operator UI understands an optional `depth_overlay_url` on a selected frame. Without it, the AI
toggle is disabled and explains why. Even when present, the source panel carries the
`AI_ASSISTED_NOT_MEASURABLE` label and the 3D distance tool never consumes the overlay.
