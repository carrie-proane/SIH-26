# Depth Anything V2 Small — visual-only experiment

**Owner:** Yosha (optional experiment), Jay (runtime/frontend integration)
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
  --model-path /approved/models/depth-anything-v2-small \
  --device cpu \
  --output-dir /approved/evidence/depth-anything \
  --limit 6
```

The command requires a local Transformers model directory, forces offline loading and disables
remote custom code. CPU is the default; any accelerator must be allocated by the caller. The
reference model ID/revision are descriptive, not a verification of the supplied checkpoint.
Model-file hashes, input/output checksums, actual device, runtime and the non-measurable label
are recorded in `depth_anything_evidence.json`. If no real selected frames exist, it exits without creating
fake evidence.

## Current evidence state

The earlier `examples/depth-anything-evidence.blocked.json` is historical. For this round,
`docs/yosha-data-ai-handoff.md` records the current state: a real development capture is supplied,
but local model weights and a demonstrated baseline failure are still missing. No real inference or A/B experiment was run. Work remains deferred behind baseline evidence.

## Frontend fallback

The operator UI understands an optional `depth_overlay_url` on a selected frame. Without it, the AI
toggle is disabled and explains why. Even when present, the source panel carries the
`AI_ASSISTED_NOT_MEASURABLE` label and the 3D distance tool never consumes the overlay.

Raw `*_relative_depth.npy` files preserve floating-point model output at its native prediction
resolution, explicitly **RELATIVE_NOT_METRIC**. The normalized 8-bit PNG is visualization only.
Neither output is eligible for measurement or geometry fusion; cross-view validation is missing.

Offline API reference: [Transformers local-files-only loading](https://huggingface.co/docs/transformers/model_doc/auto).
