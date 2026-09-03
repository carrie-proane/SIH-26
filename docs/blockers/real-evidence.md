# Real evidence gate status

Approved local DJI pairs exist at `/Users/jaykelani/Downloads/DJI_0071.MP4` with `DJI_0071.SRT`
and `/Users/jaykelani/Downloads/DJI_0596.MP4` with `DJI_0596.SRT`. FFmpeg, ffprobe and COLMAP are
installed. These facts are sufficient to exercise reconstruction, but not to pass the scientific
known-distance gate.

## Blocking evidence

- No independently tape/laser-measured distance and measurement-photo record is supplied for either
  DJI scene.
- No approved local YOLO, SuperPoint/LightGlue or Depth Anything weights/checksums are supplied.
- Ordinary DJI GNSS is a soft alignment prior, not independent ground truth.

The repository therefore does not claim a ≤10% scale result, a learned-matcher improvement, or
optional-model evidence. Synthetic fixtures cannot satisfy these gates.

## Command once a real reference is available

```bash
PYTHONPATH=src .venv/bin/python -m sih26158.cli run \
  --video /Users/jaykelani/Downloads/DJI_0071.MP4 \
  --telemetry /Users/jaykelani/Downloads/DJI_0071.SRT \
  --video-origin REAL \
  --telemetry-origin REAL \
  --known-distance MEASURED_METRES
```

After reconstruction, record a reconstructed measurement independently in a new run with
`--measured-distance RECONSTRUCTED_METRES`; do not infer either value from telemetry or synthetic
geometry.
