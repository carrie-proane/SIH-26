# DJI_0574 development capture

This is the first user-approved real capture for this round, provided on 2026-09-21. Role:
**DEVELOPMENT**. It is not a held-out capture, a ten-minute benchmark or independent ground truth.

Persistent external bundle:
`/home/scaiety/Documents/Cllg/hackathon/SIH-26-datasets/dji_0574-development`

`manifest.snapshot.json` and `benchmark.json` are small handoff snapshots. Run preparation against
the external bundle's `dataset.manifest.json`; raw paths do not resolve inside this documentation
directory. Original user files remain untouched and ignored by Git.

| Asset | SHA-256 |
|---|---|
| `raw/capture.mp4` | `72651891c53acf108fc53fd8412e183a8edb03bfb59d853b3e8f40433700e7ed` |
| `raw/telemetry.srt` | `795871e745666654c3e0b9bee06236def8c2a136144c226f7a64288eabb639ce` |
| `benchmark.json` | `0fba069e65f6971cbacf8cb67a8bd498c22d798040da4f078d08b13b18535223` |
| `dataset.manifest.json` | `2b5d7a012ab5be7c81604927169cf8893f46430b6fb5d61947e1b7cdc463978e` |

Preparation returned **READY_WITH_WARNINGS**, zero blockers, reconstruction not started. Actual report:
[DJI_0574 preparation](../../../examples/yosha/dji_0574-preparation.json).

The video is 32.265567 s, H.264, 3840×2160 at 30000/1001 fps, 241,962,258 bytes. DJI make is inferred
from the encoder; device model is **UNKNOWN**. Recording metadata is not camera calibration.
The SRT parser identifies DJI `gps_tuple`, 1,075 samples, a 32.22 s span and lat/lon/alt fields present
throughout. The estimated SRT sampling rate is 33.333 Hz, distinct from the video frame rate.
The first embedded SRT clock is ten seconds after the container creation-time string; their clock
semantics/timezones are not established, so this is a sync-review item, not a measured offset.
Telemetry time uses relative SRT cue times. Absolute embedded date strings were not used
to establish a synchronized clock. Plausible duration coverage does not independently verify sync.

Every preparation warning is explained:

- `capture.calibration`: no independently obtained intrinsics/distortion/calibration file supplied.
- `telemetry.altitude_reference`: datum is UNKNOWN. GPS/barometer fields do not establish a survey
  reference, so vertical/geolocated accuracy remains unavailable.
- `references.independent_evaluation`: no identifiable measured endpoints, instrument accuracy,
  photos or surveyed checkpoints supplied. Accuracy is `UNVERIFIED_NO_HELD_OUT_REFERENCE`.

Additional QA limitations: FFprobe display-matrix rotation is -180°, equivalent to the extractor's
180° correction. Preparation's video check reports rotation null, so the manifest records it
explicitly. A 12-frame, 960-pixel preview using existing extraction/selection retained six frames;
rotation was applied once, with no extractor warnings. The reviewed contact sheet still looks
sideways. Confirm the intended capture orientation before calibration or changing image geometry.
This preview differs from the hashed reconstruction configuration and is not A/B/performance evidence.

No real segmentation weights are available. The real preview's segmentation call returned
`UNAVAILABLE_FALLBACK` and did not create masks. Source-only thumbnails are not a segmentation
contact sheet. No reconstruction, observed dense mesh, completion or measurement was performed.

Next intake: identifiable tape/laser dimensions and photographs across the scene, separately labelled
SCALE_CONTROL or HELD_OUT_EVALUATION; surveyed checkpoints for positional validation; calibration and
vertical datum evidence; intermediate capture; fresh ten-minute capture; separate HELD_OUT capture.
The 595–605 s benchmark window is a branch convention, not organizer wording. Organizer clarification
of accuracy statistic, completeness denominator and mandatory formats remains outstanding.

Persistent source QA preview:
`/home/scaiety/Documents/Cllg/hackathon/SIH-26-datasets/preview-not-benchmark/source_contact_sheet.jpg`.
The external bundle also includes `preparation.json` and an intake README.
