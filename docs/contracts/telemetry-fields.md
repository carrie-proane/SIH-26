# Telemetry field reference

**Day-1 deliverable (Yosha): "record the necessary telemetry fields."**
Findings from building the parser spike against three DJI SRT dialects and three CSV exporter layouts.

## What DJI actually emits

DJI has shipped at least three incompatible SRT layouts. The parser detects which one it is rather than assuming, because assuming wrong fails *silently* — you get a plausible flight path in the wrong place.

| Dialect | Seen on | Coordinate form | Altitude field |
|---|---|---|---|
| `mavic3_bracket` | Mini 3/4, Mavic 3, Air 2S and later | `[latitude: X] [longitude: Y]` | `rel_alt` and `abs_alt` in one bracket |
| `legacy_spaced` | Older Mavic / Phantom | `[latitude : X] [longtitude : Y]` | single `altitude` |
| `gps_tuple` | DJI GO era / Phantom | `GPS(lon,lat,sats)` | separate `BAROMETER:` |

## Fields we consume

| Field | Source | Why |
|---|---|---|
| cue start time | SRT timecode | Becomes `timestamp_s`. Already relative to video start, so no offset estimation needed. |
| `latitude` | all dialects | Flight path, georeferencing |
| `longitude` / `longtitude` | all dialects | Same. Note the spelling. |
| `rel_alt` | mavic3 | Height above launch — directly usable in the local metric frame |
| `BAROMETER` | gps_tuple | Altitude for that dialect |

## Fields we deliberately ignore

`iso`, `shutter`, `fnum`, `ev`, `ct`, `color_md`, `focal_len` are present in most SRT cues. None feed the Day-7 must-ship list, so they are not in schema v1.

One is worth revisiting: **`focal_len`**. If COLMAP's self-calibration turns out unstable on the primary scene, a focal-length prior from telemetry could help Day-2's registration gate. Not in scope now — flagged for Jay in case Day 2 registration comes in under 80%.

## Traps found (each has a test)

**1. `abs_alt` is not altitude above ground.** DJI writes both. `abs_alt` is barometric MSL — in the fixtures, 590 m for a drone flying 30 m above launch. Using it puts everything 560 m in the air. We take `rel_alt` and record `alt_source` so this is traceable.

**2. `GPS(...)`'s third element is satellite count, not altitude.** `GPS(73.85,18.52,14)` means 14 satellites. Reading it as altitude turns a 22 m flight into a 14 m one. Altitude for that dialect comes from `BAROMETER`.

**3. The GPS tuple is longitude-first.** Positionally reading it as (lat, lon) moves a Pune flight into the Indian Ocean.

**4. DJI misspells longitude as `longtitude`** in several firmware versions. Missing this yields latitude-only data and a flight path running along a meridian.

**5. Airdata CSV exports feet by default.** `height_above_takeoff(feet)`. Read as metres, every altitude is 3.28x too large. We parse the unit from the header and convert.

**6. `lat == lon == 0` is a no-lock sentinel**, not a position off Ghana. Treated as a missing fix.

**7. Some CSV exports have no elapsed-time column** — only absolute datetime. We rebase onto the first sample and emit `TIME_REBASED`, because that silently assumes telemetry starts with the video, and automatic time-offset estimation is cut scope.

## Two bugs found and fixed during the spike

**Per-sample speed checking is unusable at 30 Hz.** DJI writes one SRT cue per video frame. At 30 Hz, samples are 33 ms apart, and consumer GPS jitters ~1 m between samples — implying 30 m/s. The check fired on every valid file. Now measured over a ≥1 s window, which still catches transposed coordinates and misplaced decimals. Regression test: `test_high_rate_srt_does_not_trigger_false_speed_warning`.

**Independent noise in the synthetic generator inflated path length by 119%.** A 157 m orbit measured 345 m, and apparent ground speed doubled. Real GPS error drifts over seconds rather than resampling each tick, so the generator now uses an AR(1) process with a ~5 s correlation time. Path length is now within 3% of truth. Regression test: `test_synthetic_noise_is_temporally_correlated`.

## Open questions for Jay

1. **Altitude datum.** We emit height above launch. Confirm that is what the local metric frame wants, or say what it should be rebased to.
2. **Sample rate.** A 30 Hz SRT over 45 s is ~1350 rows for maybe 90 retained frames. Does interpolation want the full-rate track, or a decimated one? Full rate is currently kept — decimating is lossy and one-way.
3. **`focal_len` prior.** Want it plumbed through if Day-2 registration underperforms?
