# Versioned dataset bundles

`templates/real-dataset.manifest.json` is an empty intake template. Its zero hashes and replacement
paths are deliberate placeholders, not measurements. Copy it next to an approved capture bundle,
replace every placeholder, and keep raw video, telemetry, survey evidence and generated artifacts
outside Git. `examples/` contains schema-only synthetic examples and must never be cited as real
reconstruction or accuracy evidence.

All asset paths are relative to the manifest directory and may not escape it. SHA-256 binds the
video, telemetry and benchmark configuration. Dataset roles are `DEVELOPMENT`, `VALIDATION`, and
`HELD_OUT`; held-out datasets must not be used for configuration fitting.

Reference items are optional so capture/reconstruction work is not blocked. Without independent
`HELD_OUT_EVALUATION` references, however, accuracy remains `UNVERIFIED`. `SCALE_CONTROL` items may
fit scale or a frame transform but are always excluded from independent error statistics. A
positional transform must declare every checkpoint used to fit it; using a held-out checkpoint
causes evaluation to be rejected as control leakage.

The reference frame, units, axis convention, altitude reference, acquisition method and stated
accuracy belong in the manifest. Local ENU, geographic latitude/longitude, ellipsoidal altitude,
orthometric altitude and relative-to-launch height are not interchangeable.

The first real development intake for this round is documented in
[cards/dji_0574/README.md](cards/dji_0574/README.md), with the external bundle location, hashes,
actual preparation report and unresolved reference/model/orientation limitations.
