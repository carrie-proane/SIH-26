# Scientific integrity contract

## Source provenance

Every immutable project records `video_origin`, `telemetry_origin`, and the conservative combined
`source_provenance`. Allowed values are `REAL`, `SYNTHETIC`, `DERIVED`, and `UNKNOWN`.
`SYNTHETIC` dominates the combined classification, followed by `UNKNOWN`, then `DERIVED`.
Only a project whose video and telemetry are both declared `REAL` is labelled genuine real evidence.
Telemetry parser evidence can downgrade a run to `SYNTHETIC`; execution mode cannot upgrade it.

## Sparse-model selection

All COLMAP sparse models are inspected. Selection is deterministic: highest registered-image count,
then lowest median point reprojection error, then lexical model path. The complete decision and every
candidate are written to `sparse/model_selection.json`.

## Telemetry synchronization

`sync_report.json` and the run manifest record `telemetry_offset_s`, `offset_source`,
`rmse_before_m`, and `rmse_after_m`. Without a supplied manual or calibrated value, the pipeline
searches offsets from -1.00 to +1.00 seconds at 0.05-second intervals and minimizes robust camera-to-
telemetry alignment RMSE. An offset supplied with `offset_source=calibrated` applies only to that run.

## Explicit point confidence

Photographic PLY RGB values are display colour only and are never confidence evidence. Confidence is
available only from a separately declared `point_confidence.json` with this shape:

```json
{
  "schema_version": "1.0",
  "point_order": "PLY_VERTEX_ORDER",
  "points": [
    {
      "point_id": 0,
      "supporting_views": 5,
      "track_length": 5,
      "reprojection_error": 0.42,
      "triangulation_angle": 8.5,
      "confidence_class": "OBSERVED_HIGH"
    }
  ]
}
```

Point IDs must be unique, contiguous PLY vertex indices, and the record count must equal the PLY
vertex count. Valid classes are `OBSERVED_HIGH`, `OBSERVED_MEDIUM`, `OBSERVED_LOW`,
`AI_ASSISTED_NOT_MEASURABLE`, and `UNSEEN`. If validation fails or the artifact is absent, confidence
filters are disabled and measurements are labelled `Visual estimate - verification confidence
unavailable`.

Observed thresholds are geometric and transparent. HIGH requires track length at least four,
reprojection error at most 1.0 px, and triangulation angle at least 5 degrees. MEDIUM requires track
length at least three, error at most 2.0 px, and angle at least 2 degrees. Other COLMAP-observed
points are LOW. RGB never enters classification.
