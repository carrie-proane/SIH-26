# Three-minute Jay backend walkthrough

1. Run `make doctor` and show the exact FFmpeg/COLMAP availability.
2. Start the API with `make api`.
3. Upload the controlled video and telemetry; open the returned immutable manifest and checksums.
4. Start a preview run with Yosha's preprocessing handoff.
5. Poll the run and point out the exact state, progress, retained artifacts, and actionable error field.
6. Open `ingest_report.json`, `sparse_metrics.json`, `camera_poses.csv`, and `quality_report.json`.
7. Show registered-frame rate, reprojection error, known-distance error, and the separate limitations.
8. Open `matcher_benchmark.json` and explain why SIFT or SuperPoint+LightGlue was retained.
9. Fetch the PLY through the declared artifact URL; demonstrate that undeclared paths return 404.
10. Hand the stable sample viewer manifest to Arnav's UI and finish with the offline/local boundary.

If real input or COLMAP is unavailable, use `make demo` only as an orchestration smoke test. State
clearly that its PLY and metrics are synthetic fixtures and are not feasibility evidence.

