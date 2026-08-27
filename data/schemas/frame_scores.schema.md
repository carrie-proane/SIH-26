# Frame selection artifacts — schema 1.0

`frame_scores.csv` is ordered exactly as follows. Scores range from 0 to 1 and higher is better.

1. `frame_index`: zero-based source-video frame index.
2. `timestamp_s`: seconds from video start (`frame_index / fps`).
3. `source_video`: absolute source-video path.
4. `frame_path`: extracted JPEG path consumed by reconstruction.
5. `blur_score`: normalized Laplacian variance; 1 is sharpest in this run.
6. `exposure_score`: tonal-range score with shadow/highlight clipping penalties.
7. `redundancy_score`: adjacent-frame SSIM dissimilarity; 1 is most unique.
8. `composite_score`: configured weighted sum of the three quality scores.
9. `selected`: `True` when retained as a keyframe.

`keyframes.json` has `schema_version`, `source_video`, a `selection` object containing the target, minimum spacing and weights, and a `frames` array. Each selected-frame object contains every CSV field, `image_name` (the basename consumed by reconstruction), and `path` (the extracted-frame path). Paths are not evidence that geometry is verified.
