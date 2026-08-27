# Optional YOLO segmentation contract

The optional runner is outside the core dependency set and is disabled by default. The reference
checkpoint is Ultralytics `yolov8n-seg.pt`, loaded through `ultralytics>=8.3,<9`. Ultralytics models
and code are offered under AGPL-3.0 or an enterprise licence; teams must confirm licence suitability
before distribution. The official weight source is the Ultralytics release infrastructure.

No checkpoint is stored or downloaded by this repository. An operator must place approved weights
locally, record their SHA-256 checksum and enable the run with `enable_segmentation=true` and
`segmentation_model_path`. Missing packages or weights emit
`SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES` and preserve the core COLMAP path.

The runner masks only relevant COCO dynamic classes: people, bicycles, cars, motorcycles, buses and
trucks. It does not claim sky segmentation. Per-frame masks, `dynamic_mask_fraction`, and a small
comparison JSON are declared when execution succeeds. Unit tests use a mocked provider and require
no model weights.
