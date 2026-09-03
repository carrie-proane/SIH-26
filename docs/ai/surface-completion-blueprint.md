# AI blueprint for hidden and occluded surface completion

## Executive recommendation

Build a hybrid, evidence-first completion system rather than trying to replace COLMAP/OpenMVS with
one end-to-end network. Keep the current pipeline as the geometric anchor, convert its observed
surface plus camera visibility into a partial volumetric representation, and train a conditional
generative 3D model to propose one or more completions only in unobserved regions. Reproject every
hypothesis into the source cameras, reject hypotheses that violate observed depth/free space, and
publish predicted regions with per-region uncertainty. Never allow AI-generated vertices to be used
for measurement.

This is a realistic R&D plan. The repository currently has a working reconstruction pipeline and an
integration contract for a future model runtime. It does not yet contain a trained completion model
or a dataset large enough to justify one.

## 1. Problem definition

Let the drone video provide frames I_i, calibrated camera intrinsics K_i, and recovered camera poses
T_i. Multi-view geometry reconstructs an observed surface S_obs. The goal is not to estimate more
samples on S_obs; dense MVS already addresses that. The goal is to estimate a distribution over
complete surfaces:

```text
p(S_complete | S_obs, visibility, images, semantics, camera poses)
```

where S_complete must preserve observed geometry and free space. The distribution matters because
multiple backsides or internal shapes may explain exactly the same visible images. A deterministic
single mesh hides this ambiguity and encourages overconfidence.

### Three different problems that must not be confused

1. Hole filling repairs small gaps inside an otherwise observed surface using local interpolation or
   Poisson-style smoothness. It is geometric, not semantic understanding.
2. Novel-view synthesis renders plausible pixels at a nearby unseen camera. NeRF/3D Gaussian
   Splatting can be excellent for appearance, but does not automatically produce a watertight,
   metrically correct hidden backside.
3. Generative shape/scene completion predicts geometry with learned priors where no measurement
   exists. It can create visually plausible surfaces but may hallucinate.

The project needs all three as distinct layers: observed sparse/dense evidence, optional geometric
repair for tiny holes, and optional AI completion for genuinely unobserved regions.

## 2. Why the current system is a strong starting point

The existing pipeline already solves prerequisites that many ML prototypes ignore:

- camera pose recovery and sparse 3D structure through COLMAP;
- local metric alignment using telemetry and a robust similarity transform;
- keyframe timestamps and selected source images;
- dynamic/irrelevant pixel masks;
- dense observed geometry via COLMAP or OpenMVS when available;
- explicit point confidence and source provenance;
- an artifact registry and browser viewer with measurement gates.

The model should consume these outputs rather than duplicate them. Training an end-to-end video-to-
mesh model immediately would require far more data, GPU time, and calibration work, and it would
discard the traceable observed geometry that judges can verify.

## 3. Proposed architecture

### Stage A - observed reconstruction

Keep the current capture, preprocessing, COLMAP, alignment, and optional dense stages. Tighten camera
calibration: lock zoom/focus, share intrinsics for one physical camera where valid, and calibrate
rolling shutter if the drone motion is fast. The most effective way to reduce occlusion is still
better capture: a complete orbit, multiple heights, high overlap, and no abrupt yaw-only motion.

### Stage B - visibility-aware partial volume

Convert observed depth/mesh into a sparse TSDF or signed-distance/occupancy grid in local ENU.
Maintain separate channels:

- truncated signed distance;
- observation weight or number of supporting rays;
- explicitly observed free space;
- unknown/unobserved mask;
- reprojection/triangulation confidence;
- semantic class probabilities if useful;
- optional learned image features back-projected from each keyframe.

The unknown mask is critical. Empty and unknown are not the same: a voxel in front of an observed
surface may be known free space, while a voxel behind the surface may be unobserved. Treating both as
zero occupancy teaches the network to erase plausible hidden geometry.

### Stage C - learned completion prior

For the SIH prototype, start with a coarse-to-fine conditional 3D network:

1. A sparse 3D encoder ingests the partial TSDF/occupancy, visibility, confidence, and optional
   semantic/image features.
2. A latent bottleneck captures global object/scene structure.
3. A conditional diffusion or stochastic decoder produces K signed-distance/occupancy hypotheses.
4. A refinement network predicts higher-resolution residual geometry around the zero level set.
5. Marching Cubes converts the final SDF to a mesh.

DiffComplete is a relevant reference because it treats shape completion as conditional generative
diffusion over partial 3D observations. SDFusion is relevant for latent SDF generation and optional
multi-modal conditioning. NeuralRecon and Atlas are relevant for fusing posed image features into a
coherent volumetric scene representation. These methods are research foundations, not drop-in proof
that outdoor drone footage will work without domain adaptation.

### Stage D - observed-geometry preservation

Before loss calculation and after inference, enforce hard/soft constraints:

- do not move high-confidence observed surface voxels beyond tolerance tau_obs;
- do not place generated occupancy in ray-traced observed free space;
- permit small corrections only where observed confidence is low;
- blend generated and observed fields in a narrow boundary band to avoid cracks;
- retain a binary provenance label for every output vertex/face.

This can be implemented by clamping the predicted SDF in high-confidence observed cells and applying
a free-space penalty elsewhere. Generated geometry should begin behind uncertain/occluding
boundaries, not overwrite the measured facade simply because the prior prefers symmetry.

### Stage E - multi-view consistency refinement

Render depth, silhouettes, normals, and optionally RGB from each recovered camera. Compare them with
observed masks, MVS depth, and source images. Optimize the latent code or surface only where allowed.
Useful terms include robust depth error, silhouette IoU, normal consistency, photometric error on
stable Lambertian regions, and free-space violation. Do not force photometric consistency through
specular, moving, sky, or masked pixels.

### Stage F - uncertainty and multi-hypothesis output

Run K stochastic samples (initially K=3). Estimate epistemic ambiguity from sample disagreement and
combine it with geometric confidence. Output:

- completed mesh hypothesis selected by validation score;
- optional alternate meshes;
- per-vertex/face generated-vs-observed provenance;
- uncertainty in [0,1], with higher meaning less trustworthy;
- summary statistics for observed, boundary, and generated regions;
- model/checkpoint/dataset/config hashes.

If all samples agree only because the decoder collapses to one mode, uncertainty is not calibrated.
Evaluate calibration on deliberately hidden ground-truth surfaces.

## 4. Recommended model variants

### Variant 0 - non-ML baselines

Implement these first because they reveal whether a neural model actually helps:

- Poisson/watertight meshing for small holes;
- planar/Manhattan continuation for walls and floors;
- symmetry completion for isolated symmetric objects;
- nearest-template retrieval for a constrained object class.

These baselines are fast and interpretable but fail on novel topology and should never be called AI
prediction. Their output remains visual-only if it extends beyond observed geometry.

### Variant 1 - deterministic sparse 3D U-Net

Predict occupancy/SDF from the partial volume. It is easiest to train and debug and is a good first
learned baseline. Its main weakness is averaging: ambiguous regions become over-smoothed and the
single output hides multi-modality.

### Variant 2 - conditional latent diffusion (recommended target)

Compress complete SDF blocks into a latent representation and train diffusion conditioned on partial
geometry, observation mask, and image/semantic features. This is more expensive but naturally
produces multiple plausible hypotheses and supports uncertainty by disagreement. Use classifier-free
conditioning dropout so the model can balance strong geometric evidence with learned priors.

### Variant 3 - category-specific object completion

If the SIH problem is one narrow object class (statue, monument, vehicle, facade module), crop and
canonicalize the subject and train a category-specific model. This is far more achievable than an
open-world outdoor scene completion model. A general campus scene model trained only on ShapeNet
objects will fail through domain mismatch.

## 5. Data strategy

### Ground truth requirement

Training pairs need a complete target and a simulated/real partial observation. A normal single-pass
video supplies only the partial input; it does not supply hidden-surface ground truth. Ground truth
can come from a full multi-ring capture, terrestrial/LiDAR scan, synthetic renderer, or accurately
scanned benchmark.

### Dataset mixture

Use three tiers:

1. Synthetic pretraining: render complete CAD/scene assets from drone-like trajectories and retain
   exact complete SDF, depth, visibility, material, and camera data. Randomize texture, lighting,
   weather, motion blur, exposure, background, focal length, and compression.
2. Public geometry benchmarks: DTU and Tanks and Temples for multi-view reconstruction evaluation;
   ScanNet/ScanNet++ for posed images and complete indoor geometry; ShapeNet-style CAD only if the
   target is object-centric.
3. Project-domain fine-tuning: collect each physical subject with the normal single-pass trajectory
   plus a much more complete reference scan. Split by physical site/object, never by adjacent frames,
   to prevent leakage.

### Creating partial-complete pairs

From every complete mesh:

1. Sample a camera trajectory matching actual drone height, radius, field of view, and missing arcs.
2. Rasterize depth/silhouette for each camera.
3. Fuse only visible depth into the partial TSDF.
4. Ray-carve known free space and mark never-traversed voxels unknown.
5. Save the complete target SDF/occupancy and surface.
6. Add realistic pose, depth, mask, and scale noise measured from the current pipeline.

The training partials must resemble failure modes of this system. Perfect synthetic depth paired with
real COLMAP noise will produce a brittle model.

## 6. Representation and resolution

- Use local object/scene coordinates centered and scaled for the network, but retain the transform
  back to local ENU.
- Begin at 64^3 or sparse equivalent for the prototype; use coarse-to-fine tiles for larger scenes.
- Use a narrow-band TSDF because most useful information is near surfaces.
- For buildings, use overlapping spatial blocks and a global low-resolution context encoder.
- Record voxel size in metres. A visually smooth 64^3 output over a 40 m scene cannot claim
  centimetre detail.

## 7. Training objectives

Let M_obs be high-confidence observed cells, M_free known free space, M_unknown cells with target
available during training, D_pred predicted SDF, and D_gt ground-truth SDF.

```text
L = lambda_sdf      * robust_L1(D_pred, D_gt, M_unknown)
  + lambda_surface  * BCE(occupancy(D_pred), occupancy(D_gt))
  + lambda_observed * robust_L1(D_pred, D_input, M_obs)
  + lambda_free     * free_space_violation(D_pred, M_free)
  + lambda_eikonal  * (||grad D_pred||_2 - 1)^2
  + lambda_normal   * normal_consistency
  + lambda_render   * multi_view_depth_and_silhouette_error
  + diffusion_noise_prediction_loss
```

Do not let the large number of easy empty voxels dominate the loss; use surface-focused sampling or
class-balanced weights. Use robust losses around noisy observed surfaces. Tune the observed/free-space
terms high enough that generation fills unknown regions without rewriting evidence.

## 8. Evaluation protocol

### Geometry metrics

- Accuracy/precision: predicted-to-ground-truth surface distance.
- Completeness/recall: ground-truth-to-predicted surface distance.
- F-score at fixed metric thresholds.
- Symmetric Chamfer-L1/L2 distance.
- Normal consistency.
- Volumetric IoU for occupancy at declared voxel size.
- Watertightness/manifold and connected-component statistics.

Report observed and hidden regions separately. A model can look excellent overall by copying the
visible 80 percent while failing entirely on the hidden 20 percent.

### Evidence-preservation metrics

- observed-region drift in millimetres/centimetres;
- number/volume of free-space violations;
- change in reprojection depth on original cameras;
- silhouette false positives inside known background;
- boundary crack length between observed and generated regions.

### Uncertainty metrics

- negative log likelihood or proper scoring rule when available;
- expected calibration error for binned predicted uncertainty;
- risk-coverage curve: error after discarding the most uncertain generated regions;
- correlation between multi-sample disagreement and actual hidden-region error.

### Ablations

Compare: partial geometry only; +visibility; +image features; +semantics; deterministic vs diffusion;
one vs three/five samples; with/without hard observed clamping; synthetic-only vs domain fine-tuned.

## 9. Capture improvements that may outperform AI work

Before investing heavily in completion, improve the data:

- fly a 360-degree orbit instead of a partial arc where legally/safely possible;
- add a second orbit at a different elevation;
- maintain 70-85 percent overlap and actual translation/parallax;
- lock exposure, shutter, white balance, focus, and focal length;
- use higher shutter speed to reduce motion blur;
- avoid sky-dominated frames, specular glass, moving crowds/vehicles, and foliage;
- include scale bars or surveyed control points if measurements matter;
- capture camera calibration and rolling-shutter behavior.

No model can reliably infer a unique backside when capture can simply observe it. AI is appropriate
when access, safety, time, or historical footage makes additional views impossible.

## 10. Code integration in this repository

`RunConfig` now exposes `enable_surface_completion`, `surface_completion_provider`,
`surface_completion_model_path`, and `surface_completion_samples`. After observed reconstruction and
metric alignment, `PipelineRunner` builds a `SurfaceCompletionContext`. Quality gate failure selects
an unavailable provider and records `BLOCKED`; otherwise `ExternalSurfaceCompletionProvider` calls
the configured isolated inference executable.

The runtime receives `completion_request.json` and must return `completion/completed_mesh.ply` plus
`completion/uncertainty.json`. The backend validates both, writes `completion_report.json`, registers
the outputs, and exposes the predicted mesh as `visual_models.ai_completed_mesh` in the viewer
manifest. The React UI provides an `AI Predicted Surface` mode only when that declared mesh exists.
`visualModeMeasurementEligible` returns false for every non-evidence mode.

## 11. Deployment architecture

For development, run the model as a local CLI executable. For a larger GPU deployment, keep the same
request/response schema behind a job worker:

```text
FastAPI orchestration -> immutable run artifacts -> job queue
                                           |
                                           v
                                 GPU completion worker
                                           |
                                           v
                         predicted mesh + uncertainty + logs
                                           |
                                           v
                          artifact registry -> viewer manifest
```

Do not execute GPU inference inside the request handler. Use bounded concurrency, explicit timeouts,
checkpoint hashing, deterministic seeds where possible, and GPU memory/resource limits. Store model
license and training-data provenance with every run.

## 12. Milestones

### Phase 0 - two to four days

- freeze artifact/runtime schema;
- select one target object/scene category;
- collect 5-10 complete reference scans and normal partial trajectories;
- implement non-ML hole-fill and symmetry baselines;
- define hidden-region masks and evaluation scripts.

### Phase 1 - one to two weeks

- build partial TSDF/visibility exporter from COLMAP/OpenMVS outputs;
- train a deterministic 3D U-Net on synthetic partial-complete pairs;
- implement observed/free-space losses and mesh/provenance export;
- benchmark hidden-region Chamfer, F-score, IoU, and evidence drift.

### Phase 2 - two to four weeks

- add latent diffusion and K-sample hypotheses;
- add back-projected image features/semantics;
- calibrate uncertainty and integrate alternate hypotheses in the viewer;
- fine-tune on domain reference scans.

### Phase 3 - competition hardening

- test complete API/UI flow on clean machine;
- package local inference runtime and weights with license manifest;
- add GPU/CPU capability checks and honest blocked states;
- prepare side-by-side evidence/dense/predicted views and failure demos;
- publish exact quantitative results, not cherry-picked screenshots.

## 13. Go/no-go gates

Do not present the AI model as successful unless:

- it beats Poisson/symmetry baselines on hidden-region F-score/Chamfer;
- high-confidence observed surface drift remains below a declared tolerance;
- free-space violation stays below a declared volume/ray percentage;
- performance is measured on held-out physical objects/sites;
- uncertainty rises on the model's worst errors;
- every predicted vertex/face can be distinguished from observed evidence;
- the UI cannot measure the predicted layer.

## 14. Risks and mitigations

| Risk | Why it happens | Mitigation |
|---|---|---|
| Plausible but wrong backside | The inverse problem is underdetermined. | Multi-hypothesis output, uncertainty, observed/generated overlay, additional capture. |
| Domain shift | Indoor/CAD training differs from outdoor drone footage. | Drone-like synthetic rendering plus domain reference scans and held-out site split. |
| Scale error | Monocular priors are scale ambiguous. | Anchor to COLMAP/local ENU and known-distance control, never model-only scale. |
| Evidence overwritten | Prior prefers a canonical shape. | Hard observed clamping and large observed-preservation loss. |
| False surfaces in free space | Occupancy decoder ignores camera visibility. | Explicit free-space channel, ray loss, and reprojection validation. |
| Over-smoothed detail | Low voxel resolution/deterministic averaging. | Coarse-to-fine SDF refinement and report actual voxel size. |
| Texture/geometry confusion | Great novel views can hide poor geometry. | Evaluate geometry against scans; render evidence and prediction separately. |
| Dataset leakage | Adjacent frames/one object appear in train and test. | Split by physical object/site before generating views. |
| GPU/runtime failure | 3D networks are memory-heavy. | Sparse tensors, tiles, bounded sample count, worker isolation, blocked reports. |

## 15. Primary references

- COLMAP documentation: https://colmap.github.io/tutorial
- Curless and Levoy, weighted TSDF fusion: https://lightfield.stanford.edu/papers/volrange/
- NeuralRecon: https://arxiv.org/abs/2104.00681
- Atlas: https://arxiv.org/abs/2003.10432
- DiffComplete: https://arxiv.org/abs/2306.16329
- SDFusion: https://arxiv.org/abs/2212.04493
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- Tanks and Temples benchmark: https://tanksandtemples.org/
- DTU MVS dataset: https://roboimagedata.compute.dtu.dk/?page_id=36
- ScanNet: https://arxiv.org/abs/1702.04405
