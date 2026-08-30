export type RunStatus =
  | "QUEUED"
  | "INGESTING"
  | "PREPROCESSING"
  | "RECONSTRUCTING"
  | "REPORTING"
  | "COMPLETED"
  | "FAILED";

export type ConfidenceLabel =
  | "OBSERVED_HIGH"
  | "OBSERVED_MEDIUM"
  | "OBSERVED_LOW"
  | "AI_ASSISTED_NOT_MEASURABLE"
  | "UNSEEN";

export type ProvenanceOrigin = "REAL" | "SYNTHETIC" | "DERIVED" | "UNKNOWN";

export interface InputAsset {
  role: string;
  original_name: string;
  relative_path: string;
  size_bytes: number;
  sha256: string;
  media_type?: string | null;
  origin?: ProvenanceOrigin;
}

export interface ProjectManifest {
  project_id: string;
  name: string;
  description: string;
  data_classification: string;
  created_at: string;
  immutable: boolean;
  assets: InputAsset[];
  warnings: Array<{ code?: string; message?: string }>;
  source_provenance: ProvenanceOrigin;
  video_origin: ProvenanceOrigin;
  telemetry_origin: ProvenanceOrigin;
}

export interface ArtifactEntry {
  name: string;
  relative_path: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  url: string;
  created_at: string;
}

export interface StageEvent {
  stage: RunStatus;
  status: "STARTED" | "COMPLETED" | "FAILED";
  timestamp: string;
  progress: number;
  message: string;
}

export interface RunRecord {
  project_id: string;
  run_id: string;
  stage: RunStatus;
  status: RunStatus;
  progress: number;
  created_at: string;
  updated_at: string;
  config_version: string;
  config: Record<string, unknown>;
  environment: Record<string, string | null>;
  failure_reason: string | null;
  events: StageEvent[];
  artifacts: ArtifactEntry[];
  synthetic_fixture: boolean;
  source_provenance: ProvenanceOrigin;
  video_origin: ProvenanceOrigin;
  telemetry_origin: ProvenanceOrigin;
  telemetry_offset_s: number;
  offset_source: "automatic" | "manual" | "calibrated" | "not_applicable";
  rmse_before_m: number | null;
  rmse_after_m: number | null;
}

export interface ConfidenceLegendItem {
  label: ConfidenceLabel;
  color: string;
  measurement: "ALLOWED" | "CAUTION" | "CONFIRM" | "DISABLED";
}

export interface ViewerManifest {
  schema_version: string;
  project_id: string;
  run_id: string;
  stage?: RunStatus;
  status?: RunStatus;
  synthetic_fixture: boolean;
  source_provenance: ProvenanceOrigin;
  video_origin: ProvenanceOrigin;
  telemetry_origin: ProvenanceOrigin;
  genuine_real_evidence: boolean;
  cloud: {
    url: string;
    format: "PLY";
    coordinate_frame: string;
    color_mode: "PHOTOGRAPHIC_RGB";
    color_mode_label: "Photographic RGB";
  };
  visual_models?: {
    evidence_cloud: VisualModel;
    dense_cloud: VisualModel;
    textured_mesh: VisualModel & {
      texture_urls?: string[];
      texture_validity?: TextureValidityContract | null;
    };
    gaussian_splat: VisualModel;
    dense_report_url?: string | null;
  };
  camera_path: {
    url: string;
    coordinate_frame: string;
  };
  selected_frames: { url: string };
  confidence_legend: ConfidenceLegendItem[];
  confidence: {
    available: boolean;
    url?: string | null;
    format?: "POINT_CONFIDENCE_JSON" | null;
    reason: string;
    contract: {
      schema_version: string;
      supported_artifact: string;
      point_order: "PLY_VERTEX_ORDER";
      required_fields: string[];
      valid_classes: ConfidenceLabel[];
      rgb_derivation_prohibited: true;
    };
  };
  measurement_reference: {
    label: string;
    status?: string;
    reference_m: number | null;
    measured_m: number | null;
    percent_error: number | null;
    passes_10_percent_gate?: boolean | null;
    synthetic_fixture: boolean;
  };
  quality_report_url: string;
  ingest_report_url?: string;
  scene_policy?: {
    target: "FULL_SCENE" | "PRIMARY_SUBJECT";
    masking_mode: "OFF" | "AUTO" | "REQUIRED";
    analysis_url?: string | null;
    segmentation_report_url?: string | null;
  };
  ai_overlay?: {
    available: boolean;
    label: "AI_ASSISTED_NOT_MEASURABLE";
    measurement: "DISABLED";
    url?: string;
    reason?: string;
    model?: string;
  };
}

export interface VisualModel {
  available: boolean;
  url?: string | null;
  format?: "PLY" | "GLB" | "SPLAT" | null;
  coordinate_frame?: string | null;
  measurement_eligible: boolean;
  default?: boolean;
  statement?: string;
}

export interface TextureValidityContract {
  strategy: "ATLAS_EMPTY_COLOR";
  empty_rgb: [number, number, number];
  empty_tolerance: number;
  minimum_supported_samples: number;
}

export type VisualMode = "EVIDENCE" | "TEXTURED" | "PHOTOREAL";

export interface Keyframe {
  frame_index: number;
  timestamp_s: number;
  selected: boolean;
  image_name?: string;
  filename?: string;
  image_url?: string;
  mask_url?: string;
  depth_overlay_url?: string;
  source?: string;
  blur_score?: number;
  laplacian_variance?: number;
  quality_eligible?: boolean;
  quality_rejection_reasons?: string;
  exposure_score?: number;
  redundancy_score?: number;
  dynamic_mask_fraction?: number;
  mask_semantics?: "NONZERO_IS_EXCLUDED";
  selected_automatically?: boolean;
  override?: "NONE" | "FORCE_INCLUDE" | "FORCE_EXCLUDE";
  confidence?: ConfidenceLabel;
}

export interface CameraPose {
  frameIndex: number;
  timestampS: number;
  x: number;
  y: number;
  z: number;
  source?: string;
  imageName?: string;
}

export interface QualityReport {
  schema_version: string;
  project_id: string;
  run_id: string;
  synthetic_fixture: boolean;
  source_provenance: ProvenanceOrigin;
  video_origin: ProvenanceOrigin;
  telemetry_origin: ProvenanceOrigin;
  genuine_real_evidence: boolean;
  metrics: {
    eligible_frames?: number;
    registered_frames?: number;
    registered_frame_rate?: number;
    registered_frame_gate_80_percent?: boolean;
    median_reprojection_error_px?: number;
    p95_reprojection_error_px?: number;
    reprojection_gate_1_5_px?: boolean;
    runtime_s?: number;
    metric_alignment?: Record<string, unknown>;
    telemetry_sync?: Record<string, unknown>;
    known_distance?: Record<string, unknown>;
    coverage?: Record<string, unknown>;
    frame_quality_gate?: Record<string, unknown>;
    reconstruction_policy?: Record<string, unknown>;
  };
  warnings: Array<{ code: string; message: string }>;
  limitations: string[];
  confidence_artifact: {
    available: boolean;
    measurement_confidence_available: boolean;
    reason: string | null;
    contract: Record<string, unknown>;
  };
}

export interface PointConfidenceRecord {
  point_id: number;
  supporting_views: number;
  track_length: number;
  reprojection_error: number;
  triangulation_angle: number;
  confidence_class: ConfidenceLabel;
}

export interface PointConfidenceArtifact {
  schema_version: "1.0";
  point_order: "PLY_VERTEX_ORDER";
  points: PointConfidenceRecord[];
}

export interface ViewerBundle {
  manifest: ViewerManifest;
  cameraPoses: CameraPose[];
  keyframes: Keyframe[];
  quality: QualityReport;
  ingest: IngestReport | null;
  pointConfidence: PointConfidenceArtifact | null;
}

export interface IngestReport {
  project_id: string;
  run_id: string;
  synthetic_fixture?: boolean;
  input_assets?: InputAsset[];
  video_probe?: {
    format?: Record<string, unknown>;
    streams?: Array<Record<string, unknown>>;
  };
  warnings?: Array<{ code: string; message: string }>;
  telemetry_normalization?: Record<string, unknown>;
  source_provenance?: ProvenanceOrigin;
  video_origin?: ProvenanceOrigin;
  telemetry_origin?: ProvenanceOrigin;
  genuine_real_evidence?: boolean;
}

export interface MeasurementResult {
  distanceM: number | null;
  labels: ConfidenceLabel[];
  status: "IDLE" | "SELECTING" | "ALLOWED" | "CAUTION" | "CONFIRM" | "BLOCKED";
  message: string;
}
