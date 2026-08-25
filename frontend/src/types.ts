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

export interface InputAsset {
  role: string;
  original_name: string;
  relative_path: string;
  size_bytes: number;
  sha256: string;
  media_type?: string | null;
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
  cloud: {
    url: string;
    format: "PLY";
    coordinate_frame: string;
  };
  camera_path: {
    url: string;
    coordinate_frame: string;
  };
  selected_frames: { url: string };
  confidence_legend: ConfidenceLegendItem[];
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
  ai_overlay?: {
    available: boolean;
    label: "AI_ASSISTED_NOT_MEASURABLE";
    measurement: "DISABLED";
    url?: string;
    reason?: string;
    model?: string;
  };
}

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
  exposure_score?: number;
  redundancy_score?: number;
  dynamic_mask_fraction?: number;
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
    known_distance?: Record<string, unknown>;
    coverage?: Record<string, unknown>;
  };
  warnings: Array<{ code: string; message: string }>;
  limitations: string[];
  confidence_contract: ConfidenceLabel[];
}

export interface ViewerBundle {
  manifest: ViewerManifest;
  cameraPoses: CameraPose[];
  keyframes: Keyframe[];
  quality: QualityReport;
  ingest: IngestReport | null;
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
}

export interface MeasurementResult {
  distanceM: number | null;
  labels: ConfidenceLabel[];
  status: "IDLE" | "SELECTING" | "ALLOWED" | "CAUTION" | "CONFIRM" | "BLOCKED";
  message: string;
}
