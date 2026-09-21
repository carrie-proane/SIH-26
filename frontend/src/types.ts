export type RunStatus =
  | "QUEUED"
  | "INGESTING"
  | "PREPROCESSING"
  | "RECONSTRUCTING"
  | "REPORTING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

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
  status: "STARTED" | "COMPLETED" | "FAILED" | "CANCELLED";
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
  environment: Record<string, string | boolean | number | null>;
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
  last_heartbeat_at?: string | null;
  cancel_requested_at?: string | null;
  cancelled_at?: string | null;
  processing_started_at?: string | null;
  processing_completed_at?: string | null;
  stage_timings_s?: Record<string, number>;
  derived_from_run_id?: string | null;
  capability_profile_path?: string | null;
  effective_sparse_gpu?: boolean;
  selected_dense_provider?: "colmap" | "openmvs" | null;
}

export interface RunReadiness {
  schema_version: "1.0";
  run_id: string;
  stage: RunStatus;
  status: RunStatus;
  sparse_preview_ready: boolean;
  sparse_preview_missing: string[];
  quality_report_ready: boolean;
  dense_status: "AVAILABLE" | "FAILED_OR_UNAVAILABLE" | "NOT_READY_OR_NOT_REQUESTED";
  dense_visual_artifacts: string[];
  measurement_geometry: string | null;
  inferred_geometry_measurement_eligible: false;
  export_report_ready: boolean;
  export_report_url: string | null;
  export_status: Record<string, string>;
}

export interface ExportFile {
  relative_path: string;
  url: string;
  sha256: string;
  size_bytes: number;
  media_type: string;
}

export interface GeometryExport {
  status: "AVAILABLE_VALIDATED";
  format: "OBJ" | "LAS" | "GLB";
  variant: string;
  source_artifact: { path: string; sha256: string };
  geometry_provenance: "OBSERVED" | "DERIVED_OBSERVED_VISUAL" | "INFERRED" | "UNKNOWN";
  measurement_eligible: boolean;
  measurement_api_supported: false;
  coordinate_contract: Record<string, unknown>;
  validation: Record<string, unknown>;
  reused: boolean;
  files: ExportFile[];
  manifest_url: string;
}

export interface ExportFormatStatus {
  status: "AVAILABLE_VALIDATED" | "UNAVAILABLE" | "INVALID" | "INVALID_OR_UNSUPPORTED";
  reason?: string;
  exports?: GeometryExport[];
  failures?: Array<{ source_artifact_path: string; reason: string }>;
  download_files?: ExportFile[];
  artifact_path?: string;
  artifact_url?: string;
  artifact_sha256?: string;
  coordinate_frame?: string;
  geometry_provenance?: string;
  measurement_eligible?: boolean;
  validation?: Record<string, unknown>;
}

export interface ExportReadiness {
  schema_version: "2.0";
  run_id: string;
  exporter: { name: string; version: string };
  listed_formats_mandatory: "ORGANIZER_CLARIFICATION_REQUIRED";
  coordinate_contract: Record<string, unknown>;
  formats: Record<string, ExportFormatStatus>;
  available_validated_formats: string[];
  generated_files: ExportFile[];
  partial_success: boolean;
  measurement_eligibility_statement: string;
}

export interface MeasurementEndpointPayload {
  coordinates: [number, number, number];
  point_id?: number | null;
  face_id?: number | null;
}

export interface MeasurementCreatePayload {
  geometry_artifact_path: string;
  geometry_artifact_sha256: string;
  start: MeasurementEndpointPayload;
  end: MeasurementEndpointPayload;
  coordinate_frame?: string;
  units?: "m";
  measurement_kind?: "DISTANCE_3D" | "HORIZONTAL" | "VERTICAL" | "RELATIVE_DIMENSION";
  reference_value_m?: number | null;
  reference_method?: string | null;
  reference_evidence?: string | null;
  reference_role?: "NONE" | "SCALE_CONTROL" | "HELD_OUT_EVALUATION";
}

export interface MeasurementRecordPayload extends MeasurementCreatePayload {
  measurement_id: string;
  run_id: string;
  created_at: string;
  backend_distance_m: number;
  geometry_provenance: "OBSERVED" | "INFERRED" | "UNKNOWN";
  measurement_eligible: boolean;
  eligibility_reason: string;
  legacy_manual_reconstructed_value: boolean;
}

export interface MeasurementListResponse {
  schema_version: "1.0";
  run_id: string;
  measurements: MeasurementRecordPayload[];
  evaluation: Record<string, unknown>;
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
  reconstruction_status?:
    | "NOT_SELECTED"
    | "REGISTERED_SELECTED_COMPONENT"
    | "NOT_REGISTERED_IN_SELECTED_COMPONENT";
  reconstruction_exclusion_reason?: string | null;
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
