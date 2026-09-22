import { parseCsv } from "./csv";
import { parsePointConfidence } from "./confidence";
import type {
  CameraPose,
  ExportReadiness,
  Keyframe,
  MeasurementCreatePayload,
  MeasurementListResponse,
  MeasurementRecordPayload,
  ProjectListResponse,
  ProjectManifest,
  ProvenanceOrigin,
  QualityReport,
  RunRecord,
  RunReadiness,
  RunListResponse,
  RunConfiguration,
  ViewerBundle,
  ViewerManifest,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export function resolveAssetUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(resolveAssetUrl(path), init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      message = payload.detail ?? message;
    } catch {
      // The status line remains the most honest error available.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export async function uploadProject(input: {
  name: string;
  description: string;
  video: File;
  telemetry: File;
  videoOrigin?: ProvenanceOrigin;
  telemetryOrigin?: ProvenanceOrigin;
}): Promise<ProjectManifest> {
  const body = new FormData();
  body.set("name", input.name);
  body.set("description", input.description);
  body.set("data_classification", "PUBLIC_DEMO");
  body.set("video", input.video);
  body.set("telemetry", input.telemetry);
  body.set("video_origin", input.videoOrigin ?? "UNKNOWN");
  body.set("telemetry_origin", input.telemetryOrigin ?? "UNKNOWN");
  return request<ProjectManifest>("/api/projects", { method: "POST", body });
}

export function getProjects(signal?: AbortSignal): Promise<ProjectListResponse> {
  return request<ProjectListResponse>("/api/projects", { signal });
}

export function getProject(projectId: string, signal?: AbortSignal): Promise<ProjectManifest> {
  return request<ProjectManifest>(`/api/projects/${projectId}`, { signal });
}

export function getProjectRuns(projectId: string, signal?: AbortSignal): Promise<RunListResponse> {
  return request<RunListResponse>(`/api/projects/${projectId}/runs`, { signal });
}

export async function startRun(
  projectId: string,
  config: RunConfiguration | Record<string, unknown>,
): Promise<RunRecord> {
  return request<RunRecord>(`/api/projects/${projectId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export function getRun(runId: string, signal?: AbortSignal): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${runId}`, { signal });
}

export function getRunReadiness(runId: string, signal?: AbortSignal): Promise<RunReadiness> {
  return request<RunReadiness>(`/api/runs/${runId}/readiness`, { signal });
}

export function rerunRun(
  runId: string,
  config: RunConfiguration | Record<string, unknown>,
): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${runId}/rerun`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export function resumeRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${runId}/resume`, { method: "POST" });
}

export function getExports(runId: string, signal?: AbortSignal): Promise<ExportReadiness> {
  return request<ExportReadiness>(`/api/runs/${runId}/exports`, { signal });
}

export function createExports(
  runId: string,
  includeCompletedGeometry = false,
): Promise<ExportReadiness> {
  return request<ExportReadiness>(`/api/runs/${runId}/exports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_completed_geometry: includeCompletedGeometry }),
  });
}

export function getSegmentationStatus(
  runId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/runs/${runId}/segmentation`, { signal });
}

export function getCoverageStatus(
  runId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/runs/${runId}/coverage`, { signal });
}

export function getCompletionStatus(
  runId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/runs/${runId}/completion`, { signal });
}

export function requestCompletion(
  runId: string,
  payload: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/runs/${runId}/completion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function createMeasurement(
  runId: string,
  measurement: MeasurementCreatePayload,
): Promise<MeasurementRecordPayload> {
  return request<MeasurementRecordPayload>(`/api/runs/${runId}/measurements`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(measurement),
  });
}

export function getMeasurements(
  runId: string,
  signal?: AbortSignal,
): Promise<MeasurementListResponse> {
  return request<MeasurementListResponse>(`/api/runs/${runId}/measurements`, { signal });
}

export function cancelRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${runId}/cancel`, { method: "POST" });
}

export function getViewerManifest(runId: string, signal?: AbortSignal): Promise<ViewerManifest> {
  return request<ViewerManifest>(`/api/runs/${runId}/viewer-manifest`, { signal });
}

export async function pollRun(
  runId: string,
  onUpdate: (record: RunRecord) => void,
  signal?: AbortSignal,
): Promise<RunRecord> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("Run polling cancelled", "AbortError");
    const record = await getRun(runId, signal);
    onUpdate(record);
    if (["COMPLETED", "FAILED", "CANCELLED"].includes(record.status)) return record;
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(resolve, 750);
      signal?.addEventListener(
        "abort",
        () => {
          window.clearTimeout(timeout);
          reject(new DOMException("Run polling cancelled", "AbortError"));
        },
        { once: true },
      );
    });
  }
}

function numberFrom(row: Record<string, string>, names: string[], fallback = 0): number {
  const raw = names.map((name) => row[name]).find((value) => value !== undefined && value !== "");
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export async function loadViewerBundle(
  manifest: ViewerManifest,
  signal?: AbortSignal,
): Promise<ViewerBundle> {
  const [cameraResponse, keyframeResponse, qualityResponse, ingestResponse, confidenceResponse] = await Promise.all([
    fetch(resolveAssetUrl(manifest.camera_path.url), { signal }),
    fetch(resolveAssetUrl(manifest.selected_frames.url), { signal }),
    manifest.quality_report_url
      ? fetch(resolveAssetUrl(manifest.quality_report_url), { signal })
      : Promise.resolve(null),
    manifest.ingest_report_url
      ? fetch(resolveAssetUrl(manifest.ingest_report_url), { signal })
      : Promise.resolve(null),
    manifest.confidence.available && manifest.confidence.url
      ? fetch(resolveAssetUrl(manifest.confidence.url), { signal })
      : Promise.resolve(null),
  ]);
  for (const response of [cameraResponse, keyframeResponse, qualityResponse, ingestResponse]) {
    if (response === null) continue;
    if (!response.ok) throw new ApiError(`Unable to load ${response.url}`, response.status);
  }

  const cameraRows = parseCsv(await cameraResponse.text());
  const cameraPoses: CameraPose[] = cameraRows.map((row, index) => ({
    frameIndex: numberFrom(row, ["frame_index", "image_id"], index),
    timestampS: numberFrom(row, ["timestamp_s"], index),
    x: numberFrom(row, ["x_m", "sfm_x"]),
    y: numberFrom(row, ["y_m", "sfm_y"]),
    z: numberFrom(row, ["z_m", "sfm_z"]),
    source: row.source,
    imageName: row.image_name,
  }));
  const keyframePayload = (await keyframeResponse.json()) as { frames?: Keyframe[] } | Keyframe[];
  const keyframes = Array.isArray(keyframePayload)
    ? keyframePayload
    : (keyframePayload.frames ?? []);
  const quality = qualityResponse
    ? ((await qualityResponse.json()) as QualityReport)
    : ({
        schema_version: "1.0",
        project_id: manifest.project_id,
        run_id: manifest.run_id,
        status: "NOT_READY_SPARSE_PREVIEW",
        synthetic_fixture: manifest.synthetic_fixture,
        source_provenance: manifest.source_provenance,
        video_origin: manifest.video_origin,
        telemetry_origin: manifest.telemetry_origin,
        genuine_real_evidence: manifest.genuine_real_evidence,
        metrics: {},
        warnings: [
          {
            code: "QUALITY_REPORT_PENDING",
            message: "Sparse evidence is available while final quality reporting is still pending.",
          },
        ],
        limitations: ["Final quality, dense, export, and target evaluations are not ready."],
        confidence_artifact: {
          available: false,
          measurement_confidence_available: false,
          reason: "Final confidence validation is pending.",
          contract: {},
        },
      } satisfies QualityReport);
  const ingest = ingestResponse ? await ingestResponse.json() : null;
  const pointConfidence =
    confidenceResponse?.ok
      ? parsePointConfidence(await confidenceResponse.json())
      : null;
  return { manifest, cameraPoses, keyframes, quality, ingest, pointConfidence };
}

export async function loadOfflineFixture(): Promise<ViewerBundle> {
  const manifest = await request<ViewerManifest>("/demo/viewer-manifest.json");
  return loadViewerBundle(manifest);
}

export async function createSyntheticDemo(): Promise<{ project: ProjectManifest; run: RunRecord }> {
  const video = new File(["SIH26158_SYNTHETIC_DEMO_FIXTURE"], "synthetic_demo.mp4", {
    type: "video/mp4",
  });
  const telemetry = new File(
    ["timestamp_s,lat,lon,alt_m\n0,18.5204,73.8567,2\n4.5,18.52045,73.85675,2\n"],
    "synthetic_telemetry.csv",
    { type: "text/csv" },
  );
  const project = await uploadProject({
    name: "Synthetic contract smoke test",
    description: "UI/API fixture only; never reconstruction evidence.",
    video,
    telemetry,
    videoOrigin: "SYNTHETIC",
    telemetryOrigin: "SYNTHETIC",
  });
  const run = await startRun(project.project_id, {
    execution_mode: "SYNTHETIC_DEMO",
    profile: "smoke",
    matcher: "SIFT",
    known_distance_m: 10,
    measured_distance_m: 10.6,
  });
  return { project, run };
}
