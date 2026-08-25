import { parseCsv } from "./csv";
import type {
  CameraPose,
  Keyframe,
  ProjectManifest,
  QualityReport,
  RunRecord,
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
}): Promise<ProjectManifest> {
  const body = new FormData();
  body.set("name", input.name);
  body.set("description", input.description);
  body.set("data_classification", "PUBLIC_DEMO");
  body.set("video", input.video);
  body.set("telemetry", input.telemetry);
  return request<ProjectManifest>("/api/projects", { method: "POST", body });
}

export async function startRun(
  projectId: string,
  config: Record<string, unknown>,
): Promise<RunRecord> {
  return request<RunRecord>(`/api/projects/${projectId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export function getRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${runId}`);
}

export function getViewerManifest(runId: string): Promise<ViewerManifest> {
  return request<ViewerManifest>(`/api/runs/${runId}/viewer-manifest`);
}

export async function pollRun(
  runId: string,
  onUpdate: (record: RunRecord) => void,
  signal?: AbortSignal,
): Promise<RunRecord> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("Run polling cancelled", "AbortError");
    const record = await getRun(runId);
    onUpdate(record);
    if (record.status === "COMPLETED" || record.status === "FAILED") return record;
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
}

function numberFrom(row: Record<string, string>, names: string[], fallback = 0): number {
  const raw = names.map((name) => row[name]).find((value) => value !== undefined && value !== "");
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export async function loadViewerBundle(manifest: ViewerManifest): Promise<ViewerBundle> {
  const [cameraResponse, keyframeResponse, qualityResponse, ingestResponse] = await Promise.all([
    fetch(resolveAssetUrl(manifest.camera_path.url)),
    fetch(resolveAssetUrl(manifest.selected_frames.url)),
    fetch(resolveAssetUrl(manifest.quality_report_url)),
    manifest.ingest_report_url
      ? fetch(resolveAssetUrl(manifest.ingest_report_url))
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
  const quality = (await qualityResponse.json()) as QualityReport;
  const ingest = ingestResponse ? await ingestResponse.json() : null;
  return { manifest, cameraPoses, keyframes, quality, ingest };
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
