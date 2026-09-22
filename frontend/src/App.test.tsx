import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./components/PointCloudViewer", () => ({
  PointCloudViewer: ({ onMeasurementChange }: { onMeasurementChange: (value: unknown) => void }) => <div>
    <div data-testid="point-cloud-viewer">WebGL viewer</div>
    <button type="button" onClick={() => onMeasurementChange({ distanceM: 5, labels: ["OBSERVED_HIGH"], status: "ALLOWED", message: "fixture selection", endpoints: [{ coordinates: [0, 0, 0], point_id: 0 }, { coordinates: [3, 4, 0], point_id: 1 }] })}>Select fixture points</button>
  </div>,
}));

const manifest = {
  schema_version: "1.0",
  project_id: "prj_fixture",
  run_id: "run_fixture",
  synthetic_fixture: true,
  source_provenance: "SYNTHETIC",
  video_origin: "SYNTHETIC",
  telemetry_origin: "SYNTHETIC",
  genuine_real_evidence: false,
  cloud: {
    url: "/demo/cloud.ply",
    format: "PLY",
    coordinate_frame: "LOCAL_ENU_METRES",
    color_mode: "PHOTOGRAPHIC_RGB",
    color_mode_label: "Photographic RGB",
  },
  camera_path: { url: "/demo/camera.csv", coordinate_frame: "LOCAL_ENU_METRES" },
  selected_frames: { url: "/demo/keyframes.json" },
  confidence_legend: [
    { label: "OBSERVED_HIGH", color: "#20bf6b", measurement: "ALLOWED" },
    { label: "AI_ASSISTED_NOT_MEASURABLE", color: "#a855f7", measurement: "DISABLED" },
  ],
  confidence: {
    available: false,
    reason: "Confidence unavailable for this run",
    contract: {
      schema_version: "1.0",
      supported_artifact: "point_confidence.json",
      point_order: "PLY_VERTEX_ORDER",
      required_fields: [
        "point_id",
        "supporting_views",
        "track_length",
        "reprojection_error",
        "triangulation_angle",
        "confidence_class",
      ],
      valid_classes: [
        "OBSERVED_HIGH",
        "OBSERVED_MEDIUM",
        "OBSERVED_LOW",
        "AI_ASSISTED_NOT_MEASURABLE",
        "UNSEEN",
      ],
      rgb_derivation_prohibited: true,
    },
  },
  measurement_reference: {
    label: "Independent known distance",
    reference_m: 10,
    measured_m: 10.6,
    percent_error: 6,
    passes_10_percent_gate: true,
    synthetic_fixture: true,
  },
  quality_report_url: "/demo/quality.json",
  ai_overlay: {
    available: false,
    label: "AI_ASSISTED_NOT_MEASURABLE",
    measurement: "DISABLED",
    reason: "No overlay declared.",
  },
};

const quality = {
  schema_version: "1.0",
  project_id: "prj_fixture",
  run_id: "run_fixture",
  synthetic_fixture: true,
  source_provenance: "SYNTHETIC",
  video_origin: "SYNTHETIC",
  telemetry_origin: "SYNTHETIC",
  genuine_real_evidence: false,
  metrics: {
    eligible_frames: 10,
    registered_frames: 9,
    registered_frame_rate: 0.9,
    median_reprojection_error_px: 0.9,
    reprojection_gate_1_5_px: true,
    runtime_s: 0.1,
    telemetry_sync: {},
  },
  warnings: [{ code: "SYNTHETIC_FIXTURE", message: "Not reconstruction proof." }],
  limitations: ["Unseen surfaces are not reconstructed."],
  confidence_artifact: {
    available: false,
    measurement_confidence_available: false,
    reason: "Confidence unavailable for this run",
    contract: { rgb_derivation_prohibited: true },
  },
};

function response(body: unknown, contentType = "application/json"): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": contentType },
  });
}

describe("operator application", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/projects")) return Promise.resolve(response({ schema_version: "1.0", projects: [] }));
        if (url.endsWith("viewer-manifest.json")) return Promise.resolve(response(manifest));
        if (url.endsWith("camera.csv")) {
          return Promise.resolve(
            response("frame_index,timestamp_s,x_m,y_m,z_m\n0,0,0,0,2\n", "text/csv"),
          );
        }
        if (url.endsWith("keyframes.json")) {
          return Promise.resolve(
            response({ frames: [{ frame_index: 0, timestamp_s: 0, selected: true }] }),
          );
        }
        if (url.endsWith("quality.json")) return Promise.resolve(response(quality));
        return Promise.reject(new Error(`Unexpected request: ${url}`));
      }),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it("starts on the honest upload and demo choice", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /see what the camera proved/i })).toBeVisible();
    expect(screen.getByRole("button", { name: /run api smoke fixture/i })).toBeEnabled();
    expect(screen.getByText(/advanced preprocessing options/i)).toBeVisible();
    expect(screen.getByPlaceholderText(/automatic scored preprocessing/i)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /reconstruction target/i })).toHaveValue(
      "FULL_SCENE",
    );
    expect(screen.getByRole("combobox", { name: /scene-aware masking/i })).toHaveValue("OFF");
    expect(screen.getByPlaceholderText(/nothing is downloaded automatically/i)).toBeVisible();
  });

  it("loads the offline manifest through the complete operator workspace", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /open offline ui fixture/i }));

    await waitFor(() => expect(screen.getByTestId("point-cloud-viewer")).toBeVisible());
    expect(screen.getByText(/ui \/ orchestration fixture/i)).toBeVisible();
    expect(screen.getByText("90%")).toBeVisible();
    expect(screen.getByText(/provenance: synthetic/i)).toBeInTheDocument();
    expect(screen.getByText(/confidence unavailable for this run/i)).toBeVisible();
    expect(screen.getByRole("button", { name: /photographic rgb/i })).toBeVisible();
    expect(screen.getByRole("button", { name: /^observed$/i })).toHaveClass("is-active");
    expect(screen.getByRole("button", { name: /^inferred$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^both$/i })).toBeDisabled();
    const texturedButton = screen.getByRole("button", { name: /textured model/i });
    const photorealButton = screen.getByRole("button", { name: /photoreal view/i });
    expect(texturedButton).toBeDisabled();
    expect(texturedButton).toHaveAttribute("title", expect.stringMatching(/not declared/i));
    expect(photorealButton).toBeDisabled();
    expect(photorealButton).toHaveAttribute("title", expect.stringMatching(/not declared/i));
    expect(screen.getByRole("button", { name: /measure/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /ai depth/i })).toBeDisabled();
    expect(screen.getByText(/measurement: disabled/i)).toBeVisible();
  });

  it("recovers a sparse preview, persisted measurements, and unavailable exports from the URL", async () => {
    const project = { project_id: "prj_live", name: "Server capture", description: "", data_classification: "INTERNAL", created_at: "2026-09-22T00:00:00Z", immutable: true, assets: [], warnings: [], source_provenance: "REAL", video_origin: "REAL", telemetry_origin: "REAL" };
    const run = { project_id: "prj_live", run_id: "run_live", stage: "RECONSTRUCTING", status: "RECONSTRUCTING", progress: 60, created_at: "2026-09-22T00:00:00Z", updated_at: "2026-09-22T00:01:00Z", config_version: "1.0", config: { execution_mode: "COLMAP", matcher: "SIFT" }, environment: {}, failure_reason: null, events: [], artifacts: [], synthetic_fixture: false, source_provenance: "REAL", video_origin: "REAL", telemetry_origin: "REAL", telemetry_offset_s: 0, offset_source: "automatic", rmse_before_m: null, rmse_after_m: null, requested_matcher: "SIFT", executed_matcher: "SIFT" };
    const liveManifest = { ...manifest, project_id: "prj_live", run_id: "run_live", preview_mode: "SPARSE_EARLY", quality_report_ready: false, quality_report_url: null, synthetic_fixture: false, source_provenance: "REAL", video_origin: "REAL", telemetry_origin: "REAL", genuine_real_evidence: true, cloud: { ...manifest.cloud, url: "/api/runs/run_live/artifacts/sparse/sparse_local.ply", relative_path: "sparse/sparse_local.ply", sha256: "a".repeat(64), measurement_eligible: true }, visual_models: { evidence_cloud: { available: true, url: "/api/runs/run_live/artifacts/sparse/sparse_local.ply", format: "PLY", coordinate_frame: "LOCAL_ENU_METRES", measurement_eligible: true } }, measurement_reference: { ...manifest.measurement_reference, reference_m: null, measured_m: null, percent_error: null, passes_10_percent_gate: null, synthetic_fixture: false }, completion: { status: "NOT_RUN", available: false, reason: "No completion artifacts declared.", inferred_geometry: { available: false, provenance: "INFERRED_NOT_PRODUCED", measurement_eligible: false } } };
    const persisted = { measurement_id: "measurement_1", run_id: "run_live", created_at: "2026-09-22T00:02:00Z", geometry_artifact_path: "sparse/sparse_local.ply", geometry_artifact_sha256: "a".repeat(64), start: { coordinates: [0,0,0], point_id: 0 }, end: { coordinates: [3,4,0], point_id: 1 }, coordinate_frame: "LOCAL_ENU_METRES", units: "m", measurement_kind: "DISTANCE_3D", reference_role: "NONE", backend_distance_m: 5, geometry_provenance: "OBSERVED", measurement_eligible: true, eligibility_reason: "eligible", legacy_manual_reconstructed_value: false };
    window.history.replaceState({}, "", "/?project=prj_live&run=run_live");
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/runs/run_live") && (!init?.method || init.method === "GET")) return Promise.resolve(response(run));
      if (url.endsWith("/api/projects/prj_live")) return Promise.resolve(response(project));
      if (url.endsWith("/api/projects")) return Promise.resolve(response({ schema_version: "1.0", projects: [project] }));
      if (url.endsWith("/api/projects/prj_live/runs")) return Promise.resolve(response({ schema_version: "1.0", project_id: "prj_live", runs: [run] }));
      if (url.endsWith("/api/runs/run_live/readiness")) return Promise.resolve(response({ schema_version: "1.0", run_id: "run_live", stage: "RECONSTRUCTING", status: "RECONSTRUCTING", sparse_preview_ready: true, sparse_preview_missing: [], quality_report_ready: false, dense_status: "FAILED_OR_UNAVAILABLE", dense_visual_artifacts: [], measurement_geometry: "sparse/sparse_local.ply", inferred_geometry_measurement_eligible: false, export_report_ready: true, export_report_url: "/api/runs/run_live/exports", export_status: {} }));
      if (url.endsWith("/api/runs/run_live/viewer-manifest")) return Promise.resolve(response(liveManifest));
      if (url.endsWith("camera.csv")) return Promise.resolve(response("frame_index,timestamp_s,x_m,y_m,z_m\n0,0,0,0,2\n", "text/csv"));
      if (url.endsWith("keyframes.json")) return Promise.resolve(response({ frames: [{ frame_index: 0, timestamp_s: 0, selected: true, quality_eligible: true, reconstruction_status: "REGISTERED_SELECTED_COMPONENT" }] }));
      if (url.endsWith("/api/runs/run_live/measurements") && init?.method === "POST") return Promise.resolve(new Response(JSON.stringify(persisted), { status: 201, headers: { "Content-Type": "application/json" } }));
      if (url.endsWith("/api/runs/run_live/measurements")) return Promise.resolve(response({ schema_version: "1.0", run_id: "run_live", measurements: [persisted], evaluation: {} }));
      if (url.endsWith("/api/runs/run_live/exports")) return Promise.resolve(response({ schema_version: "2.0", run_id: "run_live", exporter: { name: "fixture", version: "1" }, listed_formats_mandatory: "ORGANIZER_CLARIFICATION_REQUIRED", coordinate_contract: {}, formats: { OBJ: { status: "UNAVAILABLE", reason: "No mesh faces exist." }, GLB_GLTF: { status: "UNAVAILABLE", reason: "No mesh faces exist." }, LAS: { status: "UNAVAILABLE", reason: "Pending terminal export preparation." }, GEOTIFF: { status: "UNAVAILABLE", reason: "No raster product." }, FBX: { status: "UNAVAILABLE", reason: "Not implemented." } }, available_validated_formats: [], generated_files: [], partial_success: false, measurement_eligibility_statement: "unchanged" }));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    }));

    render(<App />);
    await waitFor(() => expect(screen.getByText(/early sparse preview/i)).toBeVisible());
    expect(screen.getByText(/failed_or_unavailable/i)).toBeVisible();
    expect(await screen.findByText("5.000 m")).toBeVisible();
    expect(screen.getAllByText(/no mesh faces exist/i)).toHaveLength(2);
    expect(screen.getByText(/not evaluated: official statistic is unspecified/i)).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: /select fixture points/i }));
    await userEvent.click(screen.getByRole("button", { name: /save backend measurement/i }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/measurements$/), expect.objectContaining({ method: "POST" })));
  });

  it("ignores a stale deep-link response after the operator changes selection", async () => {
    let releaseRun!: (value: Response) => void;
    const delayedRun = new Promise<Response>((resolve) => { releaseRun = resolve; });
    const staleRun = { project_id: "prj_stale", run_id: "run_stale", stage: "COMPLETED", status: "COMPLETED", progress: 100, created_at: "2026-09-22T00:00:00Z", updated_at: "2026-09-22T00:00:00Z", config_version: "1.0", config: { execution_mode: "COLMAP", matcher: "SIFT" }, environment: {}, failure_reason: null, events: [], artifacts: [], synthetic_fixture: false, source_provenance: "REAL", video_origin: "REAL", telemetry_origin: "REAL", telemetry_offset_s: 0, offset_source: "automatic", rmse_before_m: null, rmse_after_m: null };
    window.history.replaceState({}, "", "/?project=prj_stale&run=run_stale");
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/runs/run_stale")) return delayedRun;
      if (url.endsWith("/api/projects")) return Promise.resolve(response({ schema_version: "1.0", projects: [] }));
      return Promise.reject(new Error(`Stale selection unexpectedly requested: ${url}`));
    }));

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /trace3d home/i }));
    releaseRun(response(staleRun));

    await waitFor(() => expect(screen.getByRole("heading", { name: /see what the camera proved/i })).toBeVisible());
    expect(screen.queryByText("run_stale")).not.toBeInTheDocument();
    expect(window.location.search).toBe("");
  });
});
