import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./components/PointCloudViewer", () => ({
  PointCloudViewer: () => <div data-testid="point-cloud-viewer">WebGL viewer</div>,
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
    expect(screen.getByRole("combobox", { name: /scene-aware masking/i })).toHaveValue("AUTO");
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
    expect(screen.getByRole("button", { name: /evidence cloud/i })).toHaveClass("is-active");
    const texturedButton = screen.getByRole("button", { name: /textured model/i });
    const photorealButton = screen.getByRole("button", { name: /photoreal view/i });
    expect(texturedButton).toBeDisabled();
    expect(texturedButton).toHaveAttribute("title", expect.stringMatching(/not declared/i));
    expect(photorealButton).toBeDisabled();
    expect(photorealButton).toHaveAttribute("title", expect.stringMatching(/not declared/i));
    expect(screen.getByRole("button", { name: /measure/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /ai depth/i })).toBeDisabled();
    expect(screen.getByText(/measurement: disabled/i)).toBeVisible();
  });
});
