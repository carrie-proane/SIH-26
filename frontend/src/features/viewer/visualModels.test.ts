import { describe, expect, it } from "vitest";

import type { ViewerManifest } from "../../domain/contracts";
import { visualModeAvailable, visualModeMeasurementEligible, visualModeReason } from "./visualModels";

const manifest = {
  visual_models: {
    evidence_cloud: { available: true, measurement_eligible: true },
    dense_cloud: { available: true, url: "/dense/fused.ply", measurement_eligible: false },
    textured_mesh: {
      available: true,
      url: "/dense/textured/model.ply",
      measurement_eligible: false,
    },
    gaussian_splat: {
      available: true,
      url: "/dense/model.splat",
      measurement_eligible: true,
    },
    ai_completed_mesh: {
      available: true,
      url: "/completion/completed_mesh.ply",
      measurement_eligible: false,
      statement: "Visual hypothesis only",
    },
  },
} as ViewerManifest;

describe("visual reconstruction protections", () => {
  it("keeps measurement bound to evidence geometry", () => {
    expect(visualModeMeasurementEligible("EVIDENCE", manifest)).toBe(true);
    expect(
      visualModeMeasurementEligible("EVIDENCE", {
        visual_models: {
          evidence_cloud: { available: true, measurement_eligible: false },
        },
      } as ViewerManifest),
    ).toBe(false);
    expect(visualModeMeasurementEligible("TEXTURED", manifest)).toBe(false);
    expect(visualModeMeasurementEligible("PREDICTED", manifest)).toBe(false);
    expect(visualModeMeasurementEligible("PHOTOREAL", manifest)).toBe(false);
  });

  it("uses declared visual artifacts and falls back to evidence", () => {
    expect(visualModeAvailable("EVIDENCE", {} as ViewerManifest)).toBe(true);
    expect(visualModeAvailable("TEXTURED", {} as ViewerManifest)).toBe(false);
    expect(visualModeAvailable("TEXTURED", manifest)).toBe(true);
    expect(visualModeAvailable("PREDICTED", manifest)).toBe(true);
    expect(visualModeAvailable("PHOTOREAL", manifest)).toBe(false);
    expect(visualModeReason("PHOTOREAL", manifest)).toMatch(/Photoreal View unavailable/);
    expect(visualModeReason("TEXTURED", {} as ViewerManifest)).toMatch(/not declared/);
  });
});
