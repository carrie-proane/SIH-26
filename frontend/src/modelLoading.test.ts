import { describe, expect, it } from "vitest";

import type { ViewerManifest, VisualModel } from "./types";
import {
  declaredVisualArtifactUrls,
  formatByteSize,
  isDeclaredVisualArtifact,
  modelFormat,
} from "./modelLoading";

const manifest = {
  cloud: { url: "/api/evidence.ply" },
  visual_models: {
    evidence_cloud: { available: true, url: "/api/evidence.ply", format: "PLY" },
    dense_cloud: { available: true, url: "/api/fused.ply", format: "PLY" },
    textured_mesh: {
      available: true,
      url: "/api/model.glb",
      format: "GLB",
      texture_urls: ["/api/atlas.png"],
    },
    gaussian_splat: { available: false, url: null },
  },
} as ViewerManifest;

describe("declared visual artifact loading", () => {
  it("allows only URLs published by the manifest", () => {
    const urls = declaredVisualArtifactUrls(manifest);
    expect(urls).toEqual(new Set(["/api/evidence.ply", "/api/fused.ply", "/api/model.glb", "/api/atlas.png"]));
    expect(isDeclaredVisualArtifact(manifest, "/api/model.glb")).toBe(true);
    expect(isDeclaredVisualArtifact(manifest, "/api/secret.ply")).toBe(false);
  });

  it("uses explicit formats and safe extension fallback", () => {
    expect(modelFormat({ format: "GLB" } as VisualModel, "/model.ply")).toBe("GLB");
    expect(modelFormat(undefined, "/model.glb")).toBe("GLB");
    expect(modelFormat(undefined, "/model.splat")).toBe("SPLAT");
    expect(modelFormat(undefined, "/model.ply")).toBe("PLY");
  });

  it("formats artifact sizes for the browser status chrome", () => {
    expect(formatByteSize(42)).toBe("42 B");
    expect(formatByteSize(12_345)).toBe("12.3 KB");
    expect(formatByteSize(2_500_000)).toBe("2.5 MB");
    expect(formatByteSize(null)).toBe("—");
  });
});
