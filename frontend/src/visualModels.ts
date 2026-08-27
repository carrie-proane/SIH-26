import type { ViewerManifest, VisualMode } from "./types";

export function visualModeAvailable(mode: VisualMode, manifest: ViewerManifest): boolean {
  if (mode === "EVIDENCE") return true;
  if (mode === "TEXTURED") {
    return Boolean(
      manifest.visual_models?.textured_mesh.available &&
        manifest.visual_models.textured_mesh.url,
    );
  }
  // Phase 2 declares the Gaussian Splat contract only. Keep the control disabled
  // until a provenance-preserving browser loader is deliberately implemented.
  return false;
}

export function visualModeMeasurementEligible(
  mode: VisualMode,
  manifest: ViewerManifest,
): boolean {
  return (
    mode === "EVIDENCE" &&
    manifest.visual_models?.evidence_cloud.measurement_eligible !== false
  );
}
