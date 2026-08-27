import type { ViewerManifest, VisualMode } from "./types";
import { modelFormat } from "./modelLoading";

function modelForMode(mode: VisualMode, manifest: ViewerManifest) {
  if (mode === "TEXTURED") return manifest.visual_models?.textured_mesh;
  if (mode === "PHOTOREAL") return manifest.visual_models?.gaussian_splat;
  return manifest.visual_models?.evidence_cloud;
}

export function visualModeAvailable(mode: VisualMode, manifest: ViewerManifest): boolean {
  if (mode === "EVIDENCE") return true;
  const model = modelForMode(mode, manifest);
  if (!model?.available || !model.url) return false;
  // Gaussian Splat rendering is intentionally still gated: a declared splat
  // must not be presented as loadable until a provenance-preserving loader is
  // shipped. PLY and self-contained GLB are supported by PointCloudViewer.
  if (mode === "PHOTOREAL") return false;
  return modelFormat(model, model.url) === "PLY" || modelFormat(model, model.url) === "GLB";
}

export function visualModeReason(mode: VisualMode, manifest: ViewerManifest): string {
  if (mode === "EVIDENCE") return "Default verified evidence geometry";
  const model = modelForMode(mode, manifest);
  if (!model?.available || !model.url) {
    return model?.statement ?? `${mode === "TEXTURED" ? "Textured model" : "Photoreal view"} was not declared by this run.`;
  }
  if (mode === "PHOTOREAL") {
    return model.statement
      ? `Photoreal View unavailable: ${model.statement}`
      : "Photoreal View unavailable: no approved browser loader is shipped in this build.";
  }
  if (modelFormat(model, model.url) === "SPLAT") {
    return "Declared Gaussian Splat format is not supported by this browser build.";
  }
  return model.statement ?? "Visual only — not used for verified measurement";
}

export function visualModeMeasurementEligible(
  mode: VisualMode,
  manifest: ViewerManifest,
): boolean {
  return (
    mode === "EVIDENCE" &&
    manifest.visual_models?.evidence_cloud?.measurement_eligible !== false
  );
}
