import type { ViewerManifest, VisualModel } from "../../domain/contracts";
import { resolveAssetUrl } from "../../services/api";

/**
 * Only URLs published by the viewer manifest may be requested by the WebGL
 * surface. This is deliberately stricter than resolveAssetUrl(), which is
 * also used by the source-frame inspector.
 */
export function declaredVisualArtifactUrls(manifest: ViewerManifest): Set<string> {
  const urls = new Set<string>();
  const add = (model: VisualModel | undefined) => {
    if (model?.available && model.url) urls.add(model.url);
  };

  if (manifest.cloud.url) urls.add(manifest.cloud.url);
  add(manifest.visual_models?.evidence_cloud);
  add(manifest.visual_models?.dense_cloud);
  add(manifest.visual_models?.textured_mesh);
  add(manifest.visual_models?.ai_completed_mesh);
  add(manifest.visual_models?.gaussian_splat);
  for (const texture of manifest.visual_models?.textured_mesh?.texture_urls ?? []) {
    if (texture) urls.add(texture);
  }
  return urls;
}

export function isDeclaredVisualArtifact(
  manifest: ViewerManifest,
  url: string | null | undefined,
): url is string {
  return Boolean(url && declaredVisualArtifactUrls(manifest).has(url));
}

export function modelFormat(model: VisualModel | undefined, url: string): "PLY" | "GLB" | "SPLAT" {
  if (model?.format === "GLB" || model?.format === "SPLAT" || model?.format === "PLY") {
    return model.format;
  }
  const extension = url.split("?", 1)[0].toLowerCase().split(".").pop();
  if (extension === "glb" || extension === "gltf") return "GLB";
  if (extension === "splat" || extension === "ksplat") return "SPLAT";
  return "PLY";
}

export function formatByteSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1_000) return `${Math.round(bytes)} B`;
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}

export function visualModelLabel(
  mode: "EVIDENCE" | "TEXTURED" | "PREDICTED" | "PHOTOREAL",
): string {
  if (mode === "TEXTURED") return "Textured Model";
  if (mode === "PREDICTED") return "AI Predicted Surface";
  if (mode === "PHOTOREAL") return "Photoreal View";
  return "Evidence Cloud";
}

export async function fetchDeclaredVisualArtifact(
  manifest: ViewerManifest,
  url: string,
  onProgress?: (loadedBytes: number, totalBytes: number | null) => void,
  signal?: AbortSignal,
): Promise<{ buffer: ArrayBuffer; totalBytes: number | null }> {
  if (!isDeclaredVisualArtifact(manifest, url)) {
    throw new Error("Refusing to load an undeclared visual artifact.");
  }
  const response = await fetch(resolveAssetUrl(url), { signal });
  if (!response.ok) {
    throw new Error(`Visual artifact request failed (${response.status} ${response.statusText}).`);
  }
  const headerBytes = Number(response.headers.get("content-length"));
  const totalBytes = Number.isFinite(headerBytes) && headerBytes >= 0 ? headerBytes : null;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, totalBytes ?? buffer.byteLength);
    return { buffer, totalBytes: totalBytes ?? buffer.byteLength };
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress?.(loadedBytes, totalBytes);
  }
  const buffer = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress?.(loadedBytes, totalBytes ?? loadedBytes);
  return { buffer: buffer.buffer, totalBytes: totalBytes ?? loadedBytes };
}
