import * as THREE from "three";

const OPENMVS_EMPTY_RGB = [255, 127, 39] as const;
const EMPTY_TOLERANCE = 2;
const TARGET_BRIGHT_PERCENTILE = 170;
const MIN_DISPLAY_GAMMA = 0.55;

export interface AtlasDisplayStats {
  hiddenEmptyPixels: number;
  adjustedPhotographicPixels: number;
  displayGamma: number;
}

export interface AtlasAlphaMask {
  width: number;
  height: number;
  alpha: Uint8Array;
}

/** Apply a display-only shadow lift and make OpenMVS empty texels transparent. */
export function adjustOpenMVSAtlasPixels(
  pixels: Uint8ClampedArray,
  additionalEmptyColors: ReadonlyArray<readonly [number, number, number]> = [],
): AtlasDisplayStats {
  const emptyColors = [OPENMVS_EMPTY_RGB, ...additionalEmptyColors];
  const brightnessHistogram = new Uint32Array(256);
  let photographicPixels = 0;
  for (let offset = 0; offset + 3 < pixels.length; offset += 4) {
    const isEmpty = emptyColors.some((color) =>
      color.every(
        (expected, channel) => Math.abs(pixels[offset + channel] - expected) <= EMPTY_TOLERANCE,
      ),
    );
    if (!isEmpty && pixels[offset + 3] > 0) {
      brightnessHistogram[Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2])] += 1;
      photographicPixels += 1;
    }
  }
  const percentileTarget = photographicPixels * 0.9;
  let cumulative = 0;
  let brightPercentile = 255;
  for (let value = 0; value < brightnessHistogram.length; value += 1) {
    cumulative += brightnessHistogram[value];
    if (cumulative >= percentileTarget) {
      brightPercentile = value;
      break;
    }
  }
  const calculatedGamma =
    brightPercentile > 0 && brightPercentile < TARGET_BRIGHT_PERCENTILE
      ? Math.log(TARGET_BRIGHT_PERCENTILE / 255) / Math.log(brightPercentile / 255)
      : 1;
  const displayGamma = Math.max(MIN_DISPLAY_GAMMA, Math.min(1, calculatedGamma));
  const gamma = new Uint8ClampedArray(256);
  for (let value = 0; value < gamma.length; value += 1) {
    gamma[value] = Math.round(255 * Math.pow(value / 255, displayGamma));
  }
  let hiddenEmptyPixels = 0;
  let adjustedPhotographicPixels = 0;
  for (let offset = 0; offset + 3 < pixels.length; offset += 4) {
    const isEmpty = emptyColors.some((color) =>
      color.every(
        (expected, channel) => Math.abs(pixels[offset + channel] - expected) <= EMPTY_TOLERANCE,
      ),
    );
    if (isEmpty) {
      pixels[offset] = 0;
      pixels[offset + 1] = 0;
      pixels[offset + 2] = 0;
      pixels[offset + 3] = 0;
      hiddenEmptyPixels += 1;
      continue;
    }
    pixels[offset] = gamma[pixels[offset]];
    pixels[offset + 1] = gamma[pixels[offset + 1]];
    pixels[offset + 2] = gamma[pixels[offset + 2]];
    adjustedPhotographicPixels += 1;
  }
  return { hiddenEmptyPixels, adjustedPhotographicPixels, displayGamma };
}

/**
 * Derive a browser-only texture while keeping the declared source atlas
 * unchanged as a provenance-bearing artifact.
 */
export function prepareOpenMVSAtlasTexture(
  texture: THREE.Texture,
  emptyColors?: ReadonlyArray<readonly [number, number, number]>,
): {
  texture: THREE.CanvasTexture;
  stats: AtlasDisplayStats;
  alphaMask: AtlasAlphaMask;
} {
  const source = texture.image as CanvasImageSource & {
    width?: number;
    height?: number;
    naturalWidth?: number;
    naturalHeight?: number;
  };
  const width = source.naturalWidth ?? source.width ?? 0;
  const height = source.naturalHeight ?? source.height ?? 0;
  if (!width || !height) throw new Error("Texture atlas decoded without dimensions.");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Browser cannot prepare the photographic texture atlas.");
  context.drawImage(source, 0, 0, width, height);
  const imageData = context.getImageData(0, 0, width, height);
  const stats = adjustOpenMVSAtlasPixels(imageData.data, emptyColors);
  const alpha = new Uint8Array(width * height);
  for (let sourceOffset = 3, targetOffset = 0; sourceOffset < imageData.data.length; sourceOffset += 4) {
    alpha[targetOffset] = imageData.data[sourceOffset];
    targetOffset += 1;
  }
  context.putImageData(imageData, 0, 0);
  texture.dispose();
  const derived = new THREE.CanvasTexture(canvas);
  derived.colorSpace = THREE.SRGBColorSpace;
  derived.name = "display-derived-openmvs-atlas-source-preserved";
  derived.needsUpdate = true;
  return { texture: derived, stats, alphaMask: { width, height, alpha } };
}

export function filterTriangleIndicesByAtlasAlpha(
  indices: ArrayLike<number>,
  uvs: ArrayLike<number>,
  atlas: AtlasAlphaMask,
  minimumSupportedSamples = 1,
): { indices: number[]; removedFaces: number } {
  const filtered: number[] = [];
  let removedFaces = 0;
  const supported = (u: number, v: number) => {
    const x = Math.min(atlas.width - 1, Math.max(0, Math.round(u * (atlas.width - 1))));
    const y = Math.min(
      atlas.height - 1,
      Math.max(0, Math.round((1 - v) * (atlas.height - 1))),
    );
    return atlas.alpha[y * atlas.width + x] >= 128;
  };
  for (let offset = 0; offset + 2 < indices.length; offset += 3) {
    const triangle = [Number(indices[offset]), Number(indices[offset + 1]), Number(indices[offset + 2])];
    const samples = triangle.map((index) => [Number(uvs[index * 2]), Number(uvs[index * 2 + 1])]);
    samples.push([
      (samples[0][0] + samples[1][0] + samples[2][0]) / 3,
      (samples[0][1] + samples[1][1] + samples[2][1]) / 3,
    ]);
    const supportedSamples = samples.filter(([u, v]) => supported(u, v)).length;
    if (supportedSamples >= minimumSupportedSamples) filtered.push(...triangle);
    else removedFaces += 1;
  }
  return { indices: filtered, removedFaces };
}

/** Display the photographic atlas without multiplying it by artificial lights. */
export function makeTexturedVisualMaterial(
  texture: THREE.Texture | null,
  hasVertexColors: boolean,
): THREE.MeshBasicMaterial {
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    color: texture || hasVertexColors ? "#ffffff" : "#aabbb5",
    vertexColors: hasVertexColors && texture === null,
    side: THREE.DoubleSide,
    alphaTest: texture ? 0.5 : 0,
  });
  material.name = texture
    ? "photographic-atlas-unlit-empty-texels-hidden"
    : "textured-model-fallback-material";
  return material;
}
