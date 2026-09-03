import { describe, expect, it } from "vitest";
import * as THREE from "three";

import {
  adjustOpenMVSAtlasPixels,
  filterTriangleIndicesByAtlasAlpha,
  makeTexturedVisualMaterial,
} from "./texturedMaterial";

describe("photographic textured-model material", () => {
  it("uses an unlit material and does not multiply the atlas by vertex colours", () => {
    const texture = new THREE.Texture();

    const material = makeTexturedVisualMaterial(texture, true);

    expect(material).toBeInstanceOf(THREE.MeshBasicMaterial);
    expect(material.map).toBe(texture);
    expect(material.vertexColors).toBe(false);
    expect(material.alphaTest).toBe(0.5);
    expect(material.name).toBe("photographic-atlas-unlit-empty-texels-hidden");
  });

  it("hides OpenMVS empty pixels and lifts only photographic pixels", () => {
    const pixels = new Uint8ClampedArray([
      255, 127, 39, 255,
      10, 20, 30, 255,
    ]);

    const stats = adjustOpenMVSAtlasPixels(pixels);

    expect(Array.from(pixels.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(pixels[4]).toBeGreaterThan(10);
    expect(pixels[5]).toBeGreaterThan(20);
    expect(pixels[6]).toBeGreaterThan(30);
    expect(pixels[7]).toBe(255);
    expect(stats.hiddenEmptyPixels).toBe(1);
    expect(stats.adjustedPhotographicPixels).toBe(1);
    expect(stats.displayGamma).toBeGreaterThanOrEqual(0.55);
    expect(stats.displayGamma).toBeLessThanOrEqual(1);
  });

  it("retains vertex colours only when no photographic atlas is available", () => {
    const material = makeTexturedVisualMaterial(null, true);

    expect(material).toBeInstanceOf(THREE.MeshBasicMaterial);
    expect(material.map).toBeNull();
    expect(material.vertexColors).toBe(true);
    expect(material.alphaTest).toBe(0);
    expect(material.name).toBe("textured-model-fallback-material");
  });

  it("filters only faces whose declared texture samples are unsupported", () => {
    const result = filterTriangleIndicesByAtlasAlpha(
      [0, 1, 2, 3, 4, 5],
      [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
      {
        width: 2,
        height: 2,
        alpha: new Uint8Array([0, 0, 255, 255]),
      },
      1,
    );

    expect(result.indices).toEqual([3, 4, 5]);
    expect(result.removedFaces).toBe(1);
  });

  it("recognizes the declared magenta empty colour for future runs", () => {
    const pixels = new Uint8ClampedArray([255, 0, 255, 255]);

    adjustOpenMVSAtlasPixels(pixels, [[255, 0, 255]]);

    expect(Array.from(pixels)).toEqual([0, 0, 0, 0]);
  });
});
