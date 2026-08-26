import * as THREE from "three";
import { describe, expect, it } from "vitest";

import viewerSource from "./components/PointCloudViewer.tsx?raw";
import {
  cameraDistanceForSphere,
  pointSizeForRadius,
  robustSceneBounds,
} from "./viewerBounds";

describe("robust point-cloud framing", () => {
  it("does not let sparse reconstruction outliers hide the observed cloud", () => {
    const values: number[] = [];
    for (let index = 0; index < 1_000; index += 1) {
      values.push(index / 10, index / 20, 10 + (index % 10));
    }
    values.push(-2_000, -1_000, -500, 2_000, 1_000, 500);
    const position = new THREE.Float32BufferAttribute(values, 3);

    const bounds = robustSceneBounds(position);
    const size = bounds.getSize(new THREE.Vector3());

    expect(size.x).toBeLessThan(150);
    expect(size.y).toBeLessThan(20);
    expect(size.z).toBeLessThan(100);
  });

  it("increases rendered point size for a large metric scene", () => {
    expect(pointSizeForRadius(2)).toBe(0.13);
    expect(pointSizeForRadius(150)).toBeCloseTo(1.2);
  });

  it("does not hide metric evidence behind a fixed-distance fog effect", () => {
    expect(viewerSource).not.toContain("FogExp2");
  });

  it("fits against the narrower field of view after responsive resizing", () => {
    const landscape = cameraDistanceForSphere(100, 45, 16 / 9);
    const portrait = cameraDistanceForSphere(100, 45, 9 / 16);
    expect(portrait).toBeGreaterThan(landscape);
  });
});
