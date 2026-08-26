import * as THREE from "three";

const LOWER_QUANTILE = 0.01;
const UPPER_QUANTILE = 0.99;

function quantileIndex(length: number, quantile: number): number {
  return Math.max(0, Math.min(length - 1, Math.round((length - 1) * quantile)));
}

/**
 * Return display bounds that are not dominated by a small number of poorly
 * triangulated sparse points. The geometry itself is left untouched.
 */
export function robustSceneBounds(
  position: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
): THREE.Box3 {
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  // Keep camera setup bounded on very large clouds without changing rendered data.
  const stride = Math.max(1, Math.floor(position.count / 100_000));
  for (let index = 0; index < position.count; index += stride) {
    xs.push(position.getX(index));
    ys.push(position.getZ(index));
    zs.push(-position.getY(index));
  }
  if (!xs.length) return new THREE.Box3();
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const low = quantileIndex(xs.length, LOWER_QUANTILE);
  const high = quantileIndex(xs.length, UPPER_QUANTILE);
  return new THREE.Box3(
    new THREE.Vector3(xs[low], ys[low], zs[low]),
    new THREE.Vector3(xs[high], ys[high], zs[high]),
  );
}

export function pointSizeForRadius(radius: number): number {
  return Math.max(0.13, Math.min(radius * 0.008, 2.5));
}
