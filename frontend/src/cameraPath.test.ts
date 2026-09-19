import { expect, it } from "vitest";
import { cameraPathPositions } from "./components/PointCloudViewer";

it("draws timestamp-ordered CSV poses without sorting by frame ID", () => {
  const poses = [90, 1, 3].map((frameIndex, index) => ({
    frameIndex, timestampS: index, x: index, y: index + 2, z: index + 4,
  }));
  expect(cameraPathPositions(poses).map(point => point.toArray())).toEqual([
    [0, 4, -2], [1, 5, -3], [2, 6, -4],
  ]);
  // Even inconsistent input stays in source order: ordering has one owner.
  expect(cameraPathPositions([...poses].reverse()).map(point => point.x)).toEqual([2, 1, 0]);
});
