import { describe, expect, it } from "vitest";

import { coordinateFramePresentation } from "./coordinateFrame";

describe("coordinate-frame presentation", () => {
  it("uses geographic axes only for a declared local ENU frame", () => {
    expect(coordinateFramePresentation("LOCAL_ENU_METRES")).toEqual({
      axes: ["E", "U", "N"],
      metric: true,
      orientationVerified: true,
      gridNotice: null,
      description: "Local ENU metric frame",
    });
  });

  it("does not present raw COLMAP coordinates as verified ground or ENU", () => {
    const presentation = coordinateFramePresentation("COLMAP_SFM");
    expect(presentation.axes).toEqual(["X", "Z", "−Y"]);
    expect(presentation.metric).toBe(false);
    expect(presentation.orientationVerified).toBe(false);
    expect(presentation.gridNotice).toMatch(/not verified ground/i);
    expect(presentation.description).toMatch(/unoriented colmap/i);
  });

  it("fails closed for an unknown coordinate frame", () => {
    const presentation = coordinateFramePresentation(undefined);
    expect(presentation.orientationVerified).toBe(false);
    expect(presentation.gridNotice).toMatch(/not verified ground/i);
  });
});
