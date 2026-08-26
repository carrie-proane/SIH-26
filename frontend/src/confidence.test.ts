import { describe, expect, it } from "vitest";

import { parsePointConfidence } from "./confidence";
import viewerSource from "./components/PointCloudViewer.tsx?raw";

describe("explicit point-confidence contract", () => {
  it("accepts only records with every scientific support field", () => {
    const valid = parsePointConfidence({
      schema_version: "1.0",
      point_order: "PLY_VERTEX_ORDER",
      points: [
        {
          point_id: 0,
          supporting_views: 5,
          track_length: 5,
          reprojection_error: 0.42,
          triangulation_angle: 8.5,
          confidence_class: "OBSERVED_HIGH",
        },
      ],
    });
    expect(valid?.points[0].confidence_class).toBe("OBSERVED_HIGH");
    expect(
      parsePointConfidence({
        schema_version: "1.0",
        point_order: "PLY_VERTEX_ORDER",
        points: [{ point_id: 0, red: 32, green: 191, blue: 107 }],
      }),
    ).toBeNull();
  });

  it("never maps photographic PLY RGB values to confidence classes", () => {
    expect(viewerSource).not.toContain("nearestConfidence");
    expect(viewerSource).not.toContain("hexToRgb");
    expect(viewerSource).toContain("photographic-rgb-point-cloud");
    expect(viewerSource).toContain("Visual estimate - verification confidence unavailable");
  });
});
