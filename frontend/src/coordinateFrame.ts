export interface CoordinateFramePresentation {
  axes: [string, string, string];
  metric: boolean;
  orientationVerified: boolean;
  gridNotice: string | null;
  description: string;
}

export function coordinateFramePresentation(
  coordinateFrame: string | null | undefined,
): CoordinateFramePresentation {
  if (coordinateFrame === "LOCAL_ENU_METRES") {
    return {
      axes: ["E", "U", "N"],
      metric: true,
      orientationVerified: true,
      gridNotice: null,
      description: "Local ENU metric frame",
    };
  }

  if (coordinateFrame === "COLMAP_SFM") {
    return {
      // The renderer maps source (x, y, z) to display (x, z, -y).
      axes: ["X", "Z", "−Y"],
      metric: false,
      orientationVerified: false,
      gridNotice: "DISPLAY GRID · NOT VERIFIED GROUND",
      description: "Unoriented COLMAP SfM frame · scale unavailable",
    };
  }

  return {
    axes: ["X", "Y", "Z"],
    metric: false,
    orientationVerified: false,
    gridNotice: "DISPLAY GRID · NOT VERIFIED GROUND",
    description: "Unknown coordinate frame · orientation and scale unavailable",
  };
}
