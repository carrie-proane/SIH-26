import type {
  ConfidenceLabel,
  PointConfidenceArtifact,
  PointConfidenceRecord,
} from "./types";

const VALID_CLASSES = new Set<ConfidenceLabel>([
  "OBSERVED_HIGH",
  "OBSERVED_MEDIUM",
  "OBSERVED_LOW",
  "AI_ASSISTED_NOT_MEASURABLE",
  "UNSEEN",
]);

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function parsePoint(value: unknown): PointConfidenceRecord | null {
  if (!isObject(value)) return null;
  const confidenceClass = value.confidence_class;
  if (
    !Number.isInteger(value.point_id) ||
    !Number.isInteger(value.supporting_views) ||
    !Number.isInteger(value.track_length) ||
    !nonNegativeNumber(value.point_id) ||
    !nonNegativeNumber(value.supporting_views) ||
    !nonNegativeNumber(value.track_length) ||
    !nonNegativeNumber(value.reprojection_error) ||
    !nonNegativeNumber(value.triangulation_angle) ||
    value.triangulation_angle > 180 ||
    typeof confidenceClass !== "string" ||
    !VALID_CLASSES.has(confidenceClass as ConfidenceLabel)
  ) {
    return null;
  }
  return {
    point_id: value.point_id,
    supporting_views: value.supporting_views,
    track_length: value.track_length,
    reprojection_error: value.reprojection_error,
    triangulation_angle: value.triangulation_angle,
    confidence_class: confidenceClass as ConfidenceLabel,
  };
}

export function parsePointConfidence(payload: unknown): PointConfidenceArtifact | null {
  if (
    !isObject(payload) ||
    payload.schema_version !== "1.0" ||
    payload.point_order !== "PLY_VERTEX_ORDER" ||
    !Array.isArray(payload.points)
  ) {
    return null;
  }
  const points = payload.points.map(parsePoint);
  if (points.some((point) => point === null)) return null;
  const records = points as PointConfidenceRecord[];
  const ids = records.map((point) => point.point_id).sort((left, right) => left - right);
  if (ids.some((pointId, index) => pointId !== index)) return null;
  return { schema_version: "1.0", point_order: "PLY_VERTEX_ORDER", points: records };
}
