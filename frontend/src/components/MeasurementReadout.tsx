import type { MeasurementResult } from "../types";

export function MeasurementReadout({ measurement, referenceNote = "" }: {
  measurement: MeasurementResult; referenceNote?: string;
}) {
  return <div className="measurement-readout" data-status={measurement.status}>
    <span>{measurement.status === "IDLE" ? "DISTANCE TOOL" : measurement.status}</span>
    <strong>{measurement.distanceM === null ? "—" : `${measurement.distanceM.toFixed(3)} m`}</strong>
    <small>{measurement.message}{referenceNote}</small>
  </div>;
}
