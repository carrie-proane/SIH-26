import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { assessMeasurement } from "./confidence";
import { MeasurementReadout } from "./components/MeasurementReadout";

it("shows the numeric distance as a caution for high-support points on a degenerate run", () => {
  const measurement = assessMeasurement(2.87, ["OBSERVED_HIGH", "OBSERVED_HIGH"], {
    evidence_verdict: "NOT_VALIDATED", alignment_identifiability: "degenerate",
  });
  const { container } = render(<MeasurementReadout measurement={measurement} />);
  expect(screen.getByText("2.870 m")).toBeVisible();
  expect(screen.getByText("CAUTION")).toBeVisible();
  expect(screen.getByText(/scale not independently verified/)).toBeVisible();
  expect(container.firstChild).toHaveAttribute("data-status", "CAUTION");
});

it("allows validated high support, and never upgrades weak or disabled geometry", () => {
  const evidence = { evidence_verdict: "PASSED", alignment_identifiability: "well_conditioned" } as const;
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "OBSERVED_HIGH"], evidence).status).toBe("ALLOWED");
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "OBSERVED_HIGH"], {}).status).toBe("CAUTION");
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "OBSERVED_HIGH"], {...evidence, evidence_verdict: "FAILED"}).status).toBe("CAUTION");
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "OBSERVED_MEDIUM"], evidence).status).toBe("CAUTION");
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "OBSERVED_LOW"], evidence).status).toBe("CONFIRM");
  expect(assessMeasurement(2, ["OBSERVED_HIGH", "UNSEEN"], evidence).distanceM).toBeNull();
});
