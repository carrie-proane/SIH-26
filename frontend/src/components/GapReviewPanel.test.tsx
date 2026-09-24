import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GapReviewPanel } from "./GapReviewPanel";

const sourceHash = "a".repeat(64);
const frameHash = "b".repeat(64);
const coverage = {
  status: "COMPLETED",
  source_geometry_sha256: sourceHash,
  metric_depth_status: "NOT_DECLARED",
  candidate_missing_region_statistics: {
    regions: [{
      boundary_id: "boundary-fixture",
      boundary_coordinates_enu_m: [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
      eligible_for_operator_review: true,
      reasons: [],
    }],
  },
  review_evidence_options: [{
    image_name: "frame.png",
    artifact_path: "frames/frame.png",
    artifact_url: "/api/runs/run_fixture/artifacts/frames/frame.png",
    sha256: frameHash,
  }],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("GapReviewPanel", () => {
  it("submits an accepted gap with only the declared path and hash", async () => {
    const changed = vi.fn();
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/coverage")) return Promise.resolve(json(coverage));
      if (url.endsWith("/completion") && init?.method === "POST") {
        const request = JSON.parse(String(init.body));
        expect(request.source_geometry_sha256).toBe(sourceHash);
        expect(request.reviews[0].source_images).toEqual([
          { path: "frames/frame.png", sha256: frameHash },
        ]);
        return Promise.resolve(json({ status: "completed", inferred_regions_present: true }, 201));
      }
      if (url.endsWith("/completion")) return Promise.resolve(json({ status: "not_run" }));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    }));
    render(<GapReviewPanel runId="run_fixture" onCompleted={changed} />);
    await screen.findByText(/boundary-fixt/i);
    await userEvent.selectOptions(screen.getByLabelText("Decision"), "CONFIRMED_SMALL_GAP");
    await userEvent.type(screen.getByLabelText("Reviewer"), "Yosha");
    await userEvent.type(screen.getByLabelText("Explanation"), "Occluded bounded patch");
    await userEvent.selectOptions(screen.getByLabelText("Declared image evidence"), "frames/frame.png");
    await userEvent.click(screen.getByRole("button", { name: /submit reviewed gaps/i }));
    await waitFor(() => expect(changed).toHaveBeenCalledOnce());
    expect(screen.getByText(/completion: completed/i)).toBeVisible();
  });

  it("submits a structural opening without fabricated image evidence and shows refusal", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/coverage")) return Promise.resolve(json(coverage));
      if (url.endsWith("/completion") && init?.method === "POST") {
        const request = JSON.parse(String(init.body));
        expect(request.reviews[0].decision).toBe("STRUCTURAL_OPENING");
        expect(request.reviews[0].source_images).toEqual([]);
        return Promise.resolve(json({ status: "refused", failure_reason: "No reviewed gap passed" }, 201));
      }
      if (url.endsWith("/completion")) return Promise.resolve(json({ status: "not_run" }));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    }));
    render(<GapReviewPanel runId="run_fixture" onCompleted={() => undefined} />);
    await screen.findByText(/boundary-fixt/i);
    await userEvent.selectOptions(screen.getByLabelText("Decision"), "STRUCTURAL_OPENING");
    await userEvent.type(screen.getByLabelText("Reviewer"), "Operator");
    await userEvent.type(screen.getByLabelText("Explanation"), "This is a doorway");
    await userEvent.click(screen.getByRole("button", { name: /submit reviewed gaps/i }));
    expect(await screen.findByText(/no reviewed gap passed/i)).toBeVisible();
  });
});
