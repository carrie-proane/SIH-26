import { expect, test } from "@playwright/test";

test("offline fixture opens the WebGL operator workspace", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /See what the camera proved/i })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("operator-setup.png"), fullPage: true });
  await page.goto("/?fixture=1");
  await expect(page.getByText("UI / orchestration fixture")).toBeVisible();
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /AI depth/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Photographic RGB/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Evidence Cloud/i })).toHaveClass(/is-active/);
  await expect(page.getByRole("button", { name: /Textured Model/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Photoreal View/i })).toBeDisabled();
  await expect(page.getByText("Confidence unavailable for this run")).toBeVisible();
  await expect(page.getByText("Reconstruction policy")).toBeVisible();
  await expect(page.getByText("UNMASKED_FALLBACK")).toBeVisible();
  await page.setViewportSize({ width: 1600, height: 900 });
  const viewport = await page.getByLabel("Interactive reconstruction viewport").boundingBox();
  expect(viewport?.height).toBeLessThan(900);
  await page.screenshot({ path: testInfo.outputPath("operator-ui-fixture.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();

  await expect(page.getByRole("button", { name: /Measure/i })).toBeDisabled();
  await expect(page.getByText(/verified measurement unavailable/i).first()).toBeVisible();

  await page.getByRole("button", { name: /Source frame/i }).click();
  await expect(page.getByText("SOURCE PREVIEW NOT DECLARED")).toBeVisible();
});

test("processing screen reports server state without an invented ETA", async ({ page }, testInfo) => {
  const current = Date.now();
  const run = {
    project_id: "prj_processing", run_id: "run_processing", stage: "RECONSTRUCTING",
    status: "RECONSTRUCTING", progress: 55, created_at: new Date(current - 120_000).toISOString(),
    updated_at: new Date(current).toISOString(), processing_started_at: new Date(current - 110_000).toISOString(),
    processing_completed_at: null, config_version: "1.0", config: { execution_mode: "COLMAP", matcher: "SIFT" },
    environment: {}, failure_reason: null, events: [{ stage: "RECONSTRUCTING", status: "STARTED", timestamp: "2026-09-22T00:02:00Z", progress: 55, message: "Sparse mapper running on the API host" }],
    artifacts: [], synthetic_fixture: false, source_provenance: "REAL", video_origin: "REAL",
    telemetry_origin: "REAL", telemetry_offset_s: 0, offset_source: "automatic", rmse_before_m: null,
    rmse_after_m: null, stage_timings_s: { INGEST: 1.2, PREPROCESS: 34.5 }, requested_matcher: "SIFT",
    executed_matcher: "SIFT", effective_sparse_gpu: false,
  };
  const project = { project_id: "prj_processing", name: "Processing capture", description: "", data_classification: "INTERNAL", created_at: "2026-09-22T00:00:00Z", immutable: true, assets: [], warnings: [], source_provenance: "REAL", video_origin: "REAL", telemetry_origin: "REAL" };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/runs/run_processing") return route.fulfill({ json: run });
    if (path === "/api/runs/run_processing/readiness") return route.fulfill({ json: { schema_version: "1.0", run_id: "run_processing", stage: "RECONSTRUCTING", status: "RECONSTRUCTING", sparse_preview_ready: false, sparse_preview_missing: ["sparse/sparse_local.ply", "camera_poses.csv", "keyframes.json"], quality_report_ready: false, dense_status: "NOT_READY_OR_NOT_REQUESTED", dense_visual_artifacts: [], measurement_geometry: null, inferred_geometry_measurement_eligible: false, export_report_ready: false, export_report_url: null, export_status: {} } });
    if (path === "/api/projects/prj_processing") return route.fulfill({ json: project });
    if (path === "/api/projects") return route.fulfill({ json: { schema_version: "1.0", projects: [project] } });
    if (path === "/api/projects/prj_processing/runs") return route.fulfill({ json: { schema_version: "1.0", project_id: "prj_processing", runs: [run] } });
    return route.abort();
  });
  await page.goto("/?project=prj_processing&run=run_processing");
  await expect(page.getByRole("heading", { name: /Building reconstruction evidence/i })).toBeVisible();
  await expect(page.getByText("No browser ETA is inferred.")).toBeVisible();
  await expect(page.getByText("NOT READY", { exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("operator-processing.png"), fullPage: true });
});

test("live local API smoke run reaches the viewer manifest", async ({ page }, testInfo) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Run API smoke fixture/i }).click();

  await expect(page.getByText("UI / orchestration fixture")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%", { exact: true })).toBeVisible();
  await expect(page.getByText(/not empirical reconstruction results/i)).toBeVisible();
  await expect(page.getByText("Confidence unavailable for this run")).toBeVisible();

  await page.getByText("Persisted measurements").scrollIntoViewIfNeeded();
  await page.getByText("Geometry exports").scrollIntoViewIfNeeded();
  await page.locator(".inspector").screenshot({ path: testInfo.outputPath("operator-measurements-exports.png") });

  await expect(page).toHaveURL(/\?project=prj_.*&run=run_/);
  await page.reload();
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%", { exact: true })).toBeVisible();
});
