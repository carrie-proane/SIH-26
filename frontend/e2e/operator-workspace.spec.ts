import { expect, test } from "@playwright/test";

test("offline fixture opens the WebGL operator workspace", async ({ page }) => {
  await page.goto("/?fixture=1");
  await expect(page.getByText("UI / orchestration fixture")).toBeVisible();
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%")).toBeVisible();
  await expect(page.getByRole("button", { name: /AI depth/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Photographic RGB/i })).toBeVisible();
  await expect(page.getByText("Confidence unavailable for this run")).toBeVisible();
  await page.screenshot({ path: "../evidence/arnav/operator-ui-fixture.png", fullPage: true });

  await page.getByRole("button", { name: /Measure/i }).click();
  await expect(page.getByText("Select two visible points", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /Source frame/i }).click();
  await expect(page.getByText("SOURCE PREVIEW NOT DECLARED")).toBeVisible();
});

test("live local API smoke run reaches the viewer manifest", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Run API smoke fixture/i }).click();

  await expect(page.getByText("UI / orchestration fixture")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%")).toBeVisible();
  await expect(page.getByText(/not empirical reconstruction results/i)).toBeVisible();
  await expect(page.getByText("Confidence unavailable for this run")).toBeVisible();

  await expect(page).toHaveURL(/\?run=run_/);
  await page.reload();
  await expect(page.getByLabel("Interactive reconstruction viewport")).toBeVisible();
  await expect(page.getByText("90%")).toBeVisible();
});
