import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
    channel: "chrome",
  },
  webServer: [
    {
      command:
        "env SIH_DATA_ROOT=/private/tmp/sih26158-e2e-projects .venv/bin/python -m uvicorn sih26158.app:app --host 127.0.0.1 --port 8000",
      cwd: "..",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command: "npm run dev -- --port 4173",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
});
