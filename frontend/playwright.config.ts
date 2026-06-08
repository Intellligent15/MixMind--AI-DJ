import { defineConfig, devices } from "@playwright/test";

// Phase 11 e2e: drive the three-state UI machine (building → processing →
// playing) against STUBBED backend data, not a live pipeline. All flow pages
// are client components that fetch through `lib/api` → `NEXT_PUBLIC_API_BASE`,
// so every test intercepts `**/api/**` with `page.route` (see e2e/stubs.ts)
// and serves fixtures. No Postgres, no worker, no 2-minute render — the suite
// is fast and deterministic, which is the whole point of testing the state
// machine rather than the audio.
const PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command: `npm run dev -- --port ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    // Point the browser's API base at a host nothing listens on, so any
    // request we forgot to stub fails loudly instead of silently hitting a
    // real dev backend on :8000.
    env: { NEXT_PUBLIC_API_BASE: "http://127.0.0.1:9" },
  },
});
