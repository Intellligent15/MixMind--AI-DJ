import { expect, test } from "@playwright/test";
import {
  installApiStubs,
  mixPlan,
  queue,
  queueRender,
  song,
} from "./stubs";

// End-to-end coverage of the three-state flow (building → processing →
// playing) and the Phase 11 failure/retry surfaces. Backend is fully stubbed
// (see stubs.ts), so these assert the UI STATE MACHINE deterministically.

test.describe("building state", () => {
  test("home page shows the queue builder with an empty queue", async ({ page }) => {
    await installApiStubs(page, { currentQueue: queue("q1", [], false) });
    await page.goto("/");
    // QueueBuilder renders its YouTube search affordance.
    await expect(page.getByPlaceholder(/search/i)).toBeVisible();
  });
});

test.describe("processing state", () => {
  const sReady = song({ id: "s1", title: "First Track", status: "ready" });
  const sProcessing = song({ id: "s2", title: "Second Track", status: "separating" });

  test("shows per-song and per-transition progress while rendering", async ({
    page,
  }) => {
    const q = queue("q1", [sReady, sProcessing]);
    await installApiStubs(page, {
      currentQueue: q,
      songsById: { s1: sReady, s2: sProcessing },
      mix: queueRender({ id: "r1", queue_id: "q1", status: "rendering" }),
      mixPlans: [
        mixPlan({
          id: "p1",
          queue_id: "q1",
          from_song_id: "s1",
          to_song_id: "s2",
          status: "rendering",
        }),
      ],
    });
    await page.goto("/processing");

    await expect(page.getByText("First Track", { exact: true })).toBeVisible();
    await expect(page.getByText("Second Track", { exact: true })).toBeVisible();
    // Gate reflects the rendering mix, and the per-transition list shows.
    await expect(page.getByText(/Rendering continuous mix/i)).toBeVisible();
    await expect(page.getByText(/Transitions \(/i)).toBeVisible();
  });

  test("auto-redirects to the player once the continuous mix is ready", async ({
    page,
  }) => {
    const s1 = song({ id: "s1", title: "First Track", status: "ready" });
    const s2 = song({ id: "s2", title: "Second Track", status: "ready" });
    const q = queue("q1", [s1, s2]);
    await installApiStubs(page, {
      currentQueue: q,
      songsById: { s1, s2 },
      mix: queueRender({
        id: "r1",
        queue_id: "q1",
        status: "ready",
        timeline: {
          duration: 360,
          songs: [
            { index: 0, song_id: "s1", title: "First Track", artist: "a", start: 0, end: 180 },
            { index: 1, song_id: "s2", title: "Second Track", artist: "a", start: 180, end: 360 },
          ],
          transitions: [],
        },
      }),
      mixPlans: [
        mixPlan({
          id: "p1",
          queue_id: "q1",
          from_song_id: "s1",
          to_song_id: "s2",
          status: "ready",
        }),
      ],
    });
    await page.goto("/processing");
    // The honest gate auto-advances on mix-ready.
    await page.waitForURL("**/player", { timeout: 10_000 });
  });
});

test.describe("processing failure + retry", () => {
  test("a failed song shows its error and a Retry that re-dispatches", async ({
    page,
  }) => {
    const failed = song({
      id: "s1",
      title: "Broken Track",
      status: "failed",
      error_text: "analysis failed: librosa exploded",
    });
    const ok = song({ id: "s2", title: "Fine Track", status: "ready" });
    const q = queue("q1", [failed, ok]);

    let retried: string | null = null;
    await installApiStubs(page, {
      currentQueue: q,
      songsById: { s1: failed, s2: ok },
      mix: null,
      mixPlans: [
        mixPlan({
          id: "p1",
          queue_id: "q1",
          from_song_id: "s1",
          to_song_id: "s2",
          status: "pending",
        }),
      ],
      onRetrySong: (id) => {
        retried = id;
      },
    });

    await page.goto("/processing");
    await expect(page.getByText(/analysis failed: librosa exploded/)).toBeVisible();
    await expect(page.getByText(/A song failed to process/i)).toBeVisible();

    await page.getByRole("button", { name: "Retry", exact: true }).click();
    await expect.poll(() => retried).toBe("s1");
  });

  test("a failed transition exposes a queue-level Retry rendering", async ({
    page,
  }) => {
    const s1 = song({ id: "s1", title: "First Track", status: "ready" });
    const s2 = song({ id: "s2", title: "Second Track", status: "ready" });
    const q = queue("q1", [s1, s2]);

    let stitched = false;
    await installApiStubs(page, {
      currentQueue: q,
      songsById: { s1, s2 },
      mix: queueRender({
        id: "r1",
        queue_id: "q1",
        status: "failed",
        error_text: "stitch failed",
      }),
      mixPlans: [
        mixPlan({
          id: "p1",
          queue_id: "q1",
          from_song_id: "s1",
          to_song_id: "s2",
          status: "failed",
          error_text: "render failed",
        }),
      ],
      onStitch: () => {
        stitched = true;
      },
    });

    await page.goto("/processing");
    const retryBtn = page.getByRole("button", { name: /Retry rendering/i });
    await expect(retryBtn).toBeVisible();
    await retryBtn.click();
    await expect.poll(() => stitched).toBe(true);
  });
});

test.describe("playing state", () => {
  test("player renders the stitched mix when ready", async ({ page }) => {
    const s1 = song({ id: "s1", title: "First Track", status: "ready" });
    const s2 = song({ id: "s2", title: "Second Track", status: "ready" });
    const q = queue("q1", [s1, s2]);
    await installApiStubs(page, {
      currentQueue: q,
      songsById: { s1, s2 },
      mix: queueRender({
        id: "r1",
        queue_id: "q1",
        status: "ready",
        timeline: {
          duration: 360,
          songs: [
            { index: 0, song_id: "s1", title: "First Track", artist: "a", start: 0, end: 180 },
            { index: 1, song_id: "s2", title: "Second Track", artist: "a", start: 180, end: 360 },
          ],
          transitions: [],
        },
      }),
      mixPlans: [],
    });
    await page.goto("/player");
    // The player mounts and exposes a download link for the stitched mix.
    await expect(
      page.getByRole("link", { name: /download/i }).first()
    ).toBeVisible({ timeout: 10_000 });
  });
});
