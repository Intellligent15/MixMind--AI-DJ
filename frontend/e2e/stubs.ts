import type { Page, Route } from "@playwright/test";

// Minimal fixture builders + a single `page.route("**/api/**")` installer that
// serves them. Everything the three flow pages fetch goes through
// `lib/api` → these endpoints, so stubbing them lets us drive the UI state
// machine with zero backend.

export type SongStatus =
  | "pending"
  | "downloading"
  | "downloaded"
  | "analyzing"
  | "analyzed"
  | "separating"
  | "transcribing"
  | "ready"
  | "failed";

export function song(overrides: {
  id: string;
  title: string;
  status: SongStatus;
  artist?: string | null;
  error_text?: string | null;
}) {
  return {
    id: overrides.id,
    youtube_video_id: `yt-${overrides.id}`,
    title: overrides.title,
    artist: overrides.artist ?? "Test Artist",
    duration_seconds: 180,
    thumbnail_url: null,
    audio_path: overrides.status === "pending" ? null : `audio/${overrides.id}.wav`,
    status: overrides.status,
    error_text: overrides.error_text ?? null,
    created_at: "2026-06-08T00:00:00Z",
    updated_at: "2026-06-08T00:00:00Z",
    has_stems: ["ready"].includes(overrides.status),
    has_transcription: ["ready"].includes(overrides.status),
  };
}

export function queue(id: string, songs: ReturnType<typeof song>[], locked = true) {
  return {
    id,
    locked,
    created_at: "2026-06-08T00:00:00Z",
    locked_at: locked ? "2026-06-08T00:00:01Z" : null,
    items: songs.map((s, i) => ({
      id: `item-${i}`,
      queue_id: id,
      position: i,
      song: s,
    })),
  };
}

export function mixPlan(overrides: {
  id: string;
  queue_id: string;
  from_song_id: string;
  to_song_id: string;
  status: "pending" | "rendering" | "ready" | "failed";
  error_text?: string | null;
}) {
  return {
    id: overrides.id,
    queue_id: overrides.queue_id,
    from_song_id: overrides.from_song_id,
    to_song_id: overrides.to_song_id,
    plan_json: null,
    rendered_audio_path: overrides.status === "ready" ? `mixes/${overrides.id}.wav` : null,
    status: overrides.status,
    error_text: overrides.error_text ?? null,
    created_at: "2026-06-08T00:00:00Z",
    updated_at: "2026-06-08T00:00:00Z",
  };
}

export function queueRender(overrides: {
  id: string;
  queue_id: string;
  status: "pending" | "rendering" | "ready" | "failed";
  error_text?: string | null;
  timeline?: unknown;
}) {
  return {
    id: overrides.id,
    queue_id: overrides.queue_id,
    rendered_audio_path:
      overrides.status === "ready" ? `queue_mixes/${overrides.queue_id}.flac` : null,
    status: overrides.status,
    error_text: overrides.error_text ?? null,
    timeline: overrides.timeline ?? null,
    created_at: "2026-06-08T00:00:00Z",
    updated_at: "2026-06-08T00:00:00Z",
  };
}

export type Scenario = {
  currentQueue: ReturnType<typeof queue> | null;
  songsById?: Record<string, ReturnType<typeof song>>;
  mix?: ReturnType<typeof queueRender> | null;
  mixPlans?: ReturnType<typeof mixPlan>[];
  // Test hooks — invoked when the UI POSTs a recovery action.
  onRetrySong?: (songId: string) => void;
  onStitch?: () => void;
};

function json(route: Route, status: number, body: unknown) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

export async function installApiStubs(page: Page, scenario: Scenario) {
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();

    // --- Audio: satisfy wavesurfer / <audio> with an empty 200 so a missing
    // backend doesn't error the page under test. ---
    if (path.endsWith("/audio")) {
      return route.fulfill({ status: 200, contentType: "audio/wav", body: "" });
    }

    if (path === "/api/queues/current") {
      return scenario.currentQueue
        ? json(route, 200, scenario.currentQueue)
        : json(route, 404, { detail: "no queue" });
    }

    // QueueBuilder auto-creates a queue when none exists.
    if (path === "/api/queues" && method === "POST") {
      return json(
        route,
        201,
        scenario.currentQueue ?? queue("q-new", [], false)
      );
    }

    const mixMatch = path.match(/^\/api\/queues\/([^/]+)\/mix$/);
    if (mixMatch) {
      return scenario.mix
        ? json(route, 200, scenario.mix)
        : json(route, 404, { detail: "no mix" });
    }

    const plansMatch = path.match(/^\/api\/queues\/([^/]+)\/mix_plans$/);
    if (plansMatch) {
      return json(route, 200, scenario.mixPlans ?? []);
    }

    const stitchMatch = path.match(/^\/api\/queues\/([^/]+)\/stitch$/);
    if (stitchMatch && method === "POST") {
      scenario.onStitch?.();
      return json(route, 202, { message: "Stitching started" });
    }

    const retryMatch = path.match(/^\/api\/songs\/([^/]+)\/retry$/);
    if (retryMatch && method === "POST") {
      const id = retryMatch[1];
      scenario.onRetrySong?.(id);
      const s = scenario.songsById?.[id];
      return json(route, 202, s ?? { id });
    }

    const stemsMatch = path.match(/^\/api\/songs\/([^/]+)\/stems$/);
    if (stemsMatch) {
      const id = stemsMatch[1];
      const s = scenario.songsById?.[id];
      return s?.has_stems
        ? json(route, 200, { id: `stems-${id}`, song_id: id, status: "separated" })
        : json(route, 404, { detail: "no stems" });
    }

    const transMatch = path.match(/^\/api\/songs\/([^/]+)\/transcription$/);
    if (transMatch) {
      const id = transMatch[1];
      const s = scenario.songsById?.[id];
      return s?.has_transcription
        ? json(route, 200, {
            id: `trans-${id}`,
            song_id: id,
            status: "success",
            segments: [],
          })
        : json(route, 404, { detail: "no transcription" });
    }

    if (path.match(/^\/api\/songs\/([^/]+)\/vocal_safe_regions$/)) {
      return json(route, 200, { regions: [] });
    }

    const songMatch = path.match(/^\/api\/songs\/([^/]+)$/);
    if (songMatch) {
      const s = scenario.songsById?.[songMatch[1]];
      return s ? json(route, 200, s) : json(route, 404, { detail: "no song" });
    }

    // Anything else (search, etc.) — empty success so nothing hangs.
    return json(route, 200, []);
  });
}
