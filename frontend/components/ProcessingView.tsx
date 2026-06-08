"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  api,
  isStatusError,
  type MixPlan,
  type Queue,
  type QueueRender,
  type Song,
  type SongStatus,
  type Stems,
  type Transcription,
} from "@/lib/api";

// Songs at or past this point can be played back immediately via the
// player's per-song hard-cut fallback (stems/transcription aren't needed to
// just play a song). Used to enable the "Play now (hard cut)" escape hatch.
const PLAYABLE_STATUSES: ReadonlyArray<SongStatus> = [
  "downloaded",
  "analyzing",
  "analyzed",
  "separating",
  "transcribing",
  "ready",
];

// Logical pipeline progression. "separated" and "transcribed" aren't
// SongStatus values — they're derived from the presence of a Stems or
// Transcription row. The worker bounces Song.status back to `analyzed`
// after separating, and to `ready` after transcribing, in separate
// transactions from inserting the row.
type PipelineStep =
  | "pending"
  | "downloading"
  | "downloaded"
  | "analyzing"
  | "analyzed"
  | "separating"
  | "separated"
  | "transcribing"
  | "transcribed";

const PIPELINE_STEPS: { key: PipelineStep; label: string }[] = [
  { key: "pending", label: "queued" },
  { key: "downloading", label: "downloading" },
  { key: "downloaded", label: "downloaded" },
  { key: "analyzing", label: "analyzing" },
  { key: "analyzed", label: "analyzed" },
  { key: "separating", label: "separating" },
  { key: "separated", label: "separated" },
  { key: "transcribing", label: "transcribing" },
  { key: "transcribed", label: "ready" },
];

function stepIndex(
  status: SongStatus,
  hasStems: boolean,
  hasTranscription: boolean
): number {
  if (status === "failed") return -1;
  if (hasTranscription) return PIPELINE_STEPS.length - 1; // "transcribed"
  if (status === "transcribing") {
    return PIPELINE_STEPS.findIndex((s) => s.key === "transcribing");
  }
  if (hasStems) {
    return PIPELINE_STEPS.findIndex((s) => s.key === "separated");
  }
  if (status === "separating") {
    return PIPELINE_STEPS.findIndex((s) => s.key === "separating");
  }
  // analyzed/ready without stems sit at the "analyzed" step.
  if (status === "analyzed" || status === "ready") {
    return PIPELINE_STEPS.findIndex((s) => s.key === "analyzed");
  }
  return PIPELINE_STEPS.findIndex((s) => s.key === status);
}

function statusBadgeClass(status: SongStatus): string {
  if (status === "failed") return "bg-red-500/20";
  if (PLAYABLE_STATUSES.includes(status)) return "bg-green-500/20";
  if (status === "downloaded") return "bg-blue-500/20";
  return "bg-yellow-500/20";
}

// Songs whose status sits at `separating` or `transcribing` for longer
// than this get a yellow "worker may be down" warning. The native Celery
// worker is the only thing that can pick up either job; if it isn't
// running, the message just sits in Redis indefinitely.
const WORKER_STUCK_WARN_MS = 120_000;

export function ProcessingView() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const queue = useQuery<Queue | null>({
    queryKey: ["queue", "current"],
    queryFn: async () => {
      try {
        return await api.getCurrentQueue();
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
  });

  const items = queue.data?.items ?? [];

  // Per-song stems lookup. 404 just means "not separated yet" → null.
  // This page only renders for a locked queue, and lock-time fan-out
  // chains separate after analyze, so every song is expected to get a
  // Stems row — keep polling until one lands.
  const stemsQueries = useQueries({
    queries: items.map((item) => ({
      queryKey: ["stems", item.song.id],
      queryFn: async (): Promise<Stems | null> => {
        try {
          return await api.getStems(item.song.id);
        } catch (err) {
          if (isStatusError(err, 404)) return null;
          throw err;
        }
      },
      retry: false,
      refetchInterval: (q: { state: { data?: Stems | null } }) =>
        q.state.data ? false : 1500,
    })),
  });

  // Per-song transcription lookup. 404 = "not transcribed yet" → null.
  // Lock-time fan-out chains transcribe after separate, so we keep polling
  // until a Transcription row lands (success OR skipped_instrumental both
  // count as "done" for pipeline progress).
  const transcriptionQueries = useQueries({
    queries: items.map((item) => ({
      queryKey: ["transcription", item.song.id],
      queryFn: async (): Promise<Transcription | null> => {
        try {
          return await api.getTranscription(item.song.id);
        } catch (err) {
          if (isStatusError(err, 404)) return null;
          throw err;
        }
      },
      retry: false,
      refetchInterval: (q: { state: { data?: Transcription | null } }) =>
        q.state.data ? false : 1500,
    })),
  });

  const songQueries = useQueries({
    queries: items.map((item, idx) => ({
      queryKey: ["song", item.song.id],
      queryFn: () => api.getSong(item.song.id),
      initialData: item.song,
      refetchInterval: (q: { state: { data?: Song } }) => {
        const s = q.state.data;
        if (!s) return 1000;
        if (s.status === "failed" || s.status === "ready") return false;
        return 1000;
      },
    })),
  });

  const songs: Song[] = useMemo(
    () => songQueries.map((q, i) => q.data ?? items[i].song),
    [songQueries, items]
  );

  // Track when we first observed each song in `separating` or
  // `transcribing` so we can surface the "worker may be down" hint after
  // WORKER_STUCK_WARN_MS. Keyed by "songId:status" so a stuck separation
  // followed by a stuck transcription warn independently.
  const workerStuckSinceRef = useRef<Map<string, number>>(new Map());
  const [, forceTick] = useState(0);
  useEffect(() => {
    const now = Date.now();
    let dirty = false;
    const seen = workerStuckSinceRef.current;
    const liveKeys = new Set<string>();
    for (const s of songs) {
      if (s.status === "separating" || s.status === "transcribing") {
        const k = `${s.id}:${s.status}`;
        liveKeys.add(k);
        if (!seen.has(k)) {
          seen.set(k, now);
          dirty = true;
        }
      }
    }
    for (const k of seen.keys()) {
      if (!liveKeys.has(k)) {
        seen.delete(k);
        dirty = true;
      }
    }
    if (dirty) forceTick((n) => n + 1);
  }, [songs]);

  // Re-render once per 5s while any song is still in a worker-bound
  // status, so the warning appears even if nothing else changes.
  useEffect(() => {
    const anyStuck = songs.some(
      (s) => s.status === "separating" || s.status === "transcribing"
    );
    if (!anyStuck) return;
    const t = setInterval(() => forceTick((n) => n + 1), 5000);
    return () => clearInterval(t);
  }, [songs]);

  const queueId = queue.data?.id;
  const hasTransitions = items.length >= 2;

  // Per-transition (MixPlan) progress. Plans are seeded at lock; they flip
  // pending → rendering → ready as the eager auto-stitch renders them (which
  // only starts once every song is `ready`). Poll until all are terminal.
  const mixPlansQuery = useQuery<MixPlan[]>({
    queryKey: ["mix_plans", queueId],
    queryFn: () => (queueId ? api.listMixPlansForQueue(queueId) : Promise.resolve([])),
    enabled: !!queueId && hasTransitions,
    refetchInterval: (q) => {
      const plans = q.state.data ?? [];
      if (plans.length === 0) return 1000;
      const allDone = plans.every(
        (p) => p.status === "ready" || p.status === "failed"
      );
      return allDone ? false : 1000;
    },
  });
  const mixPlans = mixPlansQuery.data ?? [];

  // The stitched continuous mix. Poll while it's being produced.
  const mixQuery = useQuery<QueueRender | null>({
    queryKey: ["mix", queueId],
    queryFn: () => (queueId ? api.getQueueMix(queueId) : Promise.resolve(null)),
    enabled: !!queueId && hasTransitions,
    refetchInterval: (q) =>
      q.state.data?.status === "pending" || q.state.data?.status === "rendering"
        ? 1000
        : q.state.data?.status === "ready"
          ? false
          : 1500,
  });
  const mix = mixQuery.data;

  // Retry a failed song: re-dispatch from the stage it died on, then resume
  // polling (the song's refetchInterval stopped when it hit `failed`).
  const retrySongMutation = useMutation({
    mutationFn: (songId: string) => api.retrySong(songId),
    onSuccess: (_data, songId) => {
      queryClient.invalidateQueries({ queryKey: ["song", songId] });
      queryClient.invalidateQueries({ queryKey: ["stems", songId] });
      queryClient.invalidateQueries({ queryKey: ["transcription", songId] });
    },
  });

  // Retry the continuous mix: re-renders pending+failed transitions and
  // re-stitches (the existing chord recovery path). Covers both a failed
  // transition and a failed stitch.
  const retryMixMutation = useMutation({
    mutationFn: () =>
      queueId ? api.stitchQueue(queueId) : Promise.resolve({ message: "" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mix", queueId] });
      queryClient.invalidateQueries({ queryKey: ["mix_plans", queueId] });
    },
  });

  // Honest playback gate (Phase 10):
  //   * The continuous mix is the product, so auto-advancing once it's
  //     `ready` gives the best experience — that's the eager-stitch payoff.
  //   * A 1-song queue has no transitions/mix, so it can start as soon as
  //     the song is playable.
  //   * If a song or the mix fails, the mix never goes `ready`; the
  //     "Play now (hard cut)" button below is the always-available fallback
  //     (the player upgrades to the stitched mix if it lands later).
  const headPlayable =
    songs.length > 0 && PLAYABLE_STATUSES.includes(songs[0].status);
  const mixReady = mix?.status === "ready";
  const readyTransitions = mixPlans.filter((p) => p.status === "ready").length;
  const canAutoStart = mixReady || (!hasTransitions && headPlayable);

  useEffect(() => {
    if (canAutoStart) {
      // Small delay so the user sees the gate flip green before redirect.
      const t = setTimeout(() => router.push("/player"), 750);
      return () => clearTimeout(t);
    }
  }, [canAutoStart, router]);

  const transitionCount = mixPlans.length || Math.max(0, items.length - 1);
  const allSongsReady =
    songs.length > 0 && songs.every((s) => s.status === "ready");
  const anySongFailed = songs.some((s) => s.status === "failed");
  const anyTransitionFailed = mixPlans.some((p) => p.status === "failed");
  const mixFailed = mix?.status === "failed";
  // A mix retry only makes sense once songs are fixed — a failed transition
  // can't re-render until its two songs are `ready`.
  const canRetryMix =
    hasTransitions && (mixFailed || anyTransitionFailed) && !anySongFailed;
  const gateText = canAutoStart
    ? "Ready — starting playback…"
    : anySongFailed
      ? "A song failed to process — retry it below to continue"
      : mixFailed
        ? "Mix render failed — retry rendering, or play hard-cut now"
        : anyTransitionFailed
          ? "A transition failed — retry rendering below"
          : mix?.status === "rendering" || mix?.status === "pending"
            ? "Rendering continuous mix…"
            : allSongsReady
              ? "All songs ready — preparing the mix…"
              : "Processing songs…";

  const songTitleById = (id: string) =>
    songs.find((s) => s.id === id)?.title ?? "—";

  if (queue.isLoading) {
    return <p className="text-sm opacity-70">Loading…</p>;
  }
  if (!queue.data) {
    return (
      <p className="text-sm opacity-70">
        No queue exists. Start building one on the{" "}
        <Link href="/" className="underline">
          home page
        </Link>
        .
      </p>
    );
  }
  if (!queue.data.locked) {
    return (
      <p className="text-sm opacity-70">
        Queue not locked yet. Go back to{" "}
        <Link href="/" className="underline">
          /
        </Link>{" "}
        and click Done.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="border rounded p-4 flex flex-col gap-3">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <p className="text-sm opacity-70">Playback gate</p>
            <p className="font-medium">{gateText}</p>
          </div>
          {hasTransitions && (
            <div className="text-2xl tabular-nums">
              {readyTransitions}/{transitionCount}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => router.push("/player")}
            disabled={!headPlayable}
            className="px-4 py-2 border rounded text-sm hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-40"
          >
            {mixReady ? "Open player" : "Play now (hard cut)"}
          </button>
          {canRetryMix && (
            <button
              type="button"
              onClick={() => retryMixMutation.mutate()}
              disabled={retryMixMutation.isPending}
              className="px-4 py-2 border rounded text-sm border-amber-500/60 text-amber-700 dark:text-amber-400 hover:bg-amber-500/10 disabled:opacity-40"
            >
              {retryMixMutation.isPending ? "Retrying…" : "Retry rendering"}
            </button>
          )}
          {hasTransitions && !mixReady && !canRetryMix && !anySongFailed && (
            <span className="text-xs opacity-60">
              The mix keeps rendering in the background — the player upgrades
              to it automatically when it&apos;s ready.
            </span>
          )}
        </div>
      </div>

      <ul className="flex flex-col gap-3">
        {songs.map((song, idx) => {
          const hasStems = !!stemsQueries[idx]?.data;
          const hasTranscription = !!transcriptionQueries[idx]?.data;
          const step = stepIndex(song.status, hasStems, hasTranscription);
          const pct =
            step < 0
              ? 0
              : (step / (PIPELINE_STEPS.length - 1)) * 100;
          const stepLabel =
            step < 0 ? "failed" : PIPELINE_STEPS[step].label;
          // Queue items are unique even when the same song is queued twice;
          // fall back to a composite if the item isn't there yet.
          const key = items[idx]?.id ?? `${song.id}:${idx}`;
          const stuckSince =
            song.status === "separating" || song.status === "transcribing"
              ? workerStuckSinceRef.current.get(`${song.id}:${song.status}`)
              : undefined;
          const stuck =
            stuckSince !== undefined &&
            Date.now() - stuckSince > WORKER_STUCK_WARN_MS;
          return (
            <li key={key} className="border rounded p-3 flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <span className="opacity-60 w-6 tabular-nums">{idx + 1}</span>
                {song.thumbnail_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={song.thumbnail_url}
                    alt=""
                    className="w-16 h-10 object-cover rounded"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{song.title}</p>
                  <p className="text-xs opacity-70 truncate">
                    {song.artist ?? "—"}
                  </p>
                </div>
                <span
                  className={`text-xs px-2 py-1 rounded ${statusBadgeClass(song.status)}`}
                >
                  {stepLabel}
                </span>
                {song.status === "failed" && (
                  <button
                    type="button"
                    onClick={() => retrySongMutation.mutate(song.id)}
                    disabled={
                      retrySongMutation.isPending &&
                      retrySongMutation.variables === song.id
                    }
                    className="text-xs px-2 py-1 rounded border border-amber-500/60 text-amber-700 dark:text-amber-400 hover:bg-amber-500/10 disabled:opacity-40"
                  >
                    {retrySongMutation.isPending &&
                    retrySongMutation.variables === song.id
                      ? "Retrying…"
                      : "Retry"}
                  </button>
                )}
              </div>
              <div className="h-1 bg-black/10 dark:bg-white/10 rounded overflow-hidden">
                <div
                  className={
                    song.status === "failed"
                      ? "h-full bg-red-500"
                      : "h-full bg-green-500 transition-all"
                  }
                  style={{ width: `${song.status === "failed" ? 100 : pct}%` }}
                />
              </div>
              {song.status === "failed" && song.error_text && (
                <p className="text-xs text-red-500 break-words">
                  {song.error_text}
                </p>
              )}
              {stuck && (
                <p className="text-xs text-amber-700 dark:text-amber-400">
                  {song.status === "separating" ? "Separation" : "Transcription"}{" "}
                  has been pending for &gt;
                  {Math.floor(WORKER_STUCK_WARN_MS / 1000)}s. Demucs and
                  Whisper run on the native worker (MPS) — check that{" "}
                  <code>./start-dev.sh</code> is running.
                </p>
              )}
            </li>
          );
        })}
      </ul>

      {hasTransitions && (
        <section className="flex flex-col gap-2">
          <p className="text-xs opacity-60">
            Transitions ({readyTransitions}/{transitionCount} rendered)
          </p>
          <ul className="flex flex-col gap-2">
            {mixPlans.map((mp, idx) => {
              const label =
                mp.status === "rendering"
                  ? "rendering"
                  : mp.status === "ready"
                    ? "ready"
                    : mp.status === "failed"
                      ? "failed"
                      : allSongsReady
                        ? "queued"
                        : "waiting for songs";
              const pct =
                mp.status === "ready"
                  ? 100
                  : mp.status === "rendering"
                    ? 55
                    : mp.status === "failed"
                      ? 100
                      : 8;
              const badge =
                mp.status === "failed"
                  ? "bg-red-500/20"
                  : mp.status === "ready"
                    ? "bg-green-500/20"
                    : "bg-yellow-500/20";
              return (
                <li
                  key={mp.id}
                  className="border rounded p-3 flex flex-col gap-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="opacity-60 w-6 tabular-nums">
                      {idx + 1}→{idx + 2}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">
                        {songTitleById(mp.from_song_id)}{" "}
                        <span className="opacity-50">→</span>{" "}
                        {songTitleById(mp.to_song_id)}
                      </p>
                      {mp.status === "failed" && mp.error_text && (
                        <p className="text-xs text-red-500 truncate">
                          {mp.error_text}
                        </p>
                      )}
                    </div>
                    <span className={`text-xs px-2 py-1 rounded ${badge}`}>
                      {label}
                    </span>
                  </div>
                  <div className="h-1 bg-black/10 dark:bg-white/10 rounded overflow-hidden">
                    <div
                      className={
                        mp.status === "failed"
                          ? "h-full bg-red-500"
                          : "h-full bg-green-500 transition-all"
                      }
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          {mix?.status === "failed" && mix.error_text && (
            <p className="text-xs text-red-500">
              Continuous mix failed: {mix.error_text}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
