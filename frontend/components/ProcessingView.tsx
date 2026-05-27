"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQueries, useQuery } from "@tanstack/react-query";
import {
  api,
  isStatusError,
  type Queue,
  type Song,
  type SongStatus,
  type Stems,
  type Transcription,
} from "@/lib/api";

// Songs at or past this point in the pipeline are considered "ready enough"
// for the playback gate. Stems aren't required for hard-cut playback.
const PLAYABLE_STATUSES: ReadonlyArray<SongStatus> = [
  "analyzed",
  "separating",
  "transcribing",
  "ready",
];
// Playback starts once this many songs from the head of the queue are analyzed.
// Phase 7+ will replace this gate with "first transition rendered".
const GATE_ANALYZED = 2;

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

  const analyzedHeadCount = useMemo(() => {
    let n = 0;
    for (const s of songs) {
      if (PLAYABLE_STATUSES.includes(s.status)) n += 1;
      else break;
    }
    return n;
  }, [songs]);

  const requiredAnalyzed = Math.min(GATE_ANALYZED, songs.length);
  const gateMet = analyzedHeadCount >= requiredAnalyzed && songs.length > 0;

  useEffect(() => {
    if (gateMet) {
      // Small delay to let the user see the gate flip green before redirect.
      const t = setTimeout(() => router.push("/player"), 750);
      return () => clearTimeout(t);
    }
  }, [gateMet, router]);

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
      <div className="border rounded p-4 flex items-center gap-4">
        <div className="flex-1">
          <p className="text-sm opacity-70">Playback gate</p>
          <p className="font-medium">
            {gateMet
              ? "Ready — starting playback…"
              : `Waiting for ${analyzedHeadCount}/${requiredAnalyzed} songs analyzed`}
          </p>
        </div>
        <div className="text-2xl tabular-nums">
          {analyzedHeadCount}/{requiredAnalyzed}
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
    </div>
  );
}
