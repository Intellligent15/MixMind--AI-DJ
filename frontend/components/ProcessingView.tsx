"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, isStatusError, type Queue, type Song, type SongStatus } from "@/lib/api";

// Songs at or past this point in the pipeline are considered "ready enough"
// for the playback gate. Phase 4 only goes as far as analyzed.
const PLAYABLE_STATUSES: ReadonlyArray<SongStatus> = [
  "analyzed",
  "separating",
  "transcribing",
  "ready",
];
// Playback starts once this many songs from the head of the queue are analyzed.
// Phase 7+ will replace this gate with "first transition rendered".
const GATE_ANALYZED = 2;

const PIPELINE_STEPS: { key: SongStatus; label: string }[] = [
  { key: "pending", label: "queued" },
  { key: "downloading", label: "downloading" },
  { key: "downloaded", label: "downloaded" },
  { key: "analyzing", label: "analyzing" },
  { key: "analyzed", label: "analyzed" },
];

function stepIndex(status: SongStatus): number {
  // Maps post-Phase-4 statuses back to "analyzed" so progress bars don't go
  // backwards in Phase 5+.
  if (status === "failed") return -1;
  if (PLAYABLE_STATUSES.includes(status)) return PIPELINE_STEPS.length - 1;
  return PIPELINE_STEPS.findIndex((s) => s.key === status);
}

function statusBadgeClass(status: SongStatus): string {
  if (status === "failed") return "bg-red-500/20";
  if (PLAYABLE_STATUSES.includes(status)) return "bg-green-500/20";
  if (status === "downloaded") return "bg-blue-500/20";
  return "bg-yellow-500/20";
}

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

  // Poll each song individually so we get fast updates without re-fetching
  // the whole queue. Stops polling a song once it reaches a terminal status.
  const songQueries = useQueries({
    queries: items.map((item) => ({
      queryKey: ["song", item.song.id],
      queryFn: () => api.getSong(item.song.id),
      initialData: item.song,
      refetchInterval: (q: { state: { data?: Song } }) => {
        const s = q.state.data;
        if (!s) return 1000;
        if (s.status === "failed") return false;
        if (PLAYABLE_STATUSES.includes(s.status)) return false;
        return 1000;
      },
    })),
  });

  const songs: Song[] = useMemo(
    () => songQueries.map((q, i) => q.data ?? items[i].song),
    [songQueries, items]
  );

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
        <a href="/" className="underline">
          home page
        </a>
        .
      </p>
    );
  }
  if (!queue.data.locked) {
    return (
      <p className="text-sm opacity-70">
        Queue not locked yet. Go back to{" "}
        <a href="/" className="underline">
          /
        </a>{" "}
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
          const step = stepIndex(song.status);
          const pct =
            step < 0
              ? 0
              : (step / (PIPELINE_STEPS.length - 1)) * 100;
          // Queue items are unique even when the same song is queued twice;
          // fall back to a composite if the item isn't there yet.
          const key = items[idx]?.id ?? `${song.id}:${idx}`;
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
                  {song.status}
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
            </li>
          );
        })}
      </ul>
    </div>
  );
}
