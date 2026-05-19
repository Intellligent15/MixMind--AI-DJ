"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  api,
  isStatusError,
  type Song,
  type Stems,
  type Transcription,
} from "@/lib/api";
import {
  isFullyProcessed,
  maySoonHaveStems,
  maySoonHaveTranscription,
} from "@/lib/song-status";

const SEPARATABLE: ReadonlyArray<Song["status"]> = [
  "analyzed",
  "ready",
  "failed",
];
const TRANSCRIBABLE: ReadonlyArray<Song["status"]> = [
  "analyzed",
  "ready",
  "failed",
];

// A song's "displayed" status is derived: once a Stems / Transcription row
// exists, show `separated` / `transcribed` even though the worker bounces
// the Song row back to `analyzed` or `ready`. Mirrors ProcessingView.
type DisplayStatus = Song["status"] | "separated" | "transcribed";

function displayStatus(
  song: Song,
  hasStems: boolean,
  hasTranscription: boolean
): DisplayStatus {
  if (
    hasTranscription &&
    (song.status === "analyzed" || song.status === "ready")
  ) {
    return "transcribed";
  }
  if (hasStems && (song.status === "analyzed" || song.status === "ready")) {
    return "separated";
  }
  return song.status;
}

function badgeClass(status: DisplayStatus): string {
  if (status === "failed") return "bg-red-500/20";
  if (status === "transcribed") return "bg-emerald-500/40";
  if (status === "separated") return "bg-emerald-500/30";
  if (status === "analyzed" || status === "ready") return "bg-green-500/20";
  if (status === "downloaded") return "bg-blue-500/20";
  return "bg-yellow-500/20";
}

export function DownloadedSongs() {
  const qc = useQueryClient();
  // Tracks song ids the user just clicked "Separate" / "Transcribe" on —
  // keeps polling alive across the worker-status flicker until the Stems /
  // Transcription row actually shows up. Cleared automatically.
  const [pendingStems, setPendingStems] = useState<Set<string>>(new Set());
  const [pendingTranscription, setPendingTranscription] = useState<
    Set<string>
  >(new Set());

  const songs = useQuery({
    queryKey: ["songs"],
    queryFn: api.listSongs,
    // Poll while any song is mid-pipeline. The worker bounces Song.status
    // through `analyzed` twice during the queue-lock chain (post-analyze
    // pre-separate, post-separate pre-transcribe), so we can't treat
    // `analyzed` as terminal — instead, we treat the song as in-flight
    // until it lands at `ready` with both a Stems row and a Transcription
    // row. The stemsBySongId / transcriptionBySongId maps are computed
    // below and re-captured by this closure on every render.
    refetchInterval: (q) => {
      const data = q.state.data as Song[] | undefined;
      if (!data) return 1000;
      const anyInFlight = data.some(
        (s) =>
          !isFullyProcessed(
            s,
            !!stemsBySongId.get(s.id),
            !!transcriptionBySongId.get(s.id),
          ),
      );
      if (anyInFlight) return 1000;
      if (pendingStems.size > 0 || pendingTranscription.size > 0) return 1000;
      return false;
    },
  });

  // Per-song stems lookup. 404 = "no stems yet" → null. Polls during
  // separating and during analyzed/ready (where the worker writes the
  // Stems row + rolls Song.status back in a single transaction, so the
  // status flip is the user-visible signal that stems exist). Slower
  // 3 s cadence for the analyzed/ready window so a lonely song that
  // never gets separated doesn't busy-loop forever.
  const stemsQueries = useQueries({
    queries: (songs.data ?? []).map((song) => ({
      queryKey: ["stems", song.id],
      queryFn: async (): Promise<Stems | null> => {
        try {
          return await api.getStems(song.id);
        } catch (err) {
          if (isStatusError(err, 404)) return null;
          throw err;
        }
      },
      retry: false,
      refetchInterval: (q: { state: { data?: Stems | null } }) => {
        if (q.state.data) return false;
        if (song.status === "failed") return false;
        if (pendingStems.has(song.id) || song.status === "separating") {
          return 1500;
        }
        if (maySoonHaveStems(song)) return 3000;
        return false;
      },
    })),
  });

  const stemsBySongId = useMemo(() => {
    const m = new Map<string, Stems | null>();
    (songs.data ?? []).forEach((s, i) => {
      m.set(s.id, stemsQueries[i]?.data ?? null);
    });
    return m;
  }, [songs.data, stemsQueries]);

  // Per-song transcription lookup. 404 = "not transcribed yet" → null.
  // Mirrors the stems polling: fast during transcribing / user-clicked,
  // slower 3 s during analyzed/ready (where the chain may eventually
  // dispatch transcribe). The worker writes the Transcription row +
  // sets Song.status to "ready" in one transaction, so the row + status
  // become visible together.
  const transcriptionQueries = useQueries({
    queries: (songs.data ?? []).map((song) => ({
      queryKey: ["transcription", song.id],
      queryFn: async (): Promise<Transcription | null> => {
        try {
          return await api.getTranscription(song.id);
        } catch (err) {
          if (isStatusError(err, 404)) return null;
          throw err;
        }
      },
      retry: false,
      refetchInterval: (q: { state: { data?: Transcription | null } }) => {
        if (q.state.data) return false;
        if (song.status === "failed") return false;
        if (
          pendingTranscription.has(song.id) ||
          song.status === "transcribing"
        ) {
          return 1500;
        }
        if (maySoonHaveTranscription(song)) return 3000;
        return false;
      },
    })),
  });

  const transcriptionBySongId = useMemo(() => {
    const m = new Map<string, Transcription | null>();
    (songs.data ?? []).forEach((s, i) => {
      m.set(s.id, transcriptionQueries[i]?.data ?? null);
    });
    return m;
  }, [songs.data, transcriptionQueries]);

  // Auto-clear pending* entries once the corresponding row lands.
  useEffect(() => {
    setPendingStems((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const id of prev) {
        if (stemsBySongId.get(id)) {
          next.delete(id);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [stemsBySongId]);

  useEffect(() => {
    setPendingTranscription((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const id of prev) {
        if (transcriptionBySongId.get(id)) {
          next.delete(id);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [transcriptionBySongId]);

  const analyze = useMutation({
    mutationFn: (id: string) => api.triggerAnalyze(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["songs"] }),
  });

  const separate = useMutation({
    mutationFn: (id: string) => api.triggerSeparate(id),
    onMutate: (id: string) => {
      // Optimistic: flip the visible row to `separating` immediately so
      // there's instant feedback instead of waiting on the next poll.
      setPendingStems((prev) => new Set(prev).add(id));
      qc.setQueryData<Song[] | undefined>(["songs"], (list) =>
        list?.map((s) => (s.id === id ? { ...s, status: "separating" } : s))
      );
    },
    onSuccess: (_data, id) => {
      // Re-sync with server. The optimistic `separating` stays in cache
      // until the next refetch returns the real status.
      qc.invalidateQueries({ queryKey: ["songs"] });
      qc.invalidateQueries({ queryKey: ["stems", id] });
    },
    onError: (_err, id) => {
      // Roll back: remove from pending and re-fetch.
      setPendingStems((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      qc.invalidateQueries({ queryKey: ["songs"] });
    },
  });

  const transcribe = useMutation({
    mutationFn: (id: string) => api.triggerTranscribe(id),
    onMutate: (id: string) => {
      setPendingTranscription((prev) => new Set(prev).add(id));
      qc.setQueryData<Song[] | undefined>(["songs"], (list) =>
        list?.map((s) => (s.id === id ? { ...s, status: "transcribing" } : s))
      );
    },
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["songs"] });
      qc.invalidateQueries({ queryKey: ["transcription", id] });
    },
    onError: (_err, id) => {
      setPendingTranscription((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      qc.invalidateQueries({ queryKey: ["songs"] });
    },
  });

  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-semibold">Library</h2>
      {songs.isLoading && <p className="text-sm opacity-70">Loading…</p>}
      {songs.data?.length === 0 && (
        <p className="text-sm opacity-70">No songs yet. Search and add one.</p>
      )}
      <ul className="flex flex-col gap-3">
        {songs.data?.map((s) => {
          const hasStems = !!stemsBySongId.get(s.id);
          const hasTranscription = !!transcriptionBySongId.get(s.id);
          const display = displayStatus(s, hasStems, hasTranscription);
          const isMidSeparation =
            s.status === "separating" || pendingStems.has(s.id);
          const isMidTranscription =
            s.status === "transcribing" || pendingTranscription.has(s.id);
          // Transcription requires stems to exist first (the worker reads
          // vocals_path off the Stems row); gating the button avoids a 409
          // from the API.
          const transcribable =
            hasStems && TRANSCRIBABLE.includes(s.status);
          return (
            <li key={s.id} className="border rounded p-3 flex flex-col gap-2">
              <div className="flex items-center gap-3">
                {s.thumbnail_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={s.thumbnail_url}
                    alt=""
                    className="w-20 h-12 object-cover rounded"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{s.title}</p>
                  <p className="text-xs opacity-70 truncate">{s.artist ?? "—"}</p>
                </div>
                <span
                  className={"text-xs px-2 py-1 rounded " + badgeClass(display)}
                >
                  {display}
                </span>
              </div>
              {(s.status === "downloaded" ||
                s.status === "analyzed" ||
                s.status === "ready") && (
                <audio
                  controls
                  preload="none"
                  src={api.audioUrl(s.id)}
                  className="w-full"
                />
              )}
              <div className="flex gap-2">
                {s.status === "downloaded" && (
                  <button
                    type="button"
                    onClick={() => analyze.mutate(s.id)}
                    disabled={analyze.isPending}
                    className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
                  >
                    Analyze
                  </button>
                )}
                {s.status === "failed" && (
                  <button
                    type="button"
                    onClick={() => analyze.mutate(s.id)}
                    disabled={analyze.isPending}
                    className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
                  >
                    Retry analyze
                  </button>
                )}
                {(s.status === "analyzed" || s.status === "ready") && (
                  <Link
                    href={`/songs/${s.id}/debug`}
                    className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
                  >
                    Debug
                  </Link>
                )}
                {(SEPARATABLE.includes(s.status) ||
                  s.status === "separating") && (
                  <button
                    type="button"
                    onClick={() => separate.mutate(s.id)}
                    disabled={isMidSeparation || separate.isPending}
                    className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
                  >
                    {isMidSeparation
                      ? "Separating…"
                      : hasStems
                        ? "Re-separate stems"
                        : "Separate stems"}
                  </button>
                )}
                {(transcribable || s.status === "transcribing") && (
                  <button
                    type="button"
                    onClick={() => transcribe.mutate(s.id)}
                    disabled={isMidTranscription || transcribe.isPending}
                    className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
                  >
                    {isMidTranscription
                      ? "Transcribing…"
                      : hasTranscription
                        ? "Re-transcribe"
                        : "Transcribe"}
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
