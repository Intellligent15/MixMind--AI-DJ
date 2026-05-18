"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, isStatusError, type Song, type Stems } from "@/lib/api";

function isTerminal(status: Song["status"]): boolean {
  return status === "analyzed" || status === "ready" || status === "failed";
}

const SEPARATABLE: ReadonlyArray<Song["status"]> = [
  "analyzed",
  "ready",
  "failed",
];

// A song's "displayed" status is derived: once the Stems row exists, show
// `separated` even though the worker bounces the Song row back to
// `analyzed`. Mirrors ProcessingView's pipeline-step derivation.
type DisplayStatus = Song["status"] | "separated";

function displayStatus(song: Song, hasStems: boolean): DisplayStatus {
  if (hasStems && (song.status === "analyzed" || song.status === "ready")) {
    return "separated";
  }
  return song.status;
}

function badgeClass(status: DisplayStatus): string {
  if (status === "failed") return "bg-red-500/20";
  if (status === "separated") return "bg-emerald-500/30";
  if (status === "analyzed" || status === "ready") return "bg-green-500/20";
  if (status === "downloaded") return "bg-blue-500/20";
  return "bg-yellow-500/20";
}

export function DownloadedSongs() {
  const qc = useQueryClient();
  // Tracks song ids the user just clicked "Separate" on — keeps polling
  // alive across the analyzed -> separating -> analyzed flicker until the
  // Stems row actually shows up. Cleared automatically once stems land.
  const [pendingStems, setPendingStems] = useState<Set<string>>(new Set());

  const songs = useQuery({
    queryKey: ["songs"],
    queryFn: api.listSongs,
    refetchInterval: (q) => {
      const data = q.state.data as Song[] | undefined;
      if (!data) return 1000;
      if (data.some((s) => !isTerminal(s.status))) return 1000;
      // Keep polling while we're still waiting on stems for any song,
      // since the worker bounces Song.status back to `analyzed` (terminal)
      // before the Stems row appears in a separate transaction.
      if (pendingStems.size > 0) return 1000;
      return false;
    },
  });

  // Per-song stems lookup. 404 = "no stems yet" → null. Polls while we're
  // expecting a stems row (just-clicked or song is mid-separation).
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
        if (pendingStems.has(song.id)) return 1500;
        if (song.status === "separating") return 1500;
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

  // Auto-clear pendingStems entries once their stems land.
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
          const display = displayStatus(s, hasStems);
          const isMidSeparation =
            s.status === "separating" || pendingStems.has(s.id);
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
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
