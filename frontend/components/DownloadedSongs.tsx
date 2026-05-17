"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Song } from "@/lib/api";

function isTerminal(status: Song["status"]): boolean {
  return status === "analyzed" || status === "ready" || status === "failed";
}

export function DownloadedSongs() {
  const qc = useQueryClient();
  const songs = useQuery({
    queryKey: ["songs"],
    queryFn: api.listSongs,
    refetchInterval: (q) => {
      const data = q.state.data as Song[] | undefined;
      if (!data) return 1000;
      return data.some((s) => !isTerminal(s.status)) ? 1000 : false;
    },
  });

  const analyze = useMutation({
    mutationFn: (id: string) => api.triggerAnalyze(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["songs"] }),
  });

  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-semibold">Library</h2>
      {songs.isLoading && <p className="text-sm opacity-70">Loading…</p>}
      {songs.data?.length === 0 && (
        <p className="text-sm opacity-70">No songs yet. Search and add one.</p>
      )}
      <ul className="flex flex-col gap-3">
        {songs.data?.map((s) => (
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
                className={
                  "text-xs px-2 py-1 rounded " +
                  (s.status === "failed"
                    ? "bg-red-500/20"
                    : s.status === "analyzed" || s.status === "ready"
                      ? "bg-green-500/20"
                      : s.status === "downloaded"
                        ? "bg-blue-500/20"
                        : "bg-yellow-500/20")
                }
              >
                {s.status}
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
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
