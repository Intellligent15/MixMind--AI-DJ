"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type SearchResult } from "@/lib/api";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function SearchPanel() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");

  const results = useQuery({
    queryKey: ["search", submitted],
    queryFn: () => api.search(submitted, 10),
    enabled: submitted.length > 0,
  });

  const qc = useQueryClient();
  const add = useMutation({
    mutationFn: (r: SearchResult) => api.createSong(r),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["songs"] }),
  });

  return (
    <section className="flex flex-col gap-4">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(query.trim());
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search YouTube..."
          className="flex-1 border rounded px-3 py-2 bg-transparent"
        />
        <button
          type="submit"
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Search
        </button>
      </form>

      {results.isFetching && <p className="text-sm opacity-70">Searching…</p>}
      {results.error && (
        <p className="text-sm text-red-600">
          {(results.error as Error).message}
        </p>
      )}

      <ul className="flex flex-col gap-2">
        {results.data?.map((r) => (
          <li
            key={r.youtube_video_id}
            className="flex items-center gap-3 border rounded p-2"
          >
            {r.thumbnail_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={r.thumbnail_url}
                alt=""
                className="w-20 h-12 object-cover rounded"
              />
            )}
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{r.title}</p>
              <p className="text-xs opacity-70 truncate">
                {r.artist ?? "—"} · {formatDuration(r.duration_seconds)}
              </p>
            </div>
            <button
              type="button"
              onClick={() => add.mutate(r)}
              disabled={add.isPending}
              className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
            >
              Add
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
