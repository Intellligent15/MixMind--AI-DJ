"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { use } from "react";
import { api, isStatusError } from "@/lib/api";
import { StemsDebug } from "@/components/StemsDebug";
import { WaveformDebug } from "@/components/WaveformDebug";

export default function SongDebugPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  const qc = useQueryClient();
  const songQ = useQuery({
    queryKey: ["song", id],
    queryFn: () => api.getSong(id),
    // Poll while separation is mid-flight so the stems panel appears as
    // soon as the worker finishes.
    refetchInterval: (q) =>
      q.state.data?.status === "separating" ? 1500 : false,
  });
  const analysisQ = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => api.getAnalysis(id),
    retry: false,
  });
  const stemsQ = useQuery({
    queryKey: ["stems", id],
    queryFn: () => api.getStems(id),
    retry: false,
    // Once a song flips out of separating, refetch stems once to pick the
    // new row up. Cheap because the query is otherwise idle.
    refetchInterval: (q) => {
      if (q.state.data) return false;
      return songQ.data?.status === "separating" ? 1500 : false;
    },
  });
  const separate = useMutation({
    mutationFn: () => api.triggerSeparate(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["song", id] });
    },
  });
  const stemsAvailable = !!stemsQ.data;
  const stemsMissing404 =
    !stemsQ.data && isStatusError(stemsQ.error, 404);

  return (
    <main className="min-h-screen max-w-5xl mx-auto p-8 flex flex-col gap-6 font-mono">
      <header className="flex items-baseline justify-between">
        <Link href="/" className="text-sm underline opacity-70">
          ← Back to library
        </Link>
        <p className="text-xs opacity-70">Debug view</p>
      </header>

      {songQ.isLoading && <p className="opacity-70">Loading song…</p>}
      {songQ.error && (
        <p className="text-red-600">{(songQ.error as Error).message}</p>
      )}

      {songQ.data && (
        <section className="flex items-center gap-3">
          {songQ.data.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={songQ.data.thumbnail_url}
              alt=""
              className="w-24 h-16 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0">
            <h1 className="font-bold text-lg truncate">{songQ.data.title}</h1>
            <p className="text-sm opacity-70 truncate">
              {songQ.data.artist ?? "—"}
            </p>
          </div>
          <span className="text-xs px-2 py-1 rounded bg-green-500/20">
            {songQ.data.status}
          </span>
        </section>
      )}

      {analysisQ.isLoading && <p className="opacity-70">Loading analysis…</p>}
      {analysisQ.error && (
        <p className="text-red-600">
          Analysis not available: {(analysisQ.error as Error).message}
        </p>
      )}

      {songQ.data && analysisQ.data && (
        <>
          <section className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
            <Stat label="BPM" value={analysisQ.data.bpm.toFixed(1)} />
            <Stat label="Key" value={analysisQ.data.key} />
            <Stat label="Camelot" value={analysisQ.data.camelot_key} />
            <Stat
              label="Time sig."
              value={`${analysisQ.data.time_signature}/4`}
            />
            <Stat
              label="Sections"
              value={analysisQ.data.sections.length.toString()}
            />
          </section>

          <WaveformDebug
            song={songQ.data}
            analysis={analysisQ.data}
            audioUrl={api.audioUrl(id)}
          />

          {stemsAvailable && (
            <StemsDebug songId={id} stems={stemsQ.data!} />
          )}
          {!stemsAvailable && (
            <section className="flex items-center justify-between border rounded p-3">
              <p className="text-sm opacity-80">
                {songQ.data.status === "separating"
                  ? "Separating stems… (this typically takes ~30s per minute of audio on MPS)"
                  : stemsMissing404
                    ? "No stems yet. Run Demucs to separate vocals, drums, bass, and other."
                    : `Stems unavailable: ${(stemsQ.error as Error)?.message ?? "unknown error"}`}
              </p>
              <button
                type="button"
                onClick={() => separate.mutate()}
                disabled={
                  separate.isPending || songQ.data.status === "separating"
                }
                className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
              >
                {separate.isPending || songQ.data.status === "separating"
                  ? "Separating…"
                  : "Separate stems"}
              </button>
            </section>
          )}

          <section className="flex flex-col gap-2">
            <h2 className="font-semibold">Sections</h2>
            <ul className="text-xs grid grid-cols-2 md:grid-cols-4 gap-2">
              {analysisQ.data.sections.map((sec, i) => (
                <li
                  key={i}
                  className="border rounded px-2 py-1 flex justify-between gap-2"
                >
                  <span>{sec.label}</span>
                  <span className="opacity-70">
                    {sec.start.toFixed(1)}–{sec.end.toFixed(1)}s
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border rounded p-3 flex flex-col gap-1">
      <span className="text-xs opacity-70">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
