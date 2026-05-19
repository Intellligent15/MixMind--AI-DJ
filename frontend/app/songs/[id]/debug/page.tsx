"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { use, useEffect, useState } from "react";
import {
  api,
  isStatusError,
  type Song,
  type Stems,
  type Transcription,
} from "@/lib/api";
import {
  isActivelyProcessing,
  isFullyProcessed,
  maySoonHaveStems,
  maySoonHaveTranscription,
} from "@/lib/song-status";
import { StemsDebug } from "@/components/StemsDebug";
import { TranscriptionDebug } from "@/components/TranscriptionDebug";
import { WaveformDebug } from "@/components/WaveformDebug";

export default function SongDebugPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  const qc = useQueryClient();
  // True from click-Separate / click-Transcribe until the corresponding
  // row lands. Survives the worker-status flicker so polling never stops
  // before the row actually exists.
  const [awaitingStems, setAwaitingStems] = useState(false);
  const [awaitingTranscription, setAwaitingTranscription] = useState(false);

  const songQ = useQuery({
    queryKey: ["song", id],
    queryFn: () => api.getSong(id),
    // Poll until the song is fully processed (or terminally failed).
    // Previously we only polled during separating/transcribing, which
    // missed the queue-lock chain's transitions THROUGH `analyzed` —
    // a user landing on this page mid-chain would see a stale status.
    // hasStems/hasTranscription are captured from stemsQ/transcriptionQ
    // declared below; the closure resolves them at refetch time.
    refetchInterval: (q) => {
      const s = q.state.data;
      if (!s) return 1500;
      if (awaitingStems || awaitingTranscription) return 1500;
      if (isActivelyProcessing(s.status)) return 1500;
      if (!isFullyProcessed(s, !!stemsQ.data, !!transcriptionQ.data)) {
        return 3000;
      }
      return false;
    },
  });
  const analysisQ = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => api.getAnalysis(id),
    retry: false,
  });
  const stemsQ = useQuery({
    queryKey: ["stems", id],
    queryFn: async (): Promise<Stems | null> => {
      try {
        return await api.getStems(id);
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
    retry: false,
    refetchInterval: (q) => {
      if (q.state.data) return false;
      if (songQ.data?.status === "failed") return false;
      if (awaitingStems || songQ.data?.status === "separating") return 1500;
      // Slow poll while the song could plausibly grow stems via the
      // queue-lock chain (analyzed/ready). 3s is enough to feel snappy
      // without busy-looping on songs the user never separates.
      if (songQ.data && maySoonHaveStems(songQ.data)) return 3000;
      return false;
    },
  });
  const transcriptionQ = useQuery({
    queryKey: ["transcription", id],
    queryFn: async (): Promise<Transcription | null> => {
      try {
        return await api.getTranscription(id);
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
    retry: false,
    refetchInterval: (q) => {
      if (q.state.data) return false;
      if (songQ.data?.status === "failed") return false;
      if (awaitingTranscription || songQ.data?.status === "transcribing")
        return 1500;
      if (songQ.data && maySoonHaveTranscription(songQ.data)) return 3000;
      return false;
    },
  });

  // If the page mounts and we land mid-separation or mid-transcription,
  // kick the awaiting flag on automatically.
  useEffect(() => {
    if (songQ.data?.status === "separating") setAwaitingStems(true);
    if (songQ.data?.status === "transcribing") setAwaitingTranscription(true);
  }, [songQ.data?.status]);

  // Stop awaiting as soon as the corresponding row shows up.
  useEffect(() => {
    if (stemsQ.data) setAwaitingStems(false);
  }, [stemsQ.data]);
  useEffect(() => {
    if (transcriptionQ.data) setAwaitingTranscription(false);
  }, [transcriptionQ.data]);

  const separate = useMutation({
    mutationFn: () => api.triggerSeparate(id),
    onMutate: () => {
      // Optimistic flip + start awaiting so the polling loops engage even
      // before the worker has picked the task up.
      setAwaitingStems(true);
      qc.setQueryData<Song | undefined>(["song", id], (prev) =>
        prev ? { ...prev, status: "separating" } : prev
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["song", id] });
      qc.invalidateQueries({ queryKey: ["stems", id] });
    },
    onError: () => {
      setAwaitingStems(false);
      qc.invalidateQueries({ queryKey: ["song", id] });
    },
  });
  const transcribe = useMutation({
    mutationFn: () => api.triggerTranscribe(id),
    onMutate: () => {
      setAwaitingTranscription(true);
      qc.setQueryData<Song | undefined>(["song", id], (prev) =>
        prev ? { ...prev, status: "transcribing" } : prev
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["song", id] });
      qc.invalidateQueries({ queryKey: ["transcription", id] });
    },
    onError: () => {
      setAwaitingTranscription(false);
      qc.invalidateQueries({ queryKey: ["song", id] });
    },
  });
  const stemsAvailable = !!stemsQ.data;
  const stemsMissing404 = !stemsQ.data && stemsQ.isFetched && !stemsQ.error;
  const transcriptionAvailable = !!transcriptionQ.data;
  const transcriptionMissing404 =
    !transcriptionQ.data && transcriptionQ.isFetched && !transcriptionQ.error;

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
                {awaitingStems || songQ.data.status === "separating"
                  ? "Separating stems… (cold model load ~30s the first time, ~10–15s per song after)"
                  : stemsMissing404
                    ? "No stems yet. Run Demucs to separate vocals, drums, bass, and other."
                    : `Stems unavailable: ${(stemsQ.error as Error)?.message ?? "unknown error"}`}
              </p>
              <button
                type="button"
                onClick={() => separate.mutate()}
                disabled={
                  separate.isPending ||
                  awaitingStems ||
                  songQ.data.status === "separating"
                }
                className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
              >
                {separate.isPending ||
                awaitingStems ||
                songQ.data.status === "separating"
                  ? "Separating…"
                  : "Separate stems"}
              </button>
            </section>
          )}

          {transcriptionAvailable && (
            <TranscriptionDebug transcription={transcriptionQ.data!} />
          )}
          {!transcriptionAvailable && stemsAvailable && (
            <section className="flex items-center justify-between border rounded p-3">
              <p className="text-sm opacity-80">
                {awaitingTranscription || songQ.data.status === "transcribing"
                  ? "Transcribing vocals… (first run downloads ~3 GB of MLX weights; subsequent runs are ~real-time)"
                  : transcriptionMissing404
                    ? "No transcription yet. Run Whisper over the vocal stem."
                    : `Transcription unavailable: ${(transcriptionQ.error as Error)?.message ?? "unknown error"}`}
              </p>
              <button
                type="button"
                onClick={() => transcribe.mutate()}
                disabled={
                  transcribe.isPending ||
                  awaitingTranscription ||
                  songQ.data.status === "transcribing"
                }
                className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
              >
                {transcribe.isPending ||
                awaitingTranscription ||
                songQ.data.status === "transcribing"
                  ? "Transcribing…"
                  : "Transcribe"}
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
