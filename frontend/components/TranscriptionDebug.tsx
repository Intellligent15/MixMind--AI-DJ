"use client";

import type { Transcription } from "@/lib/api";

function fmt(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(2);
  return `${m}:${s.padStart(5, "0")}`;
}

function statusBadgeClass(status: Transcription["status"]): string {
  switch (status) {
    case "success":
      return "bg-emerald-500/30";
    case "skipped_instrumental":
      return "bg-slate-500/30";
    case "error":
      return "bg-red-500/30";
    default:
      return "bg-yellow-500/20";
  }
}

export function TranscriptionDebug({
  transcription,
}: {
  transcription: Transcription;
}) {
  const isSkipped = transcription.status === "skipped_instrumental";
  const isError = transcription.status === "error";

  return (
    <section className="flex flex-col gap-3 border rounded p-4">
      <header className="flex items-center justify-between">
        <h2 className="font-semibold">Transcription</h2>
        <span
          className={
            "text-xs px-2 py-1 rounded " +
            statusBadgeClass(transcription.status)
          }
        >
          {transcription.status}
        </span>
      </header>

      <dl className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        <Stat label="Model" value={transcription.model_name} />
        <Stat label="Language" value={transcription.language ?? "—"} />
        <Stat
          label="Segments"
          value={transcription.segments.length.toString()}
        />
        <Stat
          label="Vocal RMS"
          value={
            transcription.vocal_rms_observed != null
              ? transcription.vocal_rms_observed.toFixed(4)
              : "—"
          }
        />
      </dl>

      {isSkipped && (
        <p className="text-xs opacity-70">
          Skipped because vocal RMS{" "}
          ({transcription.vocal_rms_observed?.toFixed(4) ?? "—"}) was below
          threshold ({transcription.vocal_rms_threshold?.toFixed(4) ?? "—"}).
          Typical for instrumentals and ambient tracks.
        </p>
      )}

      {isError && (
        <p className="text-xs text-red-700 dark:text-red-400">
          Whisper failed. Click Re-transcribe in the library to retry.
        </p>
      )}

      {transcription.segments.length > 0 && (
        <ul className="max-h-72 overflow-y-auto flex flex-col gap-1 text-xs">
          {transcription.segments.map((seg, i) => (
            <li
              key={i}
              className="border-b border-black/5 dark:border-white/10 pb-1 last:border-0 flex gap-3"
            >
              <span className="opacity-60 tabular-nums whitespace-nowrap">
                {fmt(seg.start)} → {fmt(seg.end)}
              </span>
              <span className="flex-1">{seg.text}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border rounded p-2 flex flex-col gap-1">
      <span className="opacity-70">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
