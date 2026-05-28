"use client";

import type { Transcription } from "@/lib/api";

type AlignedWord = {
  word: string;
  start: number | null;
  end: number | null;
  confidence: number | null;
  source: "whisper_match" | "whisper_substitution" | "interpolated";
};

export type LyricsView = {
  text: string | null;
  fetch_status: string;
  aligned_words: AlignedWord[] | null;
  alignment_status:
    | "not_attempted"
    | "success"
    | "whisper_only"
    | "low_quality"
    | "error";
  alignment_quality: number | null;
};

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
  lyrics,
}: {
  transcription: Transcription;
  lyrics?: LyricsView | null;
}) {
  const isSkipped = transcription.status === "skipped_instrumental";
  const isError = transcription.status === "error";

  return (
    <section className="flex flex-col gap-3 border rounded p-4">
      <header className="flex items-center justify-between">
        <h2 className="font-semibold">Transcription & Alignment</h2>
        <div className="flex gap-2">
          {lyrics && lyrics.alignment_status && (
            <span
              className={
                "text-xs px-2 py-1 rounded " +
                (lyrics.alignment_status === "success"
                  ? "bg-emerald-500/30"
                  : lyrics.alignment_status === "low_quality"
                  ? "bg-yellow-500/20"
                  : "bg-slate-500/30")
              }
            >
              Align: {lyrics.alignment_status}
            </span>
          )}
          <span
            className={
              "text-xs px-2 py-1 rounded " +
              statusBadgeClass(transcription.status)
            }
          >
            Whisper: {transcription.status}
          </span>
        </div>
      </header>

      <dl className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
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
        <Stat
          label="Alignment Q"
          value={
            lyrics && lyrics.alignment_quality != null
              ? lyrics.alignment_quality.toFixed(2)
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

      {lyrics?.aligned_words && lyrics.aligned_words.length > 0 ? (
        <ul className="max-h-72 overflow-y-auto flex flex-col gap-2 text-xs">
          {(() => {
            type Line = { start: number | null; end: number | null; words: AlignedWord[] };
            const lines: Line[] = [];
            let currentLine: Line | null = null;

            lyrics.aligned_words.forEach((w: AlignedWord) => {
              if (!currentLine) {
                currentLine = { start: w.start, end: w.end, words: [w] };
                lines.push(currentLine);
              } else {
                const lastWord = currentLine.words[currentLine.words.length - 1];
                const gap = (w.start ?? 0) - (lastWord.end ?? 0);
                const longGap = gap > 1.0;
                const veryLong = currentLine.words.length >= 24;
                if (longGap || veryLong) {
                  currentLine = { start: w.start, end: w.end, words: [w] };
                  lines.push(currentLine);
                } else {
                  currentLine.words.push(w);
                  currentLine.end = w.end ?? currentLine.end;
                }
              }
            });

            return lines.map((line, i) => (
              <li
                key={i}
                className="border-b border-black/5 dark:border-white/10 pb-2 last:border-0 flex gap-3"
              >
                <span className="opacity-60 tabular-nums whitespace-nowrap w-24 flex-shrink-0">
                  {line.start != null ? fmt(line.start) : "--"} → {line.end != null ? fmt(line.end) : "--"}
                </span>
                <span className="flex-1 flex flex-wrap gap-1">
                  {line.words.map((w: AlignedWord, j: number) => {
                    // Phonetic substitutions land around 0.6 even when
                    // Whisper was confident; only flag genuinely
                    // unreliable words (non-phonetic subs / interpolated).
                    const isLowConf =
                      w.source === "interpolated" ||
                      (w.confidence != null && w.confidence < 0.35);
                    return (
                      <span 
                        key={j} 
                        title={`Confidence: ${w.confidence?.toFixed(2) ?? '--'} (${w.source})`}
                        className={`${isLowConf ? 'opacity-40 text-red-700 dark:text-red-400' : ''}`}
                      >
                        {w.word}
                      </span>
                    );
                  })}
                </span>
              </li>
            ));
          })()}
        </ul>
      ) : transcription.segments.length > 0 ? (
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
      ) : null}
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
