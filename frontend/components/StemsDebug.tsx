"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { api, STEM_NAMES, type StemName, type Stems } from "@/lib/api";

// Color per stem keeps the four waveforms visually distinguishable.
const STEM_COLORS: Record<StemName, { wave: string; progress: string }> = {
  vocals: { wave: "#fda4af", progress: "#e11d48" },
  drums: { wave: "#fcd34d", progress: "#d97706" },
  bass: { wave: "#86efac", progress: "#16a34a" },
  other: { wave: "#93c5fd", progress: "#2563eb" },
};

type SoloState = StemName | null;

export function StemsDebug({ songId, stems }: { songId: string; stems: Stems }) {
  // One container ref per stem, one WaveSurfer instance per stem.
  const containerRefs = useRef<Record<StemName, HTMLDivElement | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  const wsRefs = useRef<Record<StemName, WaveSurfer | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  const [isPlaying, setIsPlaying] = useState(false);
  const [solo, setSolo] = useState<SoloState>(null);
  const [muted, setMuted] = useState<Record<StemName, boolean>>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });

  // Build the four WaveSurfers once when the stems row id is known. Don't
  // include solo/muted in deps — those are reflected via the volume effect
  // below so we don't recreate (and re-stream) the WaveSurfers on every
  // toggle.
  useEffect(() => {
    const created: Partial<Record<StemName, WaveSurfer>> = {};
    for (const name of STEM_NAMES) {
      const el = containerRefs.current[name];
      if (!el) continue;
      const ws = WaveSurfer.create({
        container: el,
        waveColor: STEM_COLORS[name].wave,
        progressColor: STEM_COLORS[name].progress,
        cursorColor: "#0f172a",
        height: 56,
        barWidth: 1,
        barGap: 1,
        barHeight: 0.7,
        url: api.stemAudioUrl(songId, name),
      });
      created[name] = ws;
      wsRefs.current[name] = ws;
    }

    // Master state tracks whichever stem fires play/pause/finish first
    // (they're fired together by the master button anyway).
    const offs: Array<() => void> = [];
    for (const ws of Object.values(created)) {
      if (!ws) continue;
      offs.push(ws.on("play", () => setIsPlaying(true)));
      offs.push(ws.on("pause", () => setIsPlaying(false)));
      offs.push(ws.on("finish", () => setIsPlaying(false)));
    }

    // Capture the ref object once so the cleanup reads the same dict the
    // effect populated, even if React tears down before the next run.
    const refsAtSetup = wsRefs.current;
    return () => {
      offs.forEach((off) => off());
      for (const name of STEM_NAMES) {
        created[name]?.destroy();
        refsAtSetup[name] = null;
      }
    };
  }, [songId, stems.id]);

  // Apply solo + per-stem mute as volume: solo'd stem at 1, others at 0.
  // Solo wins over manual mute.
  useEffect(() => {
    for (const name of STEM_NAMES) {
      const ws = wsRefs.current[name];
      if (!ws) continue;
      const audible =
        solo === null ? !muted[name] : solo === name;
      ws.setVolume(audible ? 1 : 0);
    }
  }, [solo, muted]);

  const playAll = async () => {
    // Pause first to reset state, seek to 0, then play together. WaveSurfer's
    // play() returns a promise we don't need to await per-stem.
    for (const name of STEM_NAMES) {
      wsRefs.current[name]?.pause();
      wsRefs.current[name]?.setTime(0);
    }
    for (const name of STEM_NAMES) {
      void wsRefs.current[name]?.play();
    }
  };

  const togglePlay = () => {
    if (isPlaying) {
      for (const name of STEM_NAMES) wsRefs.current[name]?.pause();
    } else {
      for (const name of STEM_NAMES) void wsRefs.current[name]?.play();
    }
  };

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Stems</h2>
        <div className="text-xs opacity-70">
          {stems.model_name}
          {stems.vocal_rms !== null && (
            <> · vocal RMS {stems.vocal_rms.toFixed(3)}</>
          )}
        </div>
      </div>
      <div className="flex gap-2 text-xs">
        <button
          type="button"
          onClick={togglePlay}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
        >
          {isPlaying ? "Pause all" : "Play all"}
        </button>
        <button
          type="button"
          onClick={playAll}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Restart
        </button>
        {solo !== null && (
          <button
            type="button"
            onClick={() => setSolo(null)}
            className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Clear solo ({solo})
          </button>
        )}
      </div>
      <ul className="flex flex-col gap-2">
        {STEM_NAMES.map((name) => {
          const isSolo = solo === name;
          const isMuted = muted[name];
          return (
            <li
              key={name}
              className="border rounded p-2 flex flex-col gap-1"
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono uppercase tracking-wide">
                  {name}
                </span>
                <div className="flex gap-1">
                  <button
                    type="button"
                    onClick={() => setSolo(isSolo ? null : name)}
                    className={
                      "border rounded px-2 py-0.5 " +
                      (isSolo
                        ? "bg-amber-400/30 border-amber-500"
                        : "hover:bg-black/5 dark:hover:bg-white/10")
                    }
                  >
                    Solo
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setMuted((m) => ({ ...m, [name]: !m[name] }))
                    }
                    disabled={solo !== null}
                    className={
                      "border rounded px-2 py-0.5 disabled:opacity-40 " +
                      (isMuted && solo === null
                        ? "bg-slate-500/30 border-slate-600"
                        : "hover:bg-black/5 dark:hover:bg-white/10")
                    }
                  >
                    {isMuted ? "Muted" : "Mute"}
                  </button>
                </div>
              </div>
              <div
                ref={(el) => {
                  containerRefs.current[name] = el;
                }}
              />
            </li>
          );
        })}
      </ul>
    </section>
  );
}
