"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { api, STEM_NAMES, type StemName, type Stems } from "@/lib/api";

const STEM_COLORS: Record<StemName, { wave: string; progress: string }> = {
  vocals: { wave: "#fda4af", progress: "#e11d48" },
  drums: { wave: "#fcd34d", progress: "#d97706" },
  bass: { wave: "#86efac", progress: "#16a34a" },
  other: { wave: "#93c5fd", progress: "#2563eb" },
};

type StemRecord<T> = Record<StemName, T>;
const emptyStemRecord = <T,>(v: T): StemRecord<T> => ({
  vocals: v,
  drums: v,
  bass: v,
  other: v,
});

export function StemsDebug({ songId, stems }: { songId: string; stems: Stems }) {
  // DOM + visualization refs
  const containerRefs = useRef<StemRecord<HTMLDivElement | null>>(
    emptyStemRecord<HTMLDivElement | null>(null)
  );
  const wsRefs = useRef<StemRecord<WaveSurfer | null>>(
    emptyStemRecord<WaveSurfer | null>(null)
  );

  // Web Audio graph: one ctx + four gain nodes (persistent) + four source
  // nodes (recreated per play, since AudioBufferSourceNodes are one-shot).
  // All four stems share ctx.currentTime, so a single .start(t) on all of
  // them is sample-accurate — fixes the per-element clock drift we got
  // from four separate HTMLAudioElements.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef<StemRecord<AudioBuffer | null>>(
    emptyStemRecord<AudioBuffer | null>(null)
  );
  const gainsRef = useRef<StemRecord<GainNode | null>>(
    emptyStemRecord<GainNode | null>(null)
  );
  const sourcesRef = useRef<StemRecord<AudioBufferSourceNode | null>>(
    emptyStemRecord<AudioBufferSourceNode | null>(null)
  );
  // ctx.currentTime when the most recent .start() was scheduled, and the
  // song-relative offset at that moment. currentPos = offset + (now - start).
  const startCtxTimeRef = useRef(0);
  const offsetAtStartRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [solo, setSolo] = useState<StemName | null>(null);
  const [muted, setMuted] = useState<StemRecord<boolean>>(
    emptyStemRecord<boolean>(false)
  );

  // ---- one-time setup per (songId, stems.id): graph + buffers + WS viz ----
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    offsetAtStartRef.current = 0;

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    for (const name of STEM_NAMES) {
      const gain = ctx.createGain();
      gain.gain.value = 1;
      gain.connect(ctx.destination);
      gainsRef.current[name] = gain;
    }

    // WaveSurfers exist for the waveform + cursor only. We mute them and
    // never call .play() — sound comes exclusively from the Web Audio graph.
    // We create WS instances *without* a URL and feed them a Blob below,
    // so each stem is fetched exactly once instead of twice (once by WS,
    // once for decodeAudioData).
    const createdWs: WaveSurfer[] = [];
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
      });
      ws.setVolume(0);
      wsRefs.current[name] = ws;
      createdWs.push(ws);
    }

    // Fetch each stem ONCE, share the bytes between WaveSurfer (Blob)
    // and decodeAudioData (ArrayBuffer). decodeAudioData detaches the
    // buffer it's given, so we hand it a fresh slice and keep the
    // original for the Blob.
    Promise.all(
      STEM_NAMES.map(async (name) => {
        const resp = await fetch(api.stemAudioUrl(songId, name));
        if (!resp.ok) throw new Error(`stem ${name}: HTTP ${resp.status}`);
        const arr = await resp.arrayBuffer();
        const blob = new Blob([arr], { type: "audio/wav" });
        // decodeAudioData detaches its argument — pass a copy so the
        // Blob stays usable for WaveSurfer to render the waveform.
        const buf = await ctx.decodeAudioData(arr.slice(0));
        return [name, buf, blob] as const;
      })
    )
      .then((triples) => {
        if (cancelled) return;
        for (const [name, buf, blob] of triples) {
          buffersRef.current[name] = buf;
          // Feed the already-fetched bytes to WaveSurfer — no network.
          wsRefs.current[name]?.loadBlob(blob);
        }
        setIsLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("stem decode failed", err);
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      for (const name of STEM_NAMES) {
        const src = sourcesRef.current[name];
        if (src) {
          src.onended = null;
          try {
            src.stop();
          } catch {
            // already stopped — safe to ignore
          }
          src.disconnect();
        }
        sourcesRef.current[name] = null;
        gainsRef.current[name]?.disconnect();
        gainsRef.current[name] = null;
        buffersRef.current[name] = null;
        wsRefs.current[name] = null;
      }
      createdWs.forEach((ws) => ws.destroy());
      ctx.close().catch(() => {});
      audioCtxRef.current = null;
    };
  }, [songId, stems.id]);

  // ---- solo/mute → gain ----
  useEffect(() => {
    for (const name of STEM_NAMES) {
      const gain = gainsRef.current[name];
      if (!gain) continue;
      const audible = solo === null ? !muted[name] : solo === name;
      gain.gain.value = audible ? 1 : 0;
    }
  }, [solo, muted]);

  // ---- playback ----
  const tick = useCallback(() => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    const pos =
      offsetAtStartRef.current + (ctx.currentTime - startCtxTimeRef.current);
    for (const name of STEM_NAMES) {
      wsRefs.current[name]?.setTime(Math.max(0, pos));
    }
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const startSources = useCallback(async (fromOffset: number) => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    if (ctx.state === "suspended") await ctx.resume();

    // Tear down any leftover sources before scheduling new ones.
    for (const name of STEM_NAMES) {
      const old = sourcesRef.current[name];
      if (old) {
        old.onended = null;
        try {
          old.stop();
        } catch {
          // already stopped — safe to ignore
        }
        old.disconnect();
      }
      sourcesRef.current[name] = null;
    }

    const startAt = ctx.currentTime + 0.08; // small lookahead
    const newSources: AudioBufferSourceNode[] = [];
    for (const name of STEM_NAMES) {
      const buf = buffersRef.current[name];
      const gain = gainsRef.current[name];
      if (!buf || !gain) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      sourcesRef.current[name] = src;
      newSources.push(src);
    }

    // Single ended handler — only acts on natural end-of-song (we null the
    // handler before manual stop()).
    const onAnyEnded = () => {
      const c = audioCtxRef.current;
      if (!c) return;
      const pos = offsetAtStartRef.current + (c.currentTime - startCtxTimeRef.current);
      const dur = buffersRef.current.vocals?.duration ?? 0;
      if (pos + 0.05 >= dur) {
        offsetAtStartRef.current = 0;
        setIsPlaying(false);
        if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        for (const name of STEM_NAMES) wsRefs.current[name]?.setTime(0);
      }
    };
    newSources.forEach((s) => {
      s.onended = onAnyEnded;
    });

    for (const s of newSources) s.start(startAt, fromOffset);
    startCtxTimeRef.current = startAt;
    offsetAtStartRef.current = fromOffset;
  }, []);

  const play = useCallback(async () => {
    if (isLoading) return;
    await startSources(offsetAtStartRef.current);
    setIsPlaying(true);
    if (rafRef.current === null) rafRef.current = requestAnimationFrame(tick);
  }, [isLoading, startSources, tick]);

  const pause = useCallback(() => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    const pos =
      offsetAtStartRef.current + (ctx.currentTime - startCtxTimeRef.current);
    for (const name of STEM_NAMES) {
      const src = sourcesRef.current[name];
      if (src) {
        src.onended = null;
        try {
          src.stop();
        } catch {
          // already stopped — safe to ignore
        }
        src.disconnect();
      }
      sourcesRef.current[name] = null;
    }
    offsetAtStartRef.current = Math.max(0, pos);
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setIsPlaying(false);
  }, []);

  const togglePlay = () => {
    if (isPlaying) pause();
    else void play();
  };

  const restart = async () => {
    if (isPlaying) pause();
    offsetAtStartRef.current = 0;
    for (const name of STEM_NAMES) wsRefs.current[name]?.setTime(0);
    await play();
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
          disabled={isLoading}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
        >
          {isLoading ? "Loading…" : isPlaying ? "Pause all" : "Play all"}
        </button>
        <button
          type="button"
          onClick={restart}
          disabled={isLoading}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
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
            <li key={name} className="border rounded p-2 flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono uppercase tracking-wide">{name}</span>
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
