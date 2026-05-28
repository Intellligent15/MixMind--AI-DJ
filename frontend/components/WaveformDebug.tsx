"use client";

import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Analysis, type Song } from "@/lib/api";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

const SECTION_COLORS = [
  "rgba(56, 189, 248, 0.25)",
  "rgba(244, 114, 182, 0.25)",
  "rgba(132, 204, 22, 0.25)",
  "rgba(251, 191, 36, 0.25)",
  "rgba(167, 139, 250, 0.25)",
  "rgba(248, 113, 113, 0.25)",
  "rgba(45, 212, 191, 0.25)",
  "rgba(251, 146, 60, 0.25)",
  "rgba(96, 165, 250, 0.25)",
  "rgba(232, 121, 249, 0.25)",
];

function colorForLabel(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) | 0;
  return SECTION_COLORS[Math.abs(h) % SECTION_COLORS.length];
}

export function WaveformDebug({
  song,
  analysis,
  audioUrl,
}: {
  song: Song;
  analysis: Analysis;
  audioUrl: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);

  const safeRegionsQuery = useQuery({
    queryKey: ["vocal_safe_regions", song.id],
    queryFn: () => api.getVocalSafeRegions(song.id),
    retry: false,
  });

  useEffect(() => {
    if (!containerRef.current) return;
    
    const regions = RegionsPlugin.create();
    regionsPluginRef.current = regions;
    
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#94a3b8",
      progressColor: "#0ea5e9",
      cursorColor: "#0ea5e9",
      height: 96,
      barWidth: 1,
      barGap: 1,
      barHeight: 0.6,
      url: audioUrl,
      plugins: [regions],
    });
    wsRef.current = ws;
    
    return () => {
      ws.destroy();
      wsRef.current = null;
      regionsPluginRef.current = null;
    };
  }, [audioUrl]);
  
  useEffect(() => {
    const regionsPlugin = regionsPluginRef.current;
    const ws = wsRef.current;
    if (!regionsPlugin || !ws || !safeRegionsQuery.data) return;
    
    const renderRegions = () => {
      regionsPlugin.clearRegions();
      safeRegionsQuery.data.regions.forEach((r) => {
        regionsPlugin.addRegion({
          start: r.start,
          end: r.end,
          content: "Safe",
          color: "rgba(34, 197, 94, 0.2)",
          drag: false,
          resize: false,
        });
      });
    };

    if (ws.getDuration() > 0) {
      renderRegions();
    } else {
      ws.once("decode", renderRegions);
    }
    
    return () => {
      ws.un("decode", renderRegions);
    };
  }, [safeRegionsQuery.data]);

  // Draw beat/downbeat ticks and section bands as an absolutely-positioned
  // overlay that mirrors the waveform's width. We use the audio duration from
  // the analysis row (more reliable than waiting for WaveSurfer's 'ready').
  const duration = Math.max(
    song.duration_seconds,
    analysis.beat_grid.at(-1) ?? 0,
    analysis.sections.at(-1)?.end ?? 0
  );

  const downbeatSet = new Set(analysis.downbeats);

  return (
    <div className="flex flex-col gap-2">
      <div className="relative">
        <div ref={containerRef} />
        <div
          ref={overlayRef}
          className="absolute inset-0 pointer-events-none"
        >
          {analysis.sections.map((sec, i) => (
            <div
              key={`sec-${i}`}
              className="absolute top-0 bottom-0"
              style={{
                left: `${(sec.start / duration) * 100}%`,
                width: `${((sec.end - sec.start) / duration) * 100}%`,
                background: colorForLabel(sec.label),
              }}
              title={`${sec.label} (${sec.start.toFixed(1)}s–${sec.end.toFixed(1)}s)`}
            />
          ))}
          {analysis.beat_grid.map((t, i) => {
            const isDown = downbeatSet.has(t);
            return (
              <div
                key={`beat-${i}`}
                className="absolute top-0 bottom-0"
                style={{
                  left: `${(t / duration) * 100}%`,
                  width: isDown ? "2px" : "1px",
                  background: isDown
                    ? "rgba(15, 23, 42, 0.85)"
                    : "rgba(15, 23, 42, 0.35)",
                }}
              />
            );
          })}
        </div>
      </div>
      <div className="flex gap-2 text-xs">
        <button
          type="button"
          onClick={() => wsRef.current?.playPause()}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Play / pause
        </button>
        <button
          type="button"
          onClick={() => wsRef.current?.seekTo(0)}
          className="border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Restart
        </button>
      </div>
    </div>
  );
}
