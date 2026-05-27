"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useQueries, useQuery } from "@tanstack/react-query";
import WaveSurfer from "wavesurfer.js";
import {
  api,
  isStatusError,
  type Queue,
  type Song,
  type SongStatus,
} from "@/lib/api";

const PLAYABLE_STATUSES: ReadonlyArray<SongStatus> = [
  "downloaded",
  "analyzing",
  "analyzed",
  "separating",
  "transcribing",
  "ready",
];

function isPlayable(s: Song): boolean {
  return PLAYABLE_STATUSES.includes(s.status);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function Player() {
  const queueQuery = useQuery<Queue | null>({
    queryKey: ["queue", "current"],
    queryFn: async () => {
      try {
        return await api.getCurrentQueue();
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
  });

  const items = queueQuery.data?.items ?? [];

  // Background polling of each queue song so a not-yet-downloaded next-up
  // becomes playable mid-set without a full queue refetch.
  const songQueries = useQueries({
    queries: items.map((item) => ({
      queryKey: ["song", item.song.id],
      queryFn: () => api.getSong(item.song.id),
      initialData: item.song,
      refetchInterval: (q: { state: { data?: Song } }) => {
        const s = q.state.data;
        if (!s) return 1000;
        if (s.status === "failed" || s.status === "ready") return false;
        return 1000;
      },
    })),
  });

  const songs: Song[] = useMemo(
    () => songQueries.map((q, i) => q.data ?? items[i].song),
    [songQueries, items]
  );

  const [currentIdx, setCurrentIdx] = useState(0);
  const [skipNotice, setSkipNotice] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);

  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const current = songs[currentIdx];
  const upcoming = songs[currentIdx + 1];

  const findNextPlayable = useCallback(
    (from: number): number | null => {
      for (let i = from; i < songs.length; i++) {
        if (isPlayable(songs[i])) return i;
      }
      return null;
    },
    [songs]
  );

  // Skip-forward fallback: if the current index isn't playable, advance to
  // the next playable index. Only applies once songs have loaded.
  useEffect(() => {
    if (!songs.length) return;
    const cur = songs[currentIdx];
    if (!cur) return;
    if (cur.status === "failed") {
      const next = findNextPlayable(currentIdx + 1);
      if (next != null && next !== currentIdx) {
        setSkipNotice(`Skipped: ${cur.title} (failed)`);
        setCurrentIdx(next);
      }
    }
  }, [songs, currentIdx, findNextPlayable]);

  const mixQuery = useQuery({
    queryKey: ["mix", queueQuery.data?.id],
    queryFn: () => queueQuery.data ? api.getQueueMix(queueQuery.data.id) : null,
    enabled: !!queueQuery.data?.id,
    refetchInterval: (q) => q.state.data?.status === "pending" || q.state.data?.status === "rendering" ? 1000 : false,
  });

  const mixData = mixQuery.data;

  const [playMode, setPlayMode] = useState<"mix" | "queue">("mix");
  const isMixReady = mixData?.status === "ready";
  const activeMode = isMixReady ? playMode : "queue";

  const handleRenderMix = async () => {
    if (!queueQuery.data) return;
    try {
      await api.stitchQueue(queueQuery.data.id);
      mixQuery.refetch();
    } catch (e) {
      console.error(e);
      alert("Failed to start mix render");
    }
  };

  const advance = useCallback(() => {
    const next = findNextPlayable(currentIdx + 1);
    if (next == null) {
      setIsPlaying(false);
      return;
    }
    if (next !== currentIdx + 1) {
      const skipped = songs
        .slice(currentIdx + 1, next)
        .map((s) => s.title)
        .join(", ");
      if (skipped) setSkipNotice(`Skipped: ${skipped} (not ready)`);
    } else {
      setSkipNotice(null);
    }
    setCurrentIdx(next);
  }, [currentIdx, findNextPlayable, songs]);

  const advanceRef = useRef(advance);
  useEffect(() => {
    advanceRef.current = advance;
  }, [advance]);

  const currentPlayable = current ? isPlayable(current) : false;

  const safeRegionsQuery = useQuery({
    queryKey: ["vocal_safe_regions", current?.id],
    queryFn: () => current ? api.getVocalSafeRegions(current.id) : null,
    enabled: !!current?.id && (!mixData || mixData.status !== "ready"),
    retry: false,
  });

  const regionsPluginRef = useRef<any>(null);

  // WaveSurfer setup
  useEffect(() => {
    if (!waveContainerRef.current) return;
    if (!mixData || mixData.status !== "ready") {
      if (!current || !currentPlayable) return;
    }

    setPosition(0);
    setDuration(0);

    const audioUrl = activeMode === "mix" && queueQuery.data
      ? api.queueMixAudioUrl(queueQuery.data.id)
      : api.audioUrl(current!.id);

    let isCancelled = false;
    let localWs: WaveSurfer | null = null;

    import("wavesurfer.js/dist/plugins/regions.esm.js").then(({ default: RegionsPlugin }) => {
      if (isCancelled || !waveContainerRef.current) return;
      const regions = RegionsPlugin.create();
      regionsPluginRef.current = regions;
      
      const ws = WaveSurfer.create({
        container: waveContainerRef.current,
        waveColor: "#94a3b8",
        progressColor: "#0ea5e9",
        cursorColor: "#0ea5e9",
        height: 72,
        barWidth: 1,
        barGap: 1,
        barHeight: 0.6,
        url: audioUrl,
        plugins: [regions],
      });
      localWs = ws;
      wsRef.current = ws;

      // Plot regions immediately if the API query had already resolved
      plotRegions();

      ws.on("ready", () => {
        setDuration(ws.getDuration());
        ws.play().then(
          () => setAutoplayBlocked(false),
          () => setAutoplayBlocked(true),
        );
      });
      ws.on("play", () => {
        setIsPlaying(true);
        setAutoplayBlocked(false);
      });
      ws.on("pause", () => setIsPlaying(false));
      ws.on("timeupdate", (t: number) => setPosition(t));
      ws.on("finish", () => {
        if (activeMode !== "mix") advanceRef.current();
      });
    });

    return () => {
      isCancelled = true;
      if (localWs) {
        localWs.destroy();
      } else if (wsRef.current) {
        wsRef.current.destroy();
      }
      wsRef.current = null;
      regionsPluginRef.current = null;
    };
  }, [currentIdx, current?.id, currentPlayable, activeMode, queueQuery.data?.id]);

  const plotRegions = useCallback(() => {
    const regionsPlugin = regionsPluginRef.current;
    if (!regionsPlugin || !safeRegionsQuery.data) return;
    
    regionsPlugin.clearRegions();
    safeRegionsQuery.data.regions.forEach((r: any) => {
      regionsPlugin.addRegion({
        start: r.start,
        end: r.end,
        content: "Safe",
        color: "rgba(34, 197, 94, 0.2)",
        drag: false,
        resize: false,
      });
    });
  }, [safeRegionsQuery.data]);

  useEffect(() => {
    plotRegions();
  }, [plotRegions]);

  const togglePlay = useCallback(() => {
    const ws = wsRef.current;
    if (!ws) return;
    if (ws.isPlaying()) ws.pause();
    else ws.play().catch(() => setAutoplayBlocked(true));
  }, []);

  if (queueQuery.isLoading) {
    return <p className="text-sm opacity-70">Loading…</p>;
  }
  if (!queueQuery.data || !queueQuery.data.locked || songs.length === 0) {
    return (
      <p className="text-sm opacity-70">
        No locked queue. Build and lock one on the{" "}
        <Link href="/" className="underline">
          home page
        </Link>
        .
      </p>
    );
  }
  
  

  if (!current && activeMode === "queue") {
    return (
      <p className="text-sm opacity-70">
        End of queue. Build a new one on the{" "}
        <Link href="/" className="underline">
          home page
        </Link>
        .
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {isMixReady && (
        <div className="flex bg-zinc-900 border rounded-lg p-1 self-start">
          <button
            onClick={() => setPlayMode("mix")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              playMode === "mix" ? "bg-blue-600 text-white" : "hover:bg-zinc-800 text-zinc-400"
            }`}
          >
            Continuous Mix
          </button>
          <button
            onClick={() => setPlayMode("queue")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              playMode === "queue" ? "bg-blue-600 text-white" : "hover:bg-zinc-800 text-zinc-400"
            }`}
          >
            Queue Mode
          </button>
        </div>
      )}
      
      {skipNotice && activeMode === "queue" && (
        <div className="border border-yellow-500/40 rounded p-2 text-sm bg-yellow-500/10">
          {skipNotice}
        </div>
      )}
      
      {!isMixReady && (
        <section className="border rounded p-4 flex gap-4 items-center bg-zinc-900">
          <div className="flex-1">
            <h3 className="font-semibold text-lg">Continuous DJ Mix</h3>
            <p className="text-sm opacity-70">Render all transitions into a single FLAC.</p>
            {mixData && (mixData.status === "pending" || mixData.status === "rendering") && (
              <p className="text-sm text-yellow-500 mt-2">Rendering mix...</p>
            )}
            {mixData && mixData.status === "failed" && (
              <p className="text-sm text-red-500 mt-2">Render failed: {mixData.error_text}</p>
            )}
          </div>
          <button
            onClick={handleRenderMix}
            disabled={mixData?.status === "pending" || mixData?.status === "rendering"}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium disabled:opacity-50"
          >
            Render Full Mix
          </button>
        </section>
      )}

      {activeMode === "mix" ? (
        <section className="border rounded p-4 flex flex-col gap-2">
          <p className="text-xs opacity-60">Now playing</p>
          <p className="font-semibold truncate text-lg">Full DJ Mix</p>
          <p className="text-sm opacity-70">{songs.length} tracks</p>
          <a
            href={api.queueMixAudioUrl(queueQuery.data.id)}
            download="mix.flac"
            className="text-blue-400 text-sm hover:underline mt-2 self-start"
          >
            Download FLAC
          </a>
        </section>
      ) : (
        <section className="border rounded p-4 flex gap-4">
          {current.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={current.thumbnail_url}
              alt=""
              className="w-32 h-20 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <p className="text-xs opacity-60">Now playing · {currentIdx + 1}/{songs.length}</p>
            <p className="font-semibold truncate text-lg">{current.title}</p>
            <p className="text-sm opacity-70 truncate">{current.artist ?? "—"}</p>
            <p className="text-xs opacity-60 mt-1">status: {current.status}</p>
          </div>
        </section>
      )}

      <div className="relative">
        <div ref={waveContainerRef} className="border rounded p-2" />
        {autoplayBlocked && !isPlaying && (
          <button
            type="button"
            onClick={togglePlay}
            className="absolute inset-0 flex items-center justify-center rounded bg-black/40 text-white text-sm font-medium"
          >
            Click to start playback
          </button>
        )}
      </div>

      <section className="flex items-center gap-4">
        <button
          type="button"
          onClick={togglePlay}
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        {activeMode === "queue" && (
          <button
            type="button"
            onClick={advance}
            className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Next
          </button>
        )}
        <span className="text-sm tabular-nums opacity-70">
          {formatTime(position)} / {formatTime(duration)}
        </span>
      </section>

      {upcoming && activeMode === "queue" && (
        <section className="border rounded p-3 flex items-center gap-3 opacity-80">
          <span className="text-xs opacity-60">Next up</span>
          {upcoming.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={upcoming.thumbnail_url}
              alt=""
              className="w-16 h-10 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0">
            <p className="font-medium truncate">{upcoming.title}</p>
            <p className="text-xs opacity-70 truncate">
              {upcoming.artist ?? "—"} · {upcoming.status}
            </p>
          </div>
        </section>
      )}

      <section className="border-t pt-4">
        <p className="text-xs opacity-60 mb-2">Queue</p>
        <ul className="flex flex-col gap-1 text-sm">
          {songs.map((s, i) => (
            <li
              key={s.id + ":" + i}
              className={
                "flex items-center gap-2 py-1 " +
                (i === currentIdx && activeMode === "queue" ? "font-semibold" : "opacity-70")
              }
            >
              <span className="w-6 tabular-nums">{i + 1}</span>
              <span className="flex-1 truncate">{s.title}</span>
              <span className="text-xs opacity-60">{s.status}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
