"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
        if (s.status === "failed") return false;
        // Keep polling until we have at least downloaded (good enough for
        // playback) — finer-grained statuses can change too but won't gate us.
        if (isPlayable(s) && s.status !== "downloaded") return false;
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

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const current = songs[currentIdx];
  const upcoming = songs[currentIdx + 1];

  // Find the next playable index from a starting point. Skips over songs
  // whose status is failed or that haven't reached the downloaded gate.
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

  // Drive the audio element manually whenever the current song changes:
  // a reactive `src=` prop on <audio> is unreliable across browsers (and
  // doesn't always trigger a reload when WaveSurfer is wrapping the
  // element via the `media` option). Setting .src + calling .load() then
  // .play() guarantees the new track plays. WaveSurfer is recreated in
  // the same effect so its waveform always matches the loaded src.
  const currentId = current?.id;
  const currentPlayable = current ? isPlayable(current) : false;
  useEffect(() => {
    const audio = audioRef.current;
    const container = waveContainerRef.current;
    if (!audio || !container || !currentId || !currentPlayable) return;

    const newSrc = api.audioUrl(currentId);
    if (audio.src !== newSrc) {
      audio.src = newSrc;
      audio.load();
    }
    audio.play().catch(() => {
      // Browser autoplay policy may block us before the first user gesture;
      // the Play button will recover.
    });

    const ws = WaveSurfer.create({
      container,
      waveColor: "#94a3b8",
      progressColor: "#0ea5e9",
      cursorColor: "#0ea5e9",
      height: 72,
      barWidth: 1,
      barGap: 1,
      barHeight: 0.6,
      media: audio,
    });
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [currentId, currentPlayable]);

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

  const togglePlay = useCallback(() => {
    const a = audioRef.current;
    if (!a || !current || !isPlayable(current)) return;
    if (a.paused) a.play();
    else a.pause();
  }, [current]);

  if (queueQuery.isLoading) {
    return <p className="text-sm opacity-70">Loading…</p>;
  }
  if (!queueQuery.data || !queueQuery.data.locked || songs.length === 0) {
    return (
      <p className="text-sm opacity-70">
        No locked queue. Build and lock one on the{" "}
        <a href="/" className="underline">
          home page
        </a>
        .
      </p>
    );
  }
  if (!current) {
    return (
      <p className="text-sm opacity-70">
        End of queue. Build a new one on the{" "}
        <a href="/" className="underline">
          home page
        </a>
        .
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {skipNotice && (
        <div className="border border-yellow-500/40 rounded p-2 text-sm bg-yellow-500/10">
          {skipNotice}
        </div>
      )}

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

      <div ref={waveContainerRef} className="border rounded p-2" />

      <audio
        ref={audioRef}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
        onEnded={advance}
      />

      <section className="flex items-center gap-4">
        <button
          type="button"
          onClick={togglePlay}
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={advance}
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Next
        </button>
        <span className="text-sm tabular-nums opacity-70">
          {formatTime(position)} / {formatTime(duration)}
        </span>
      </section>

      {upcoming && (
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
                (i === currentIdx ? "font-semibold" : "opacity-70")
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
