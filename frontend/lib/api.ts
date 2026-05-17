// Browser-side API base. Server Components use BACKEND_URL instead — that
// resolves inside the docker network, this one resolves from the user's
// browser.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type SongStatus =
  | "pending"
  | "downloading"
  | "downloaded"
  | "analyzing"
  | "analyzed"
  | "separating"
  | "transcribing"
  | "ready"
  | "failed";

export type SearchResult = {
  youtube_video_id: string;
  title: string;
  artist: string | null;
  duration_seconds: number;
  thumbnail_url: string | null;
};

export type Song = {
  id: string;
  youtube_video_id: string;
  title: string;
  artist: string | null;
  duration_seconds: number;
  thumbnail_url: string | null;
  audio_path: string | null;
  status: SongStatus;
  created_at: string;
  updated_at: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export type Section = {
  start: number;
  end: number;
  label: string;
};

export type Analysis = {
  id: string;
  song_id: string;
  bpm: number;
  key: string;
  camelot_key: string;
  time_signature: number;
  beat_grid: number[];
  downbeats: number[];
  sections: Section[];
  energy_curve: number[];
  vocal_segments: number[][];
  created_at: string;
  updated_at: string;
};

export const api = {
  search: (q: string, limit = 10) =>
    request<SearchResult[]>(
      `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`
    ),
  listSongs: () => request<Song[]>(`/api/songs`),
  createSong: (payload: Omit<SearchResult, never>) =>
    request<Song>(`/api/songs`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getSong: (id: string) => request<Song>(`/api/songs/${id}`),
  audioUrl: (id: string) => `${API_BASE}/api/songs/${id}/audio`,
  triggerAnalyze: (id: string) =>
    request<Song>(`/api/songs/${id}/analyze`, { method: "POST" }),
  getAnalysis: (id: string) => request<Analysis>(`/api/songs/${id}/analysis`),
};
