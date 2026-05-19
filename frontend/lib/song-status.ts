import type { Song, SongStatus } from "@/lib/api";

// Status values where a worker is actively making progress on the song.
// Pollers should refetch fast (1-2s) while any song matches.
const ACTIVELY_PROCESSING: ReadonlySet<SongStatus> = new Set<SongStatus>([
  "pending",
  "downloading",
  "analyzing",
  "separating",
  "transcribing",
]);

export function isActivelyProcessing(status: SongStatus): boolean {
  return ACTIVELY_PROCESSING.has(status);
}

/**
 * True iff a song has reached a state where no further automatic
 * progression is expected.
 *
 * `failed` is terminal (won't auto-recover) and `ready` + both rows
 * present is the happy-path end of the Phase-6 pipeline.
 *
 * Everything else — including `analyzed` (which the worker bounces
 * Song.status through twice during the queue-lock chain: post-analyze
 * pre-separate, and post-separate pre-transcribe) — is considered
 * "in flight," because a Celery chain task may still fire and advance
 * the song to the next state. Pollers should keep watching these songs
 * so the UI catches the next transition without a manual refresh.
 *
 * The cost of false positives here (a lonely `analyzed` song that
 * never actually moves) is a few extra background fetches — cheap.
 * The cost of false negatives (stopping polling too early) is a
 * visibly stale UI that needs a refresh — bad.
 */
export function isFullyProcessed(
  song: Song,
  hasStems: boolean,
  hasTranscription: boolean,
): boolean {
  if (song.status === "failed") return true;
  if (song.status === "ready" && hasStems && hasTranscription) return true;
  return false;
}

/**
 * True iff a song's current status is one where stems or transcription
 * rows might appear without the user clicking anything — used by
 * per-song polling so it doesn't stop watching during the analyzed
 * "pause" between chained worker tasks.
 */
export function maySoonHaveStems(song: Song): boolean {
  return (
    song.status === "separating" ||
    song.status === "analyzed" ||
    song.status === "ready"
  );
}

export function maySoonHaveTranscription(song: Song): boolean {
  return (
    song.status === "transcribing" ||
    song.status === "analyzed" ||
    song.status === "ready"
  );
}
