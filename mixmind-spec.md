# MixMind — Project Specification

## Overview

This is a personal-use web application that turns a queue of songs into a continuous, professionally-mixed DJ set using generative AI to plan the transitions.

The user searches for songs (via YouTube), drags them into a queue in the order they want to hear them, and clicks "Done." The system then:

1. Downloads each song's audio
2. Analyzes each song to extract its musical structure — BPM, key, beat positions, song sections, energy curve, and timestamped lyrics
3. Separates each song into four isolated stems (vocals, drums, bass, other instruments) using a deep learning model
4. For each pair of adjacent songs in the queue, asks a large language model to design a transition between them, choosing from techniques like beat-matched crossfades, filter sweeps, stem swaps, echo tails, loop-and-build sections, and harmonic layering
5. Renders the planned transitions to audio files
6. Plays the resulting continuous mix back to the user

The point of this project is to go beyond a simple crossfade. By giving the AI access to isolated stems, timestamped lyrics, and full musical analysis, it can plan transitions that swap the vocals of one song over the drums of the next, time a cut to land on a specific lyric, or build tension through filter sweeps before a key drop — the kinds of decisions a skilled human DJ makes.

This is a personal project, not a public product. It uses yt-dlp to source audio from YouTube, which is fine for personal use but would be untenable for distribution. The project is being built locally on an M4 MacBook with the intent to eventually self-host on a home machine, and possibly a cloud VM further down the line — but is designed for local-first operation.

---

## Environment

- **Target platform:** Local development on M4 MacBook, designed to be portable to self-hosted Linux home server
- **OS support:** macOS (primary), Linux (secondary)
- **Python:** 3.11, managed via **uv** (handles version + venv + dependencies)
- **Node:** 20 LTS, managed via **fnm** (single binary, fast shell hook)
- **Containerized:** Docker + Docker Compose from day one
- **Single-user, no auth**

---

## Repository Structure

Monorepo:

```
mixmind/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py              # FastAPI entry
│   │   ├── api/                 # Route handlers
│   │   ├── core/                # Config, logging, db setup
│   │   ├── models/              # SQLAlchemy models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── services/
│   │   │   ├── youtube/         # yt-dlp wrapper
│   │   │   ├── analysis/        # librosa + Spotify
│   │   │   ├── stems/           # Demucs
│   │   │   ├── transcription/   # Whisper (mlx-whisper)
│   │   │   ├── lyrics/          # Genius
│   │   │   ├── llm/             # Provider abstraction
│   │   │   ├── mixer/           # Mix plan execution / rendering
│   │   │   └── storage/         # Filesystem (modular)
│   │   ├── workers/             # Celery tasks
│   │   └── tests/
│   └── alembic/                 # DB migrations
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── app/                     # Next.js app router
│   ├── components/
│   ├── lib/
│   └── __tests__/
└── cache/                       # gitignored, mounted into containers
    ├── audio/                   # Original WAVs
    ├── stems/                   # Separated stems
    ├── analysis/                # Cached JSON analysis
    ├── transcriptions/          # Whisper output JSON
    └── mixes/                   # Rendered FLAC transitions
```

---

## Hosting & Deployment

### Hybrid Docker setup (M4-specific)

Containers on macOS run inside a Linux VM and cannot access Metal/MPS. To get GPU acceleration on M4, the Celery worker that runs Demucs, Whisper, and rubberband runs **natively on the host**. Everything else runs in Docker.

**In Docker:**
- `postgres` (postgres:16-alpine)
- `redis` (redis:7-alpine)
- `backend` (FastAPI, port 8000)
- `frontend` (Next.js, port 3000)

**Native on host:**
- `worker` (Celery worker process, started via a shell script or process manager like `honcho`)

The worker connects to the containerized Redis and Postgres via `localhost:6379` and `localhost:5432`. Shared `cache/` directory is mounted into the Docker containers and accessed directly by the native worker.

When the project is eventually moved to a Linux home server or cloud VM, the worker can join the Docker compose stack normally since Linux containers can access CUDA directly.

---

## Backend Stack

- **FastAPI** (web framework)
- **PostgreSQL 16** (database)
- **SQLAlchemy 2.0** (ORM) + **Alembic** (migrations)
- **Pydantic v2** (schemas)
- **Celery** + **Redis 7** (job queue)
- **pytest** + **pytest-asyncio** (testing)

---

## Frontend Stack

- **Next.js 15** (App Router, React 19)
- **TypeScript 5**
- **Tailwind CSS 4**
- **shadcn/ui** (components)
- **dnd-kit** (drag-and-drop)
- **WaveSurfer.js 7** (waveform display)
- **TanStack Query** (server state, polling)
- **Zustand** (client state)
- **Vitest** + **React Testing Library** (unit/integration testing)
- **Playwright** (end-to-end testing)

---

## Audio Stack

- **yt-dlp** — download and search (best available audio quality)
- **librosa** + **Spotify Web API** — hybrid analysis (BPM, key, beat grid, energy curve from librosa; section detection from Spotify when available, librosa fallback otherwise)
- **Demucs (htdemucs_ft)** — 4-stem separation (vocals, drums, bass, other), MPS-accelerated with CPU fallback
- **mlx-whisper** with `large-v3` model — timestamped vocal transcription, runs on isolated vocal stem
- **Genius API** — lyrics text (used as ground-truth reference; Whisper provides timestamps via the Lyrics Alignment Model below)
- **pyrubberband** — time-stretching and pitch-shifting (requires `rubberband-cli` installed natively)
- **soundfile** + **numpy** — audio I/O and manipulation
- **pydub** — convenience layer
- **FLAC** — output format for all rendered audio

### Whisper behavior

Whisper runs in parallel with stem separation during the analysis phase. It runs on the isolated vocal stem (after Demucs completes) for better accuracy. If the vocal stem has less than 5% non-silent content (deterministic energy threshold check — implemented as `vocal_rms < 0.005` against the RMS computed during separation), Whisper is skipped — this catches instrumentals, ambient tracks, and EDM where the "vocal" stem is mostly sample chops.

Per-decode signals — per-word `probability` and per-segment `avg_logprob` / `no_speech_prob` / `compression_ratio` / `temperature` — are preserved on the `Transcription` row alongside the text. They are inputs to the Vocal Safety Model below.

---

## Vocal Safety Model

Whisper output is treated as a hint, not as truth. For decisions about where a transition can land (hard cuts, stem swaps, drop swaps), we cross-reference Whisper's word boundaries with the **vocal stem's energy envelope** — a frame-wise RMS + peak signal computed during Demucs separation at ~10 Hz.

### Why two signals

- Whisper alone has **false negatives** (misses quiet vocals) — a hard cut placed in a "Whisper found nothing" gap might still chop a syllable that the model dropped.
- Whisper alone has **false positives** (hallucinates text on silence, e.g. the canonical "Thank you" loop) — we'd skip a perfectly safe transition zone because "Whisper said there's a word."
- The vocal stem envelope is the **physical** ground truth of vocal presence. Whisper is the **semantic** layer.

Triangulating both signals dramatically reduces the failure modes a listener would notice.

### Per-word usability test

A Whisper word is considered "real" (worth respecting in transition planning) when:

```
usable_vocal_word =
    word.probability             >= WORD_PROB_MIN          (default 0.35)
    AND segment.avg_logprob      >= SEGMENT_LOGPROB_MIN    (default -1.2)
    AND envelope.rms_near_word   >= STEM_RMS_PRESENCE      (default 0.02)
    AND envelope.peak_near_word  >= STEM_PEAK_PRESENCE     (default 0.08)
```

"Near word" = a small window around `(word.start, word.end)` (e.g. ±50 ms) over the envelope.

### Vocal-safe regions

A time interval `(t_start, t_end)` is **vocal-safe** (a transition can safely place a hard cut, swap, or drop within it) when:

```
no usable_vocal_word overlaps the interval
AND envelope.rms is below STEM_RMS_QUIET          (default 0.01) for all frames
AND envelope.peak is below STEM_PEAK_QUIET        (default 0.04) for all frames
AND (t_end - t_start) >= MIN_SAFE_REGION_SECONDS  (default 1.5)
```

The minimum-length constraint prevents picking sub-second gaps between words as "safe" — a transition needs room to breathe.

### Where this lives in the code

- **Persisted at separation time:** the vocal envelope (RMS + peak at 10 Hz) is computed in `separate_stems` and stored as a sidecar `cache/stems/<video_id>/vocal_envelope.json`. `Stems.vocal_envelope_path` points at it. Sidecar (not a JSONB column) because the rest of `cache/stems/` is the stems' canonical home, ~20 KB per song doesn't need SQL access patterns, and DB rows stay cheap. **Landed alongside Phase 5 (post-Phase 6).** Rows pre-`db729e2f9c53` are null — re-separate to populate.
- **Persisted at transcription time:** the per-word `probability` and per-segment confidence fields are kept on the `Transcription` row's `segments` JSONB.
- **Computed on demand:** `services/vocal_safety/` exposes a pure function `vocal_safe_regions(transcription, envelope, **thresholds) -> list[(start, end)]`. Thresholds are kwargs so callers can tune for the use case (e.g. the LLM mix-planner might want stricter regions than the manual mixer service).
- **API surface:** `GET /api/songs/{id}/vocal_safe_regions` (default thresholds, overridable via query params). Useful for the debug page's "show safe-cut zones on the waveform" overlay.
- **LLM hook (Phase 9):** the per-song JSON the LLM sees includes the safe-regions list at the planner's chosen thresholds. The LLM is told "hard cuts, stem swaps, and drop swaps should land inside these intervals; outside them, prefer crossfades."

### Status (when this lands)

- Per-word + per-segment confidence preservation: **landed in Phase 6** (May 2026).
- Vocal envelope computation during separation: **planned**, lands before Phase 7 so every newly-separated song captures it.
- `vocal_safe_regions` service + API + debug overlay: **planned**, lands before Phase 9 when the LLM starts consuming it.

---

## Lyrics Alignment Model

Whisper and Genius give us **complementary signals**:

- **Genius API** — authoritative lyric text. No timing.
- **mlx-whisper** — per-word timestamps. Possibly wrong words (hallucinations, dropouts, different punctuation, alternative cuts).

The goal is to assign Whisper's timestamps to Genius's exact lyric text, producing a `(word, start, end, confidence)` sequence the mix-planner can trust both *what* and *when*.

### Why this matters

The mix-planner LLM (Phase 9) plans transitions with prompts like "cut on the word `love`" or "fade out before `Hotter than hell`." For that to work:

- The textual reference (`love`, `Hotter than hell`) must match what's actually in the lyric — Whisper transcripts alone are unreliable here (we've seen "Thank you" hallucinations and word-drop regressions like "in front of" → "one of").
- The timing must match what's actually in the audio — Genius alone has no timing.

Aligning Genius (truth) onto Whisper (timing) closes the gap.

### Approaches

Three layered approaches; the first is the baseline.

#### 1. Sequence alignment (DTW / Needleman-Wunsch over words)

Pure-Python global alignment between two word sequences (Genius words, Whisper words):

- **Match** (same word at the same place): copy Whisper's timestamp onto the Genius word. Confidence = high.
- **Substitution** (similar but wrong, e.g. "flame" ↔ "name"): copy Whisper's timestamp anyway — Whisper got the timing right, just the word wrong. Score substitutions by edit distance and/or Soundex/Metaphone codes to be phonetically robust. Confidence = medium.
- **Insertion in Genius** (Whisper missed a word): interpolate timestamp linearly from the neighbors. Confidence = low.
- **Insertion in Whisper** (hallucinated word not in Genius): drop it.

Output: every Genius word gets `(word, start, end, confidence, source)` where `source ∈ {whisper_match, whisper_substitution, interpolated}`.

#### 2. Whisper `initial_prompt` priming (free quality boost, complements #1)

mlx-whisper accepts an `initial_prompt` (up to ~200 tokens) that biases output toward expected text. Passing the Genius lyrics (or just the first verse / chorus, given the token limit) as the prompt measurably reduces hallucination on familiar lyrics and increases the match rate in step #1. Doesn't *enforce* the prompted text — it's a hint, not a constraint — so we still need step #1 for the actual alignment.

Cost: one line in `services/transcription/service.py` once Genius is integrated. We feed the prompt at transcribe time (not after), so Lyrics fetch must run *before* (or in parallel with re-transcribe of) the songs we want to align.

#### 3. Forced alignment via phoneme-level CTC (escape hatch)

WhisperX, Montreal Forced Aligner, or wav2vec2-based CTC aligners take `(audio, text)` and produce word timestamps directly for the exact provided text — they never trust Whisper's word choices. Highest quality.

Cost: new dependency tree (~3 GB of weights, likely wav2vec2-based), possibly no clean MLX port. Reserve for the case where DTW alignment turns out to drop too many words on real-world songs.

### Recommendation

Ship **(1) + (2) together** as the V1 alignment. Skip (3) unless DTW quality is unacceptable in practice. The LLM mix-planner needs "approximately when does each line happen," not karaoke-grade frame accuracy — DTW + prompt priming is good enough for this bar.

### Edge cases

- **Section headers**: Genius often interleaves `[Chorus]`, `[Verse 2]`, `[Outro]`, `[Pre-Chorus]` etc. Strip these before alignment.
- **Repeated choruses**: Genius lists a chorus once with "(×4)" or similar; the song sings it four times. Either expand before alignment, or use a windowed alignment that can match the same Genius line at multiple time positions.
- **Live cuts / remixes**: the artist may sing different words than Genius lists. After alignment, segments with a long run of low-confidence matches should be flagged so the mix-planner can fall back to vocal-safety-only logic.
- **No Genius match**: fall back to raw Whisper output, mark `Lyrics.alignment_status = "whisper_only"` so downstream consumers know the data is less trustworthy.
- **Trust order**: for *what* is sung, `Genius > Whisper > stem-audio`. For *when* something is sung, `stem-audio > Whisper > Genius`. The alignment reconciles both.

### Where this lives in the code

- **Persisted at transcription time**: per-word `probability`, per-segment `avg_logprob` (landed in Phase 6).
- **Persisted at Genius fetch time**: the raw `Lyrics.text` and `Lyrics.genius_id` (Phase 8.5 — see Build Phase Order below).
- **Persisted at alignment time**: `Lyrics.aligned_words` JSONB (see data model), plus an `alignment_status` enum and a scalar `alignment_quality` aggregate so consumers can quickly decide whether to trust the alignment or fall back.
- **Computed by**: `services/lyrics_alignment/` — pure function `align(transcription: Transcription, genius_text: str) -> AlignmentResult`. Same shape as the vocal-safety service: deterministic, no I/O, kwarg-tunable thresholds.
- **Whisper feedback loop**: once a song has Lyrics, a re-transcribe pass with `initial_prompt=genius_text[:200_tokens]` is dispatched. The resulting Transcription is what step (1) aligns against.
- **API**: `GET /api/songs/{id}/lyrics` returns `{text, aligned_words, alignment_status, alignment_quality}`.
- **LLM hook (Phase 9)**: the per-song JSON the LLM sees includes the aligned word list when `alignment_status = "success"`. When `alignment_status = "whisper_only"` or quality is below a threshold, the LLM gets only the raw Whisper transcript plus vocal-safe regions — and is instructed to plan more conservative transitions (crossfades over hard cuts).

### Status (when this lands)

- Genius fetch + raw Lyrics row: **planned in Phase 8.5** (see Build Phase Order).
- DTW alignment + prompt-priming re-transcribe: **planned in Phase 8.5**, alongside Genius.
- Forced-alignment escape hatch: **deferred**, only built if real-world alignment quality is unacceptable.

---

## LLM Stack

- **Active provider:** Gemini (Google AI Studio API)
- **Abstraction:** `LLMProvider` interface with implementations for Gemini, Claude, and OpenAI. Provider selection via environment variable.
- **Interaction style:** Tool-calling
- **Granularity:** One LLM call per transition (no cross-transition awareness)
- **Input format:** Structured JSON containing analysis data and lyrics for both songs in the transition
- **No feedback loop, no style controls, no regeneration in V1**

### LLM Provider Interface

```python
class LLMProvider(Protocol):
    async def plan_transition(
        self,
        from_song: SongAnalysis,
        to_song: SongAnalysis,
        tools: list[ToolDefinition],
    ) -> MixPlan: ...
```

---

## Storage Abstraction

Modular by design so future cloud storage migration is a config change.

```python
class StorageBackend(Protocol):
    async def write(self, key: str, data: bytes) -> str: ...
    async def read(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...
    async def delete(self, key: str) -> None: ...
    async def get_url(self, key: str) -> str: ...
```

V1 implementation: `LocalFilesystemStorage`. Future: `S3Storage` plugs in without other changes.

---

## Data Model

```
Song
  id: UUID
  youtube_video_id: str (unique)
  title: str
  artist: str | None
  duration_seconds: float
  thumbnail_url: str
  audio_path: str | None
  status: enum (pending, downloading, downloaded, analyzing,
                analyzed, separating, transcribing, ready, failed)
  created_at, updated_at, last_accessed_at  # last_accessed_at drives LRU

Analysis (1:1 with Song)
  song_id: UUID
  bpm: float
  key: str                       # e.g., "Fm"
  camelot_key: str               # e.g., "5A" — for harmonic mixing logic
  time_signature: int
  beat_grid: list[float]         # JSONB
  downbeats: list[float]         # JSONB
  sections: list[Section]        # JSONB — Spotify when available, librosa fallback
  energy_curve: list[float]      # JSONB, sampled at 1Hz
  vocal_segments: list[tuple]    # JSONB — (start, end) pairs
  spotify_analysis_used: bool

Stems (1:1 with Song)
  song_id: UUID
  model_name: str                # e.g. "htdemucs"
  status: enum (pending, separating, separated, failed)
  vocals_path: str
  drums_path: str
  bass_path: str
  other_path: str
  vocal_rms: float | None        # scalar RMS over the full vocal stem (gates Whisper skip)
  vocal_envelope_path: str | None  # Sidecar JSON path for frame-wise RMS+peak at 10 Hz
                                   # over the vocal stem. Written by separate_stems.
                                   # Schema: {"frame_hz": 10, "rms": [...], "peak": [...]}.
                                   # Null on rows pre-`db729e2f9c53` — re-separate to fill.
                                   # Input to the Vocal Safety Model.

Transcription (1:1 with Song, nullable)
  song_id: UUID
  model_name: str                # e.g. "large-v3"
  status: enum (not_attempted, success, skipped_instrumental, error)
  language: str | None           # auto-detected (or forced) ISO code
  segments: list[Segment]        # JSONB — see Segment shape below
  vocal_rms_threshold: float | None  # threshold in effect at decision time
  vocal_rms_observed: float | None   # the Stems.vocal_rms at decision time
  duration_seconds: float | None

  # Segment shape (JSONB):
  #   start, end, text
  #   avg_logprob, no_speech_prob, compression_ratio, temperature    # confidence
  #   words: [{start, end, word, probability}]                       # confidence per word
  # Confidence fields are inputs to the Vocal Safety Model.

Lyrics (1:1 with Song, nullable)
  song_id: UUID
  genius_id: int | None
  text: str | None                       # raw Genius lyric text
  fetch_status: enum (not_attempted, success, not_found, error)
  # Lyrics Alignment Model — see dedicated section. Empty/null when
  # alignment hasn't run or Genius fetch failed.
  aligned_words: list[AlignedWord] | None  # JSONB; see shape below
  alignment_status: enum (not_attempted, success, whisper_only, low_quality, error)
  alignment_quality: float | None        # 0..1 aggregate; below ~0.5,
                                         # downstream code falls back to
                                         # vocal-safety-only logic.

  # AlignedWord shape (JSONB):
  #   word: str           # the Genius word (authoritative text)
  #   start, end: float   # timestamps from Whisper (or interpolated)
  #   confidence: float   # 0..1; combines match-type + Whisper word prob
  #   source: str         # "whisper_match" | "whisper_substitution" |
  #                       # "interpolated" | "whisper_only"

Queue
  id: UUID
  locked: bool                   # true once "Done" is clicked
  created_at

QueueItem
  queue_id: UUID
  song_id: UUID
  position: int

MixPlan (one per adjacent pair in a locked queue)
  id: UUID
  queue_id: UUID
  from_song_id: UUID
  to_song_id: UUID
  plan_json: JSONB               # tool-call sequence
  rendered_audio_path: str | None
  status: enum (pending, planning, rendering, ready, failed)
```

---

## Mix Plan Schema (LLM Tool Calls)

The LLM constructs a transition by calling these tools in sequence. The mixer service translates the tool-call sequence into actual audio operations using numpy / soundfile / pyrubberband.

- `set_transition_window(from_song_time_start, to_song_time_start, duration_bars)` — defines when the transition begins in each song and how long it lasts
- `crossfade_stem(stem, from_song, to_song, start_bar, duration_bars, curve)` — curve: `linear | exponential | s_curve`
- `apply_filter(stem, song, type, start_bar, duration_bars, freq_start, freq_end)` — type: `lowpass | highpass`
- `apply_echo(stem, song, start_bar, decay_bars, feedback)` — echo/delay tail-out
- `swap_stem(stem, from_song, to_song, at_bar)` — hard swap on a beat
- `loop_section(song, from_bar, to_bar, repeat_count)` — loop a bar for build-ups
- `pitch_shift(song, semitones)` — static pitch shift applied to the entire song. `semitones` may be fractional. Called once at transition start. Phase 7's hand-built generator does NOT emit this — pyrubberband artifacts at large shifts can be worse than the dissonance they fix, so the LLM (Phase 9) makes the per-pair call about whether to shift at all
- `temporary_pitch_shift(song, start_time, semitones, fade_in_bars, hold_bars, fade_out_bars)` — time-limited pitch shift that fades in, holds at the target, then fades back to the song's original key. Lets the planner introduce a brief key excursion (e.g. lift B up 3 semitones at the seam, hold for the chorus, then glide back to land on B's true key). Bar counts measured at the planner's target BPM
- `set_tempo_ramp(song, start_time, end_time, start_bpm, end_bpm)` — gradual tempo change for one song over a time window. Lets the planner avoid an instantaneous tempo lock at the seam (e.g. ramp B from its original BPM to A's BPM over the last 8 bars of B's intro). Outside the ramp window the song plays at the closer endpoint BPM
- `set_reasoning(text)` — LLM explains its choice, stored for debugging

### Transition Techniques Supported in V1

All of the following are implementable via the tool vocabulary above:

- Linear crossfade
- Filter sweep (low-pass on outgoing, high-pass on incoming)
- Echo/delay tail-out
- Vocal swap (swap vocals between songs while keeping backing tracks)
- Beat/drum swap (swap drum stems while keeping vocals)
- Drop swap (cut on a downbeat)
- Harmonic layering (sustain a note from song A over song B's intro)
- Loop and build (loop a bar of song A while song B builds in)

### Effects

- Gain envelopes per stem
- EQ filters (low-pass, high-pass)
- Echo/delay
- Pitch-shift (for key matching)

### Harmonic Matching

Pitch-shifting for key matching is **a Phase 9 LLM decision**, not a hardcoded behavior. The Phase 7 hand-built plan generator leaves keys as-is regardless of mismatch: at shifts > 2 semitones, pyrubberband's timbral artifacts often sound worse than the dissonance they're meant to fix, and at smaller shifts most listeners don't notice the mismatch.

Phase 9's LLM has three tools at its disposal for harmonic situations:

- **`pitch_shift(song, semitones)`** — static shift of the entire song. Best for adjacent-Camelot keys where a small (≤2 semitone) shift gives the LLM a clean lock without audible artifacts.
- **`temporary_pitch_shift(...)`** — shift for a region, then fade back to the original key. Useful for letting the chorus of B "ride" A's key briefly before B settles into its own.
- **`set_tempo_ramp(...)`** — gradual BPM change. Useful when paired with a key choice: ramping into a new tempo can mask the moment a brief pitch-shift fades out.

When a shift exceeds ±2 semitones, the executor logs a warning (`pitch_shift_warning=True` on the rendered transition) so the LLM's next-iteration loop can see that the planner asked for something likely to be artifact-heavy. The user can override per-transition via manual re-render with different LLM constraints (Phase 9+).

---

## Pipeline Flow

```
User searches → yt-dlp search → results displayed

User adds song to queue → song row created (status: pending),
                          low-priority download job enqueued

User clicks "Done" → queue locked → high-priority pipeline:

  For each song (parallel):
    1. Ensure download complete
    2. Run librosa analysis + Spotify lookup
    3. Demucs stem separation
    4. Whisper transcription on vocal stem (parallel with step 3 once stems exist)
    5. Genius lyrics fetch

  For first 3 transitions (sequential, prioritized):
    6. LLM call → mix plan
    7. Render transition to FLAC

  → Playback starts as soon as transition 1 is rendered

  For remaining transitions (background, while playback proceeds):
    Continue steps 6 and 7 for transitions 4, 5, 6...

If next transition isn't ready when needed: hard cut to next song.
```

---

## Frontend Flow

Single-page app with three states:

### 1. Building queue
- Search bar with yt-dlp-powered results
- Result cards with thumbnail, title, artist, duration
- Queue panel on the right
- Drag-and-drop reordering via dnd-kit
- Queue size capped at 20
- "Done" button enabled at 1+ songs

### 2. Processing
- Locked queue display
- Per-song progress: downloading → analyzing → separating → transcribing → ready
- Per-transition progress: planning → rendering → ready
- Overall pipeline status
- Polling via TanStack Query at 1-second intervals (no websockets in V1)

### 3. Playing
- Fullscreen player
- Current song info (title, artist, thumbnail)
- Upcoming song preview
- Waveform with playhead position via WaveSurfer.js
- Transition indicator (which technique is active, which stems are playing from which song)
- No queue editing during playback

---

## Docker Setup

`docker-compose.yml` services:

- `postgres` — postgres:16-alpine, port 5432, volume for data persistence
- `redis` — redis:7-alpine, port 6379
- `backend` — FastAPI app, port 8000, mounts `./cache` and `./backend/app`
- `frontend` — Next.js app, port 3000, mounts `./frontend`

Native (not in Docker on macOS):

- `worker` — Celery worker, started via `cd backend && celery -A app.workers worker --loglevel=info`. Has access to MPS via PyTorch and to natively-installed `rubberband-cli`.

A `start-dev.sh` script in the repo root brings up Docker services and starts the native worker in one command.

---

## Testing Strategy

### Backend
- **Unit tests:** All services tested in isolation with mocked dependencies. Target 90%+ coverage on `services/`, `api/`, `workers/`.
- **Integration tests:** Real Postgres (testcontainers), real Redis, mocked external APIs (yt-dlp, Spotify, Genius, Gemini).
- **Audio reference tests:** Bundle 3 short royalty-free reference tracks with known properties. Test that:
  - BPM detection returns expected value ±1 BPM
  - Stem separation produces 4 non-empty FLAC files
  - Whisper transcription includes expected words within ±100ms
  - Mix plans conform to schema
  - Rendered transitions are non-empty and the right duration

### Frontend
- **Unit tests:** Components tested with React Testing Library. Hooks tested in isolation.
- **End-to-end:** Playwright covers the critical flow: search → add 3 songs → click Done → wait for ready → verify playback starts.

### Out of scope
- Subjective audio quality is ear-tested, not automated.

---

## Environment Variables

```
DATABASE_URL=postgresql://aidj:aidj@localhost:5432/aidj
REDIS_URL=redis://localhost:6379/0
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
GENIUS_ACCESS_TOKEN=
GEMINI_API_KEY=
LLM_PROVIDER=gemini             # gemini | claude | openai
STORAGE_BACKEND=local           # local | s3 (future)
LOCAL_STORAGE_PATH=./cache
CACHE_MAX_SIZE_GB=50            # tunable
WHISPER_MODEL=large-v3
DEMUCS_MODEL=htdemucs_ft
DEMUCS_DEVICE=mps               # mps | cuda | cpu
LOG_LEVEL=INFO
```

---

## Out of Scope for V1

- User authentication
- Cloud deployment
- Mobile UI
- Manual transition editing
- Live queue reordering during playback
- Style controls (club / radio / experimental / etc.)
- AI regeneration / multi-shot transition options
- 6-stem separation
- WebSocket updates (polling only)
- S3 / cloud storage backend
- Whole-queue LLM planning (per-transition only)
- AI deciding when to invoke Whisper (Whisper always runs unless vocal stem is empty)
- YouTube Data API (yt-dlp search only)

---

## Build Phase Order

Recommended sequence — each phase ends with something testable:

1. **Infra skeleton** — Docker compose up, FastAPI hello world, Next.js hello world, native worker connects to Redis
2. **Search + download** — yt-dlp search endpoint, download endpoint, basic search UI, HTML5 playback of raw downloads
3. **Analysis pipeline** — librosa + Spotify hybrid, debug view with waveform + beat markers + BPM/key display
4. **Queue UI** — search results panel, queue panel with dnd-kit, Done button, basic back-to-back playback (no mixing yet)
5. **Stem separation** — Demucs integration on the native worker via MPS, debug view to solo each stem. **Vocal envelope sidecar** (RMS + peak at 10 Hz, `cache/stems/<video_id>/vocal_envelope.json` + `Stems.vocal_envelope_path`) is now also written during separation as a Phase 5/6 followup — input #2 to the Vocal Safety Model.
6. **Whisper transcription** — mlx-whisper integration, vocal stem energy check for skip logic, per-decode confidence signals persisted (per-word `probability`, per-segment `avg_logprob` / `no_speech_prob` / `compression_ratio` / `temperature`) for the Vocal Safety Model.
7. **Beat-matched crossfading** — first real mixing, BPM matching, beat alignment, simple crossfade. **Prerequisite:** Phase 5's vocal envelope must be in place for every song that hits the mixer (so Phase 9's vocal-safe-regions logic has its inputs available).
8. **Render-to-FLAC pipeline** — mix plan execution producing FLAC files
   - **8.5 (sub-phase) — Genius lyrics + alignment.** Fetch Genius lyric text into the `Lyrics` row; run the DTW-based Lyrics Alignment Model (see dedicated section) against the existing Whisper output to produce `aligned_words` + `alignment_status` + `alignment_quality`. Also adds an `initial_prompt`-primed re-transcribe pass over songs that have Genius text, so the alignment step has more matched anchors. Lands before Phase 9 so the LLM can consume aligned lyrics; can land in parallel with Phase 8.
9. **LLM mix planning** — Gemini provider, tool-calling, first AI-planned transitions. **Prerequisites:** the Vocal Safety service (`services/vocal_safety/`, `GET /api/songs/{id}/vocal_safe_regions`) AND the Lyrics Alignment service (`services/lyrics_alignment/`, aligned lyrics persisted on the `Lyrics` row) both land before this phase — the LLM consumes both when choosing transition windows and word-anchored cuts.
10. **Full pipeline integration** — the three-state frontend (building / processing / playing), playback orchestration, hard-cut fallback
11. **Polish** — error handling, LRU cache eviction, status display refinements, end-to-end Playwright tests

---

## Locked Decisions Summary

| Category | Decision |
|---|---|
| Deployment | Local-first, eventual self-hosted |
| Multi-user | Single-user, no auth |
| Backend | FastAPI + Python 3.11 |
| Database | PostgreSQL 16 |
| Job queue | Celery + Redis 7 |
| Frontend | Next.js 15 + TS + Tailwind + shadcn/ui |
| Drag-and-drop | dnd-kit |
| Waveform | WaveSurfer.js |
| Audio engine | Server-rendered FLAC |
| Audio format | FLAC |
| Stem separation | Demucs htdemucs_ft, 4 stems, MPS |
| Time-stretch | pyrubberband |
| Analysis | librosa + Spotify hybrid |
| Transcription | mlx-whisper large-v3 on vocal stem, skip if instrumental |
| Lyrics | Genius API |
| LLM provider | Gemini (active), modular for others |
| LLM granularity | Per-transition tool-calling |
| Search | yt-dlp |
| Audio quality | Best available |
| Cache | LRU, size TBD |
| Pre-processing | On queue lock ("Done" click) |
| Playback start | When first 3 transitions ready |
| Queue cap | 20 songs |
| Reordering | Not allowed during playback |
| Manual transition override | Not in V1 |
| Harmonic matching | Pitch-shift with ±2 semitone warning |
| Storage | Local filesystem, modular interface |
| Containerization | Docker for everything except Celery worker (native for MPS) |
| Testing | Full coverage on deterministic code, ear-test for audio quality |
