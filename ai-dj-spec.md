# AI DJ Mixing Web App — Project Specification

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
ai-dj/
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
- **Genius API** — lyrics text (used as ground-truth reference; Whisper provides timestamps)
- **pyrubberband** — time-stretching and pitch-shifting (requires `rubberband-cli` installed natively)
- **soundfile** + **numpy** — audio I/O and manipulation
- **pydub** — convenience layer
- **FLAC** — output format for all rendered audio

### Whisper behavior

Whisper runs in parallel with stem separation during the analysis phase. It runs on the isolated vocal stem (after Demucs completes) for better accuracy. If the vocal stem has less than 5% non-silent content (deterministic energy threshold check), Whisper is skipped — this catches instrumentals, ambient tracks, and EDM where the "vocal" stem is mostly sample chops. Timestamped output is cached as JSON, indexed by song.

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
  vocals_path: str
  drums_path: str
  bass_path: str
  other_path: str

Transcription (1:1 with Song, nullable)
  song_id: UUID
  segments: list[Segment]        # JSONB — {start, end, text, words: [{start, end, word}]}
  status: enum (not_attempted, success, skipped_instrumental, error)

Lyrics (1:1 with Song, nullable)
  song_id: UUID
  genius_id: int | None
  text: str | None
  fetch_status: enum (not_attempted, success, not_found, error)

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
- `pitch_shift(song, semitones)` — pitch-shift for key matching, called once at transition start
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

Pitch-shift the incoming song to match the outgoing song's key when keys are incompatible. If the required shift exceeds ±2 semitones, log a warning but proceed with the transition anyway.

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
5. **Stem separation** — Demucs integration on the native worker via MPS, debug view to solo each stem
6. **Whisper transcription** — mlx-whisper integration, vocal stem energy check for skip logic
7. **Beat-matched crossfading** — first real mixing, BPM matching, beat alignment, simple crossfade
8. **Render-to-FLAC pipeline** — mix plan execution producing FLAC files
9. **LLM mix planning** — Gemini provider, tool-calling, first AI-planned transitions
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
