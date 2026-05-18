# AI DJ

Personal-use web app that turns a queue of YouTube songs into a continuously mixed DJ set, with transitions designed per-pair by an LLM using stems, beats, key, and timestamped lyrics.

See [ai-dj-spec.md](ai-dj-spec.md) for the full project specification.

## Status

**Phase 6 complete** — mlx-whisper `large-v3-turbo` transcription on the vocal stem via the native worker, with word-level timestamps and a `vocal_rms < 0.005` skip gate for instrumentals. Songs now advance `analyzed → transcribing → ready` and the processing view picks up `transcribing` + `transcribed` pipeline steps (derived from the presence of a Transcription row). See [docs/the notes](docs/the notes). Prior phases: [Phase 5](docs/the notes), [Phase 4](docs/the notes), [Phase 3](docs/the notes), [Phase 2](docs/the notes), [Phase 1](docs/the notes).

Build phases are listed in the spec under **Build Phase Order**.

## Toolchain

- **Python 3.11** managed via [`uv`](https://docs.astral.sh/uv/)
- **Node 20 LTS** managed via [`fnm`](https://github.com/Schniz/fnm)
- **Docker** + **Docker Compose** for Postgres, Redis, backend, frontend
- **Celery worker runs natively on macOS host** (needs MPS + `rubberband-cli`)

## Layout

```
backend/    FastAPI app, Celery workers, Alembic migrations
frontend/   Next.js 15 app (App Router)
cache/      Generated audio, stems, analyses, mixes (gitignored)
docs/       Phase completion notes and reference docs
```

## First run

Prereqs (one-time):

```bash
brew install uv fnm docker ffmpeg
fnm install 20
```

`ffmpeg` is used by the native worker for the yt-dlp WAV extraction step.

Then from the repo root:

```bash
cp .env.example .env          # stub keys are fine for Phase 1
./start-dev.sh                # brings up docker stack + native worker (foreground)
```

Verify everything is talking:

```bash
curl localhost:8000/health           # -> {"status":"ok","db":"ok","redis":"ok"}
open http://localhost:3000           # search + library UI
open http://localhost:3000/health    # backend health payload (Phase 1 smoke test)
```

Add a song from the browser, or via curl:

```bash
curl 'localhost:8000/api/search?q=daft+punk+one+more+time&limit=3'
curl -X POST localhost:8000/api/songs -H 'content-type: application/json' \
  -d '{"youtube_video_id":"jNQXAC9IVRw","title":"Me at the zoo",
       "artist":"jawed","duration_seconds":19,"thumbnail_url":""}'
```

Send a no-op task through the native worker:

```bash
cd backend
uv run python -c "from app.workers.ping import ping; print(ping.delay().get(timeout=5))"
# -> pong
```

Stop the stack:

```bash
# Ctrl-C the worker, then:
./stop-dev.sh
```
