# AI DJ

Personal-use web app that turns a queue of YouTube songs into a continuously mixed DJ set, with transitions designed per-pair by an LLM using stems, beats, key, and timestamped lyrics.

See [ai-dj-spec.md](ai-dj-spec.md) for the full project specification.

## Status

**Phase 1 complete** — infrastructure skeleton is up. See [docs/the notes](docs/the notes) for what was built and what deviated from the spec.

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
brew install uv fnm docker
fnm install 20
```

Then from the repo root:

```bash
cp .env.example .env          # stub keys are fine for Phase 1
./start-dev.sh                # brings up docker stack + native worker (foreground)
```

Verify everything is talking:

```bash
curl localhost:8000/health    # -> {"status":"ok","db":"ok","redis":"ok"}
open http://localhost:3000    # renders the health payload
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
