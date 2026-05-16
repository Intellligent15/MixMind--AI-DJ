# AI DJ

Personal-use web app that turns a queue of YouTube songs into a continuously mixed DJ set, with transitions designed per-pair by an LLM using stems, beats, key, and timestamped lyrics.

See [ai-dj-spec.md](ai-dj-spec.md) for the full project specification.

## Status

Scaffold only. No code yet. Build phases are listed in the spec under **Build Phase Order**.

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
```

## Getting started

Phase 1 (infra skeleton) will fill in `docker-compose.yml`, the backend hello-world, and the frontend hello-world. Until then, this repo is structure only.
