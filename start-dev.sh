#!/usr/bin/env bash
# AI DJ — bring up the full dev stack.
#
# - Docker: postgres, redis, backend, frontend
# - Native (host): Celery worker (needs MPS access on macOS)
#
# The worker runs in the foreground; Ctrl-C stops it. Run ./stop-dev.sh to tear
# down the Docker stack afterwards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# yt-dlp uses ffmpeg for the WAV extraction postprocessor on the host worker.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH. Install with: brew install ffmpeg" >&2
  exit 1
fi

echo "==> Starting Docker services (postgres, redis, backend, frontend)..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

echo "==> Waiting for postgres + redis to be healthy..."
for i in {1..30}; do
  pg_status=$(docker compose ps --format json postgres | grep -o '"Health":"[^"]*"' | head -1 || true)
  redis_status=$(docker compose ps --format json redis | grep -o '"Health":"[^"]*"' | head -1 || true)
  if [[ "$pg_status" == '"Health":"healthy"' && "$redis_status" == '"Health":"healthy"' ]]; then
    echo "    postgres + redis healthy."
    break
  fi
  sleep 1
done

cd backend

# Make the host worker share the same cache directory the dockerized backend
# mounts (./cache at the repo root). Without this, the worker's CWD changes
# to backend/ and a relative LOCAL_STORAGE_PATH would land in backend/cache/
# — invisible to the container that serves the audio.
export LOCAL_STORAGE_PATH="${LOCAL_STORAGE_PATH:-$REPO_ROOT/cache}"

echo "==> Applying database migrations..."
uv run alembic upgrade head

echo "==> Starting native Celery worker (Ctrl-C to stop)..."
echo "    Backend:  http://localhost:8000/health"
echo "    Frontend: http://localhost:3000"
echo "    Stop docker stack with: ./stop-dev.sh"
echo

exec uv run celery -A app.workers worker --loglevel=info
