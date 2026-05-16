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

echo "==> Starting native Celery worker (Ctrl-C to stop)..."
echo "    Backend:  http://localhost:8000/health"
echo "    Frontend: http://localhost:3000"
echo "    Stop docker stack with: ./stop-dev.sh"
echo

cd backend
exec uv run celery -A app.workers worker --loglevel=info
