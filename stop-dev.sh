#!/usr/bin/env bash
# AI DJ — stop the dev stack. The native Celery worker is Ctrl-C'd separately.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Stopping Docker services..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
echo "    Done. (Postgres volume preserved. Use 'docker compose down -v' to wipe it.)"
