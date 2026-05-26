#!/usr/bin/env bash
# Deploy ai-dj to the DigitalOcean droplet.
#
# Idempotent: rsyncs source, makes sure Docker + swap are in place,
# (re)builds the compose stack with the `prod` profile (which enables
# the worker container), runs alembic migrations, and prints the
# /health endpoint.
#
# Run from the repo root:    ./deploy/deploy_to_droplet.sh
#
# Requires:
#   - sshpass         (`brew install hudochenkov/sshpass/sshpass`)
#   - rsync           (system)
#   - The Modal app deployed once via
#       `cd backend && uv run modal deploy app/workers/modal_stubs.py`
#     so the worker container can call run_separation / run_transcription.

set -euo pipefail

DROPLET_IP="${DROPLET_IP:-137.184.211.233}"
DROPLET_USER="${DROPLET_USER:-root}"
DROPLET_PASSWORD="${DROPLET_PASSWORD:-joxbaZ-1sowfi-teskec}"
REMOTE_DIR="/root/ai-dj"
PUBLIC_API_BASE="http://${DROPLET_IP}:8000"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "ERROR: sshpass not installed. brew install hudochenkov/sshpass/sshpass" >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync not on PATH." >&2
  exit 1
fi

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)
SSH="sshpass -p $DROPLET_PASSWORD ssh ${SSH_OPTS[*]} ${DROPLET_USER}@${DROPLET_IP}"
RSYNC_SSH="sshpass -p ${DROPLET_PASSWORD} ssh ${SSH_OPTS[*]}"

run_remote() {
  # Pipes a heredoc into a single bash invocation on the droplet.
  sshpass -p "$DROPLET_PASSWORD" ssh "${SSH_OPTS[@]}" \
    "${DROPLET_USER}@${DROPLET_IP}" "bash -se" "$@"
}

echo "==> 1/6  Preparing droplet (docker, swap)..."
run_remote <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

# 1 GB RAM droplet cannot run `pip install librosa` + numba/scipy in
# a docker build without swap. 2 GB swapfile is cheap insurance.
if ! swapon --show | grep -q '/swapfile'; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

mkdir -p /root/ai-dj
REMOTE

echo "==> 1.5/6  Ensuring cookies.txt placeholder exists (so the docker bind-mount works)..."
run_remote <<REMOTE
set -euo pipefail
mkdir -p ${REMOTE_DIR}
test -e ${REMOTE_DIR}/cookies.txt || touch ${REMOTE_DIR}/cookies.txt
REMOTE

echo "==> 2/6  rsyncing source to ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}..."
rsync -az --delete \
  --exclude='.git/' \
  --exclude='backend/.venv/' \
  --exclude='frontend/node_modules/' \
  --exclude='frontend/.next/' \
  --exclude='cache/*/*' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='.DS_Store' \
  --exclude='cookies.txt' \
  -e "${RSYNC_SSH}" \
  ./ "${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/"

echo "==> 3/6  Building docker images on the droplet (this will take a few minutes)..."
run_remote <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}
# Make sure backend/.env is in place (rsync copied it; double-check).
test -s backend/.env || { echo 'backend/.env missing on droplet'; exit 1; }
export COMPOSE_PROFILES=prod
export NEXT_PUBLIC_API_BASE='${PUBLIC_API_BASE}'
docker compose build
REMOTE

echo "==> 4/6  Bringing the stack up (postgres, redis, backend, frontend, worker)..."
run_remote <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}
export COMPOSE_PROFILES=prod
export NEXT_PUBLIC_API_BASE='${PUBLIC_API_BASE}'
docker compose up -d
# Wait for postgres healthcheck.
for _ in \$(seq 1 60); do
  if docker compose ps --format json postgres | grep -q '"Health":"healthy"'; then
    break
  fi
  sleep 1
done
REMOTE

echo "==> 5/6  Running alembic migrations..."
run_remote <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}
docker compose exec -T backend alembic upgrade head
REMOTE

echo "==> 6/6  Verifying /health..."
run_remote <<'REMOTE'
set -euo pipefail
for _ in $(seq 1 30); do
  out=$(curl -s http://localhost:8000/health || true)
  if [[ "$out" == *'"status":"ok"'* ]]; then
    echo "$out"
    exit 0
  fi
  sleep 1
done
echo "health check did not return ok within 30s" >&2
echo "$out" >&2
docker compose -f /root/ai-dj/docker-compose.yml logs --tail=80
exit 1
REMOTE

cat <<EOF

==========================================================
✅ Deployed.
   Backend:  http://${DROPLET_IP}:8000/health
   Frontend: http://${DROPLET_IP}:3000
==========================================================
EOF
