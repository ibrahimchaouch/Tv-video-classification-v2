#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-6381}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DBDIR="${BASE_DIR}/.redis/${PORT}"
CONF="${DBDIR}/redis.conf"

mkdir -p "${DBDIR}"

cat > "${CONF}" <<EOF
port ${PORT}
dir ${DBDIR}
dbfilename dump.rdb
save ""
appendonly no
protected-mode no
EOF

if command -v redis-cli >/dev/null 2>&1 && redis-cli -p "${PORT}" ping >/dev/null 2>&1; then
  echo "[start_redis] Redis already running on port ${PORT}. FLUSHALL…"
  redis-cli -p "${PORT}" FLUSHALL
else
  echo "[start_redis] Starting redis-server on port ${PORT}…"
  if ! command -v redis-server >/dev/null 2>&1; then
    echo "[start_redis] ERROR: redis-server not found. Please install Redis." >&2
    exit 1
  fi
  redis-server "${CONF}" >/dev/null 2>&1 &
  sleep 0.5
  redis-cli -p "${PORT}" ping >/dev/null
  redis-cli -p "${PORT}" FLUSHALL
fi

echo "[start_redis] Redis ready on port ${PORT} (FLUSHALL done)."
