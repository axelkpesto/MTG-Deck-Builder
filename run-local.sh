#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

BACKEND_URL="${FLASK_API_PATH_LOCAL%/}"
FRONTEND_URL="${OAUTH_JS_ORIGIN_LOCAL%/}"
BACKEND_PORT="${BACKEND_PORT:-${BACKEND_URL##*:}}"
FRONTEND_PORT="${FRONTEND_PORT:-${FRONTEND_URL##*:}}"

export DEFAULT_PORT="$BACKEND_PORT"
export FLASK_API_PATH="$BACKEND_URL"
export OAUTH_REDIRECT_URI="$OAUTH_REDIRECT_URI_LOCAL"
export API_ENDPOINT="$BACKEND_URL"
export OAUTH_JS_ORIGIN="$FRONTEND_URL"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/Scripts/python.exe}"

cleanup() {
  local exit_code=$?
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting vector DB server on $BACKEND_URL"
"$PYTHON_BIN" -m backend.api.vector_db_server &
BACKEND_PID=$!

php -S "127.0.0.1:${FRONTEND_PORT}" -t "$ROOT_DIR/frontend/public" &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"
