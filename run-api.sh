#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [[ -f ".env.local" ]]; then
  set -a
  source ".env.local"
  set +a
fi

BACKEND_URL="${FLASK_API_PATH%/}"
FRONTEND_URL="${OAUTH_JS_ORIGIN_LOCAL%/}"
FRONTEND_PORT="${FRONTEND_PORT:-${FRONTEND_URL##*:}}"

export FLASK_API_PATH="$BACKEND_URL"
export OAUTH_REDIRECT_URI="$OAUTH_REDIRECT_URI_LOCAL"
export API_ENDPOINT="$BACKEND_URL"
export OAUTH_JS_ORIGIN="$FRONTEND_URL"

cleanup() {
  local exit_code=$?
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Using GCP API: $BACKEND_URL"

php -S "127.0.0.1:${FRONTEND_PORT}" -t "$ROOT_DIR/frontend/public" &
FRONTEND_PID=$!

wait "$FRONTEND_PID"
