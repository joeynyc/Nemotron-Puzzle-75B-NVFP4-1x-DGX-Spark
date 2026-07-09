#!/usr/bin/env bash
# Poll /v1/models + a tiny chat until the server is usable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/common.sh"

PORT="${PORT:-8000}"
URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
TIMEOUT="${READY_TIMEOUT_S:-900}"
start=$(date +%s)

echo "Waiting for ${URL} (timeout ${TIMEOUT}s)…"
while true; do
  now=$(date +%s)
  if (( now - start > TIMEOUT )); then
    echo "TIMEOUT after ${TIMEOUT}s"
    docker logs --tail 40 "${CONTAINER_NAME:-nemotron-puzzle-75b}" 2>/dev/null || true
    exit 1
  fi
  if curl -sf "${URL}/v1/models" >/dev/null 2>&1; then
    code=$(curl -s -o /tmp/puzzle-ready.json -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      "${URL}/v1/chat/completions" || true)
    if [[ "${code}" == "200" ]]; then
      echo "READY after $((now - start))s"
      exit 0
    fi
  fi
  sleep 5
  if (( (now - start) % 60 < 5 )); then
    echo "  still loading… $((now - start))s"
  fi
done
