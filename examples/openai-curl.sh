#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
MODEL="${SERVED_NAME:-nemotron-puzzle-75b-nvfp4}"

echo "=== models ==="
curl -sS "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool

echo
echo "=== chat ==="
curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: ok\"}],
    \"max_tokens\": 8,
    \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | python3 -m json.tool
