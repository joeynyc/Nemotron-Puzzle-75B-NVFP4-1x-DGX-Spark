#!/usr/bin/env bash
# Tiny OpenAI-compatible chat against the local Puzzle server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/common.sh"

PROMPT="${*:-Write a short haiku about unified memory on GB10.}"
PORT="${PORT:-8000}"
URL="${BASE_URL:-http://127.0.0.1:${PORT}}/v1/chat/completions"

export PROMPT SERVED_NAME
python3 - <<'PY' | curl -sS "${URL}" -H 'Content-Type: application/json' -d @- | python3 -m json.tool
import json, os
print(json.dumps({
    "model": os.environ["SERVED_NAME"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 256,
    "temperature": 0.4,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
