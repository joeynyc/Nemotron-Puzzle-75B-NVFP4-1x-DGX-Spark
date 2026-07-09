#!/usr/bin/env python3
"""Single-stream wall tok/s probes for Nemotron Puzzle on DGX Spark."""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("SERVED_NAME", "nemotron-puzzle-75b-nvfp4")
URL = f"{BASE}/v1/chat/completions"

PROBES = [
    ("structured_count", "List integers 1 to 60 comma-separated, nothing else.", 140),
    ("code_snippet", "Write a Python function fib(n) with memoization. Code only, no prose.", 180),
    ("prose", "Write one dense technical paragraph about CUDA shared memory. About 100 words.", 160),
]


def post(payload: dict, timeout: float = 300.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body), time.time() - t0


def main() -> None:
    # warmup
    post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )

    print(f"endpoint={URL}  model={MODEL}")
    print("=== single-stream wall tok/s ===")
    rates = []
    for name, prompt, max_tok in PROBES:
        j, elapsed = post(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tok,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        u = j.get("usage") or {}
        c = int(u.get("completion_tokens") or 0)
        p = int(u.get("prompt_tokens") or 0)
        tps = c / elapsed if elapsed else 0.0
        rates.append(tps)
        preview = ((j["choices"][0]["message"].get("content") or "")[:60]).replace("\n", " ")
        print(
            f"{name:18}  prompt={p:3d}  completion={c:4d}  "
            f"wall={elapsed:6.2f}s  tok/s={tps:6.1f}  preview={preview!r}"
        )

    print(f"\nmean={statistics.mean(rates):.1f}  min={min(rates):.1f}  max={max(rates):.1f}")


if __name__ == "__main__":
    main()
