#!/usr/bin/env python3
"""Four concurrent streams → aggregate tok/s (requires max_num_seqs >= 4)."""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("SERVED_NAME", "nemotron-puzzle-75b-nvfp4")
URL = f"{BASE}/v1/chat/completions"
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "100"))
N = int(os.environ.get("BENCH_STREAMS", "4"))

PROMPTS = [
    "List integers 1 to 40 comma-separated only. (stream 0)",
    "List integers 1 to 40 comma-separated only. (stream 1)",
    "List integers 1 to 40 comma-separated only. (stream 2)",
    "List integers 1 to 40 comma-separated only. (stream 3)",
]


def one(i: int):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read())
    elapsed = time.time() - t0
    c = int((j.get("usage") or {}).get("completion_tokens") or 0)
    return c, elapsed, c / elapsed if elapsed else 0.0


def main() -> None:
    # warmup
    one(0)
    print(f"endpoint={URL}  model={MODEL}  streams={N}  max_tokens={MAX_TOKENS}")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        outs = list(ex.map(one, range(N)))
    wall = time.time() - t0
    total = sum(o[0] for o in outs)
    print(f"per-stream tok/s: {[round(o[2], 1) for o in outs]}")
    print(f"per-stream wall:  {[round(o[1], 2) for o in outs]}")
    print(f"aggregate: {total} tokens in {wall:.2f}s => {total / wall:.1f} tok/s")
    print(f"mean per-stream: {statistics.mean(o[2] for o in outs):.1f}")


if __name__ == "__main__":
    main()
