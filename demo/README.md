# 4-stream wall-clock demo

Records four parallel live `/v1/chat/completions` streams against the local
Puzzle server, then renders a **1080p / 30 fps** dashboard video with **no speedup**
(frame *N* = wall clock *N/30* seconds).

## Requirements

- Server up (`../scripts/start-puzzle.sh`)
- Python 3.10+: `httpx`, `Pillow`
- `ffmpeg` on `PATH`

```bash
pip install httpx pillow
# or: apt install ffmpeg && pip install --user httpx pillow
```

## Run

```bash
python3 record_and_render.py
# → demo/out/nemotron-puzzle-4stream-30fps.mp4
# → demo/out/recording.json  (token counts + aggregate tok/s)
```

Our reference take: **75.3 tok/s aggregate** (834 tokens / 11.08 s) — see
[`../benchmarks/4stream-recording.json`](../benchmarks/4stream-recording.json).
