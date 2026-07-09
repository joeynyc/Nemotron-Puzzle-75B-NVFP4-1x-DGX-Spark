# 4-stream wall-clock demo

Four parallel live `/v1/chat/completions` streams against the local Puzzle
server, rendered as a **1080p / 30 fps** dashboard with **no speedup**
(frame *N* = wall clock *N/30* seconds).

## Reference take (in this repo)

| File | Notes |
| --- | --- |
| [`nemotron-puzzle-4stream-30fps.mp4`](nemotron-puzzle-4stream-30fps.mp4) | 1080p · 30 fps · ~15.5 s · ~712 KB |
| [`../benchmarks/4stream-recording.json`](../benchmarks/4stream-recording.json) | **75.3 tok/s** aggregate (834 tokens / 11.08 s) |

https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark/raw/main/demo/nemotron-puzzle-4stream-30fps.mp4

<video src="nemotron-puzzle-4stream-30fps.mp4" controls width="100%"></video>

## Regenerate

**Requirements**

- Server up (`../scripts/start-puzzle.sh`)
- Python 3.10+: `httpx`, `Pillow`
- `ffmpeg` on `PATH`

```bash
pip install httpx pillow
python3 record_and_render.py
# → demo/out/nemotron-puzzle-4stream-30fps.mp4
# → demo/out/recording.json
```
