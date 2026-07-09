# Nemotron Puzzle 75B NVFP4 on 1× DGX Spark — ~40 tok/s solo, ~75 tok/s ×4 parallel

Serve [**NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4)
on a single **NVIDIA DGX Spark (GB10)** with the official NGC vLLM container,
**MTP×3** speculative decoding, prefix caching, and **4 concurrent sequences**.

Validated **2026-07-08** on `spark-db08` (GB10, driver 580, CUDA 13, unified ~121 GiB).

| Mode | Before (stock-ish serve) | After (this recipe) |
| --- | ---: | ---: |
| **Single stream** | ~38–42 tok/s | ~36–40 tok/s (decode was already fine) |
| **4 parallel streams** | **serialized** (`max_num_seqs=1`) ≈ **~40 tok/s total** while others **queued** | **~75 tok/s aggregate** (all four live) |
| Prefix cache | off | **on** (~70% hit rate under agent load) |
| Concurrent capacity | 1 | **4** |

**Honest takeaway:** the big win is **concurrency + prefix cache**, not a free jump in solo tok/s. MTP acceptance was already healthy (~75% draft tokens). Turning `max_num_seqs` from 1 → 4 is what made multi-session / multi-stream actually parallel.

### Demo — 4 parallel streams (wall-clock, no speedup)

Four live `/v1/chat/completions` requests on one GB10. **~75 tok/s aggregate.** 1080p · 30 fps · real time.

https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark/raw/main/demo/nemotron-puzzle-4stream-30fps.mp4

<video src="demo/nemotron-puzzle-4stream-30fps.mp4" controls width="100%"></video>

[Download MP4](demo/nemotron-puzzle-4stream-30fps.mp4) · [poster](demo/poster.png) · regenerate with [`demo/record_and_render.py`](demo/record_and_render.py)

---

## Hardware / software

| Item | Value |
| --- | --- |
| Device | 1× NVIDIA DGX Spark · **GB10** · sm_121 |
| Memory | ~128 GB unified (≈121 GiB usable) |
| Image | [`nvcr.io/nvidia/vllm:26.06-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) (vLLM **0.22.1**) |
| Model | `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4` (~50 GiB weights) |
| Spec decode | native **MTP**, `num_speculative_tokens=3` |
| KV | auto **fp8_e4m3** |
| MoE path (observed) | FlashInfer CUTLASS NVFP4 |

No custom vLLM fork. No TP2. No second Spark required.

---

## Quick start

### 0. Prerequisites

- DGX Spark with Docker + NVIDIA Container Toolkit
- ~60 GiB free disk for image + model cache (first pull is large)
- Hugging Face access for the NVFP4 checkpoint (accept model terms if required)

```bash
# optional: export a token if the model is gated
export HF_TOKEN=hf_...
```

### 1. Clone

```bash
git clone https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark.git
cd Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark
```

### 2. Start the server

```bash
./scripts/start-puzzle.sh
```

First boot downloads weights into `~/.cache/huggingface` (mounted into the container).
Expect **~6–15 minutes** on a cold cache (weight load + torch.compile + CUDA graphs).

Health check:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

Stop:

```bash
./scripts/stop-puzzle.sh
```

### 3. Smoke chat

```bash
./examples/chat.sh "Write a haiku about GB10 unified memory."
```

Or raw OpenAI-compatible:

```bash
./examples/openai-curl.sh
```

### 4. Benchmarks

```bash
# single-stream wall tok/s (structured / code / prose)
./scripts/bench-single.py

# four concurrent streams → aggregate tok/s
./scripts/bench-4stream.py
```

Protocol and our measured numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

---

## What this recipe actually changes

Default / earlier serve (what we had “before”):

```text
--gpu-memory-utilization 0.85
--max-model-len 262144
--max-num-seqs 1
--speculative-config {"method":"mtp","num_speculative_tokens":3}
# no --enable-prefix-caching
# no --max-num-batched-tokens (vLLM warned: scheduled tokens capped ~2048 under MTP)
```

**This recipe (after):**

```text
--enable-prefix-caching
--max-num-batched-tokens 8192
--max-num-seqs 4
--gpu-memory-utilization 0.88
--max-model-len 262144
--mamba-backend flashinfer
--async-scheduling
--speculative-config {"method":"mtp","num_speculative_tokens":3}
+ tool/reasoning parsers for Nemotron v3 + qwen3_coder tools
```

Full flag list and env: [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) · recipe file: [`recipes/nemotron-puzzle-75b-nvfp4.sh`](recipes/nemotron-puzzle-75b-nvfp4.sh).

### Boot evidence (after)

```text
Available KV cache memory: ~54.7 GiB
GPU KV cache size: ~2,000,731 tokens
enable_prefix_caching=True
max_num_seqs=4
max_num_batched_tokens=8192
cudagraph_capture_sizes up to 32
```

Tradeoff vs `max_num_seqs=1`: fewer total KV *tokens* in the pool (hybrid/mamba + concurrent graphs), still plenty for normal chat; less headroom for many simultaneous 256K contexts.

---

## Measured results (2026-07-08)

### Single stream (wall = `completion_tokens / wall_time`, temp 0, warm engine)

| Workload | Before | After |
| --- | ---: | ---: |
| Structured count | 41.8 | 35.7 |
| Code | 40.7 | 36.6 |
| Prose | 32.3 | 35.4 |
| **Mean** | **38.3** | **35.9** |

### Four parallel streams (after only — before could not run 4 live)

| Metric | Value |
| --- | ---: |
| Completion tokens (sum) | 834 |
| Wall span | 11.08 s |
| **Aggregate** | **75.3 tok/s** |
| Per-stream (earlier probe) | ~18–21 tok/s each |
| TTFT (4-stream demo) | ~0.45 s |

MTP lifetime accept (before reload): ~**75%** draft tokens, mean accept length ~2.5–3.2 with 3 speculative tokens. Content-dependent: structured/code drafts better than free prose.

Details: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

---

## Demo video

Checked-in reference take (also at top of README):

- **[`demo/nemotron-puzzle-4stream-30fps.mp4`](demo/nemotron-puzzle-4stream-30fps.mp4)** — 1080p / 30 fps, wall-clock, **75.3 tok/s aggregate**
- Metrics: [`benchmarks/4stream-recording.json`](benchmarks/4stream-recording.json)

Regenerate on your own Spark:

```bash
# server must already be up
python3 demo/record_and_render.py
# → demo/out/nemotron-puzzle-4stream-30fps.mp4
```

Requires: `httpx`, `Pillow`, `ffmpeg` on `PATH`. Details: [`demo/README.md`](demo/README.md).

---

## Agent / Hermes notes (optional)

If you front this with [Hermes](https://github.com/NousResearch/hermes-agent) or similar:

| Setting | Suggestion |
| --- | --- |
| Endpoint | `http://127.0.0.1:8000/v1` |
| Model id | `nemotron-puzzle-75b-nvfp4` |
| Thinking | `chat_template_kwargs: { enable_thinking: false }` — better MTP accept + cleaner tokens |
| `max_tokens` | 1k–2k for chat; 16k only for long coding sessions |
| Clarify / multi-choice tool | Lower `clarify_timeout` (e.g. 90s). A 600s clarify wait can look like a “dead” Telegram bot while the GPU is fine. |

These are client-side; the serve recipe does not depend on Hermes.

---

## Gotchas (hit live)

1. **First request after boot is slow** — compile/graph warmup. Warm with a tiny completion before benchmarking or pointing agents at it.
2. **Cold HF cache** — first `docker run` can spend many minutes downloading ~50 GiB. Mount `~/.cache/huggingface` (the start script does).
3. **`max_num_seqs=1` silently serializes multi-user agents** — Hermes can open many sessions; vLLM only runs one. Raise seqs if you multi-task.
4. **vLLM warning under MTP** — without `--max-num-batched-tokens`, scheduled tokens can stick at 2048. We set **8192**.
5. **Unified memory is tight** — ~50 GiB weights + KV + host. Avoid heavy co-tenants (big Comfy/Firecrawl builds) on the same Spark while serving.
6. **Don’t copy NCCL “LL” knobs from dense multi-node recipes** — irrelevant on solo; on multi-node MoE they can *hurt* (see MiniMax notes elsewhere).
7. **Container restart ≈ minutes offline** — drain agents first.

More ops detail: [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md).

---

## Credits

- **Model:** [NVIDIA Nemotron Labs — Puzzle 75B-A9B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4)
- **Runtime:** [NVIDIA NGC vLLM](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) / [vLLM](https://github.com/vllm-project/vllm)
- **Hardware platform:** [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- Community multi-node Spark vLLM work (not required here, useful context): [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)

**This repo’s contribution:** measured solo baseline, the concurrency/prefix/batched-token recipe that unlocked ~75 tok/s aggregate on one GB10, launch scripts, bench harnesses, and the wall-clock 4-stream demo pipeline.

---

## License

Scripts and docs: [MIT](LICENSE).

Model weights, NGC image, and CUDA/vLLM stack remain under their respective NVIDIA / upstream licenses. You must accept Hugging Face / NGC terms yourself.
