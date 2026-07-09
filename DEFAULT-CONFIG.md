# Default config — Nemotron Puzzle 75B NVFP4 on 1× DGX Spark

Byte-exact settings validated 2026-07-08 on GB10.

## Container

| Key | Value |
| --- | --- |
| Image | `nvcr.io/nvidia/vllm:26.06-py3` |
| Name | `nemotron-puzzle-75b` |
| Port | `8000` |
| GPU | all |
| Restart | `unless-stopped` |
| Volume | `$HOME/.cache/huggingface` → `/root/.cache/huggingface` |

### Environment

```bash
HF_HOME=/root/.cache/huggingface
NVIDIA_TF32_OVERRIDE=1
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## vLLM command (after / recommended)

```bash
vllm serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
  --served-model-name nemotron-puzzle-75b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --mamba-backend flashinfer \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.88
```

Canonical script: [`scripts/start-puzzle.sh`](scripts/start-puzzle.sh) · [`recipes/nemotron-puzzle-75b-nvfp4.sh`](recipes/nemotron-puzzle-75b-nvfp4.sh).

## Before vs after (flags only)

| Flag | Before | After | Why |
| --- | --- | --- | --- |
| `max_num_seqs` | **1** | **4** | Multi-session / 4-stream actually concurrent |
| `enable-prefix-caching` | off | **on** | Multi-turn agent TTFT |
| `max-num-batched-tokens` | default (~2048 under MTP warn) | **8192** | Avoid MTP schedule bottleneck |
| `gpu-memory-utilization` | 0.85 | **0.88** | Slightly more KV headroom |
| MTP | 3 | 3 | Already strong accept rate |
| `max-model-len` | 262144 | 262144 | Keep long context; right-size if you never need 256K |

## Client tips

```json
{
  "model": "nemotron-puzzle-75b-nvfp4",
  "chat_template_kwargs": { "enable_thinking": false },
  "temperature": 0.3,
  "max_tokens": 2048
}
```

- **Thinking off** improves structured/code MTP accept and avoids empty `content` with huge `reasoning` fields.
- For pure peak solo decode experiments, set `max_num_seqs=1` again; aggregate multi-user will regress.

## Resource notes

| Resource | Observed |
| --- | --- |
| Model load | ~50 GiB |
| Available KV (after) | ~55 GiB |
| GPU KV tokens (after) | ~2.0M |
| Host RAM while serving | nearly full unified pool — expected |

If OOM at boot: drop `gpu-memory-utilization` to `0.85`, or `max-model-len` to `65536` / `131072`, or `max_num_seqs` to `2`.

## API

| | |
| --- | --- |
| Base URL | `http://127.0.0.1:8000/v1` |
| Models | `GET /v1/models` |
| Chat | `POST /v1/chat/completions` |
| Metrics | `GET /metrics` (Prometheus; includes spec-decode counters) |

Useful metric names:

- `vllm:num_requests_running` / `vllm:num_requests_waiting`
- `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total`
- `vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_draft_tokens_total`
