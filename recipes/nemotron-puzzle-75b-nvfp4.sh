#!/usr/bin/env bash
# Recipe: Nemotron Puzzle 75B-A9B NVFP4 — 1× DGX Spark (GB10)
# Intended to run *inside* nvcr.io/nvidia/vllm:26.06-py3 (or as docker CMD args).
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nemotron-puzzle-75b-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GMU="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_LEN="${MAX_MODEL_LEN:-262144}"
MAX_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED="${MAX_NUM_BATCHED_TOKENS:-8192}"
MTP_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --mamba-backend flashinfer \
  --async-scheduling \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --max-num-batched-tokens "${MAX_BATCHED}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GMU}"
