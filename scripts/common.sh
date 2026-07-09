#!/usr/bin/env bash
# Shared defaults for Puzzle serve scripts.
# Override any variable before calling start-puzzle.sh.

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nemotron-puzzle-75b-nvfp4}"
GMU="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_LEN="${MAX_MODEL_LEN:-262144}"
MAX_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED="${MAX_NUM_BATCHED_TOKENS:-8192}"
MTP_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
