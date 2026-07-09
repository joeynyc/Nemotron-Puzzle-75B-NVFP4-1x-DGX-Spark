#!/usr/bin/env bash
# Start Nemotron Puzzle 75B NVFP4 on 1× DGX Spark via NGC vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/common.sh"

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.06-py3}"
NAME="${CONTAINER_NAME:-nemotron-puzzle-75b}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

mkdir -p "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Removing existing container: ${NAME}"
  docker rm -f "${NAME}" >/dev/null
fi

echo "Starting ${NAME}"
echo "  image:  ${IMAGE}"
echo "  port:   ${PORT}"
echo "  cache:  ${HF_CACHE}"
echo "  model:  ${MODEL}"
echo

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE}" \
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
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

echo
echo "Container started. First boot can take 6–15 minutes (download + load + compile)."
echo "  logs:   docker logs -f ${NAME}"
echo "  health: curl -s http://127.0.0.1:${PORT}/v1/models"
echo
"${ROOT}/scripts/wait-ready.sh" || true
