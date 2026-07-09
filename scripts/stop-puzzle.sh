#!/usr/bin/env bash
set -euo pipefail
NAME="${CONTAINER_NAME:-nemotron-puzzle-75b}"
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Stopping and removing ${NAME}…"
  docker rm -f "${NAME}"
  echo "Done."
else
  echo "Container ${NAME} not found."
fi
