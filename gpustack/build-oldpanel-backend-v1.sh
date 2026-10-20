#!/usr/bin/env bash
# build-oldpanel-backend-v1.sh — 薄层构建：旧 Ultra 面板引擎 + v4 GPUStack/面板入口
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
OLD_IMAGE="${OLD_IMAGE:-vllm-sm75:v0.1.7-ultra-beta-to20s-pcieipc-incple-20261010}"
PANEL_IMAGE="${PANEL_IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v4}"
TAG="${TAG:-vllm-sm75-gpustack:v0.1.7-beta-ultra-pcieipc-incple-panel-v1}"

echo "OLD_IMAGE=$OLD_IMAGE"
echo "PANEL_IMAGE=$PANEL_IMAGE"
echo "TAG=$TAG"
docker build \
  -f "$ROOT/Dockerfile.oldpanel-backend-v1" \
  --build-arg OLD_IMAGE="$OLD_IMAGE" \
  --build-arg PANEL_IMAGE="$PANEL_IMAGE" \
  -t "$TAG" \
  "$ROOT"
echo "BUILT $TAG"
