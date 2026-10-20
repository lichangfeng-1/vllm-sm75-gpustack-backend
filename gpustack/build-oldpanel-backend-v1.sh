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

for img in "$OLD_IMAGE" "$PANEL_IMAGE"; do
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    echo "红：本地缺少镜像 $img" >&2
    if [ "$img" = "$PANEL_IMAGE" ]; then
      echo "  先构建面板后端：bash gpustack/build-backend-v1.sh（产物默认 vllm-sm75-gpustack:v0.1.7-beta2-v4）" >&2
      echo "  或用 PANEL_IMAGE=<你已有的面板后端 tag> 覆盖。" >&2
    else
      echo "  本路线需要旧 Ultra 引擎镜像；没有它请改走 v4 路线：bash gpustack/build-backend-v1.sh" >&2
      echo "  或用 OLD_IMAGE=<本机旧 Ultra tag> 覆盖。" >&2
    fi
    exit 2
  fi
done

docker build \
  -f "$ROOT/Dockerfile.oldpanel-backend-v1" \
  --build-arg OLD_IMAGE="$OLD_IMAGE" \
  --build-arg PANEL_IMAGE="$PANEL_IMAGE" \
  -t "$TAG" \
  "$ROOT"
echo "BUILT $TAG"
