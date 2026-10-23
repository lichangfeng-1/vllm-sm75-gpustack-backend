#!/usr/bin/env bash
# 在服务器侧（带 v8 父镜像）构建 v9 薄层：替换 pi-ai adapter，把 vLLM 的
# At most N image(s) 映射成 IMAGE_OFFLOAD_REQUIRED，交给自带 image-offload 插件。
# 布局：件在 gpustack/layers/harness/，Dockerfile 在 gpustack/，本脚本在 gpustack/。
# 逻辑审查：已逐行检查；实机：2026-10-23 已跑（v9 薄层构建成功，镜像 18a2c57e5a13）。
set -euo pipefail

REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CTX=$REPO/gpustack/layers/harness
DF=$REPO/gpustack/Dockerfile.harness-offload-v1
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v8}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v9}
PANEL_SHA=${PANEL_SHA:-cfd50aaf18b36fb49a8c58335cf272a4bca271ff9f5b4c3fc331aa2bad5fc153}

for f in "$CTX/dsh-llm-pi-ai-index.js" "$DF" "$CTX/verify-image-offload-v1.py"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null
python3 "$CTX/verify-image-offload-v1.py" "$CTX/dsh-llm-pi-ai-index.js"
docker build --pull=false --build-arg PARENT="$PARENT" -f "$DF" -t "$IMAGE" "$CTX"

want=$(sha256sum "$CTX/dsh-llm-pi-ai-index.js" | cut -d' ' -f1)
got=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/harness/node_modules/@deepseek-ai/dsh-llm-pi-ai/lib/index.js | cut -d' ' -f1)
test "$got" = "$want" || { echo "FAIL: in-image adapter sha $got != $want" >&2; exit 7; }
plugin=$(docker run --rm --entrypoint sh "$IMAGE" -c "test -d /opt/harness/node_modules/@deepseek-ai/dsh-compaction-image-offload && echo yes")
test "$plugin" = "yes" || { echo "FAIL: image-offload plugin missing" >&2; exit 8; }
app=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$app" = "$PANEL_SHA" || { echo "FAIL: panel app.js sha $app != $PANEL_SHA" >&2; exit 9; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "HARNESS_OFFLOAD_V9_OK adapter_sha=$want image=$IMAGE"
