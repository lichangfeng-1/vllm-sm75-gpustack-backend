#!/usr/bin/env bash
# 在服务器侧（带 v7 父镜像）构建 v8 薄层：只替换 console/public/app.js。
# 布局：件在 gpustack/layers/panel/，Dockerfile 在 gpustack/，本脚本在 gpustack/。
# 逻辑审查：已逐行检查；实机：2026-10-23 已跑（v8 薄层构建成功）。
set -euo pipefail

REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CTX=$REPO/gpustack/layers/panel/public
VERIFY=$REPO/gpustack/layers/panel/verify-settings-v1.py
DF=$REPO/gpustack/Dockerfile.panel-settings-v1
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v7}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v8}

for f in "$CTX/app.js" "$DF" "$VERIFY"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null
python3 "$VERIFY" "$CTX/app.js"
docker build --pull=false --build-arg PARENT="$PARENT" -f "$DF" -t "$IMAGE" "$CTX"

want=$(sha256sum "$CTX/app.js" | cut -d' ' -f1)
got=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$got" = "$want" || { echo "FAIL: in-image app.js sha $got != $want" >&2; exit 7; }
tabs=$(docker run --rm --entrypoint sh "$IMAGE" -c "grep -c 'data-settings-tab' /opt/sm75-workbench/console/public/index.html")
test "$tabs" = "6" || { echo "FAIL: settings tabs=$tabs, want 6" >&2; exit 8; }
button=$(docker run --rm --entrypoint sh "$IMAGE" -c "grep -c 'openUnifiedSettings' /opt/sm75-workbench/console/public/index.html")
test "$button" = "1" || { echo "FAIL: settings button count=$button, want 1" >&2; exit 9; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "PANEL_SETTINGS_V8_OK app_sha=$want image=$IMAGE"
