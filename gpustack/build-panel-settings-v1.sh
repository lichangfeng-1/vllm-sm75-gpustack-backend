#!/usr/bin/env bash
# 在服务器侧（带 v7 父镜像）构建 v8 薄层：只替换 console/public/app.js。
# 逻辑审查：已逐行检查；实机：2026-10-23 已跑（v8 薄层构建成功）。
# 用法：在包含 app.js / Dockerfile / verify-settings-v1.py 的目录执行本脚本。
set -euo pipefail

CTX=$(cd -- "$(dirname -- "$0")" && pwd)
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v7}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v8}

for f in app.js Dockerfile verify-settings-v1.py; do
  test -s "$CTX/$f" || { echo "BLOCK: missing $CTX/$f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null
python3 "$CTX/verify-settings-v1.py" "$CTX/app.js"
docker build --pull=false --build-arg PARENT="$PARENT" -f "$CTX/Dockerfile" -t "$IMAGE" "$CTX"

want=$(sha256sum "$CTX/app.js" | cut -d' ' -f1)
got=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$got" = "$want" || { echo "FAIL: in-image app.js sha $got != $want" >&2; exit 7; }
tabs=$(docker run --rm --entrypoint sh "$IMAGE" -c "grep -c 'data-settings-tab' /opt/sm75-workbench/console/public/index.html")
test "$tabs" = "6" || { echo "FAIL: settings tabs=$tabs, want 6" >&2; exit 8; }
button=$(docker run --rm --entrypoint sh "$IMAGE" -c "grep -c 'openUnifiedSettings' /opt/sm75-workbench/console/public/index.html")
test "$button" = "1" || { echo "FAIL: settings button count=$button, want 1" >&2; exit 9; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "PANEL_SETTINGS_V8_OK app_sha=$want image=$IMAGE"
