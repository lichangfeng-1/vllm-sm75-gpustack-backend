#!/usr/bin/env bash
# 构建 v11 响应式薄层（父 v10）：面板壳 + 内嵌测速工具宽度
# 逻辑审查：已过；实机跑过：2026-10-25
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CTX=$REPO/gpustack/layers/panel-responsive/public
DF=$REPO/gpustack/Dockerfile.panel-responsive-v1
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v10}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v11}
APP_SHA=${APP_SHA:-cfd50aaf18b36fb49a8c58335cf272a4bca271ff9f5b4c3fc331aa2bad5fc153}
for f in "$CTX/style.css" "$CTX/speedtest.html" "$DF"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null
docker build --pull=false --build-arg PARENT="$PARENT" -f "$DF" -t "$IMAGE" "$CTX"
want=$(sha256sum "$CTX/style.css" | cut -d' ' -f1)
got=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/style.css | cut -d' ' -f1)
test "$got" = "$want" || { echo "FAIL: style.css sha" >&2; exit 7; }
got2=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html | cut -d' ' -f1)
want2=$(sha256sum "$CTX/speedtest.html" | cut -d' ' -f1)
test "$got2" = "$want2" || { echo "FAIL: speedtest.html sha" >&2; exit 8; }
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q -- '--sm75-sidebar-w' /opt/sm75-workbench/console/public/style.css" || { echo FAIL_responsive >&2; exit 9; }
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q 'min(1400px, 96vw)' /opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html" || { echo FAIL_ext_width >&2; exit 10; }
app=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$app" = "$APP_SHA" || { echo "FAIL: app.js sha $app" >&2; exit 11; }
docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "PANEL_RESPONSIVE_V11_OK image=$IMAGE"
