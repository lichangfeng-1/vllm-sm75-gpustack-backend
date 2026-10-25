#!/usr/bin/env bash
# v12 面板 UI 修正薄层（父 v11）：登录卡片只放大不缩小、代码块样式、命令换行
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CTX=$REPO/gpustack/layers/panel-ui/public
DF=$REPO/gpustack/Dockerfile.panel-ui-v1
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v11}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v12}
APP_SHA=${APP_SHA:-cfd50aaf18b36fb49a8c58335cf272a4bca271ff9f5b4c3fc331aa2bad5fc153}
for f in "$CTX/index.html" "$CTX/style.css" "$DF"; do test -s "$f" || { echo "BLOCK: $f" >&2; exit 3; }; done
docker image inspect "$PARENT" >/dev/null
docker build --pull=false --build-arg PARENT="$PARENT" -f "$DF" -t "$IMAGE" "$CTX"
for f in index.html style.css; do
  want=$(sha256sum "$CTX/$f" | cut -d' ' -f1)
  got=$(docker run --rm --entrypoint sha256sum "$IMAGE" "/opt/sm75-workbench/console/public/$f" | cut -d' ' -f1)
  test "$got" = "$want" || { echo "FAIL sha $f" >&2; exit 7; }
done
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q 'max(440px, calc(440px' /opt/sm75-workbench/console/public/style.css" || { echo FAIL_width >&2; exit 8; }
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q 'panel/key; echo' /opt/sm75-workbench/console/public/index.html" || { echo FAIL_echo >&2; exit 9; }
app=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$app" = "$APP_SHA" || { echo "FAIL app.js $app" >&2; exit 10; }
docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "PANEL_UI_V12_OK image=$IMAGE"
