#!/usr/bin/env bash
# 构建 v10 面板登录/禁用态修复薄层（父 v9）：替换 public/index.html 与 public/style.css
# 布局：件在 gpustack/layers/panel-login/public/，Dockerfile 在 gpustack/
# 逻辑审查：已过；实机跑过：2026-10-25
set -euo pipefail

REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CTX=$REPO/gpustack/layers/panel-login/public
DF=$REPO/gpustack/Dockerfile.panel-login-v1
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v9}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v10}
APP_SHA=${APP_SHA:-cfd50aaf18b36fb49a8c58335cf272a4bca271ff9f5b4c3fc331aa2bad5fc153}

for f in "$CTX/index.html" "$CTX/style.css" "$DF"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null
docker build --pull=false --build-arg PARENT="$PARENT" -f "$DF" -t "$IMAGE" "$CTX"

for f in index.html style.css; do
  want=$(sha256sum "$CTX/$f" | cut -d' ' -f1)
  got=$(docker run --rm --entrypoint sha256sum "$IMAGE" "/opt/sm75-workbench/console/public/$f" | cut -d' ' -f1)
  test "$got" = "$want" || { echo "FAIL: $f sha $got != $want" >&2; exit 7; }
done
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q '440px \* var(--ui-fs' /opt/sm75-workbench/console/public/style.css" || { echo FAIL_scaled_width >&2; exit 8; }
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q 'button.primary:disabled' /opt/sm75-workbench/console/public/style.css" || { echo FAIL_disabled_style >&2; exit 9; }
docker run --rm --entrypoint sh "$IMAGE" -c "grep -q 'docker ps -q --filter name=run-0' /opt/sm75-workbench/console/public/index.html" || { echo FAIL_help_cmd >&2; exit 10; }
app=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/console/public/app.js | cut -d' ' -f1)
test "$app" = "$APP_SHA" || { echo "FAIL: app.js sha $app" >&2; exit 11; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "PANEL_LOGIN_V10_OK image=$IMAGE"
