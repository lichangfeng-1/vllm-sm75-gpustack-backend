#!/usr/bin/env bash
# 0→1：上游 vLLM 0.30.0 -> 公开 v7 基座 -> final-v1 -> SQL 提示词审计层。
# 逻辑审查：2026-10-29；实机跑过：2026-10-29（v7 已存在；两个薄层在 v7 上构建并验装通过；
# v7 本身的 two-stage 构建调用既有 build-backend-v1.sh，本次未重跑）。
# 用法：
#   bash gpustack/build-latest-from-zero-v1.sh
#   SKIP_V7=1 bash gpustack/build-latest-from-zero-v1.sh   # 已有 v7，只叠两个薄层
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
V7=${V7:-vllm-sm75-gpustack:v0.1.7-beta2-v7}
FINAL=${FINAL:-vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1}
AUDIT=${AUDIT:-vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1-sqlprompt-audit}
SQL_28=28acd6a809dd4b8ccc45c427dfdbadefc7a8d047375b7812e2933e313de57fc7
SPEED_28=cb05faa3cdc196c71da7903ac09fef36b8c25a04118211ba98efc10fdeb05bd1
SQL_29=6f8c836fe3702eef7909463f7051de53666617080db5c98c9dfc200eaea829da
MANIFEST_29=9acabdbc3673c50d02488def1f761065b40833290aebe929d6d8295334333f84

if [ "${SKIP_V7:-0}" != 1 ]; then
  bash "$REPO_ROOT/gpustack/build-backend-v1.sh" --two-stage
fi
docker image inspect "$V7" >/dev/null

docker build --pull=false -f "$REPO_ROOT/gpustack/Dockerfile.final-v1-20261029" \
  --build-arg PARENT="$V7" -t "$FINAL" "$REPO_ROOT"
docker build --pull=false -f "$REPO_ROOT/gpustack/Dockerfile.sql-prompt-audit-v20261029" \
  --build-arg PARENT="$FINAL" -t "$AUDIT" "$REPO_ROOT"

got=$(docker run --rm --entrypoint sha256sum "$FINAL" /opt/sm75-workbench/extensions/sql/sql_benchmark.html | cut -d' ' -f1)
test "$got" = "$SQL_28" || { echo "FAIL: final SQL sha $got" >&2; exit 7; }
got=$(docker run --rm --entrypoint sha256sum "$FINAL" /opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html | cut -d' ' -f1)
test "$got" = "$SPEED_28" || { echo "FAIL: final speedtest sha $got" >&2; exit 8; }
docker run --rm --entrypoint grep "$FINAL" -qF 'maxTokens: Math.min(65536' /opt/sm75-workbench/console/harness-config.mjs \
  || { echo "FAIL: console maxTokens marker" >&2; exit 9; }
docker run --rm --entrypoint grep "$FINAL" -qF 'sm75-finalanswer' /opt/sm75-workbench/extensions/sql/sql_benchmark.html \
  || { echo "FAIL: finalanswer marker" >&2; exit 10; }

got=$(docker run --rm --entrypoint sha256sum "$AUDIT" /opt/sm75-workbench/extensions/sql/sql_benchmark.html | cut -d' ' -f1)
test "$got" = "$SQL_29" || { echo "FAIL: audit SQL sha $got" >&2; exit 11; }
got=$(docker run --rm --entrypoint sha256sum "$AUDIT" /opt/sm75-workbench/extensions/sql/manifest.json | cut -d' ' -f1)
test "$got" = "$MANIFEST_29" || { echo "FAIL: audit manifest sha $got" >&2; exit 12; }
docker run --rm --entrypoint grep "$AUDIT" -qF 'Preserve the original formatting of fiscal period labels' /opt/sm75-workbench/extensions/sql/sql_benchmark.html \
  || { echo "FAIL: prompt marker" >&2; exit 13; }
docker run --rm --entrypoint grep "$AUDIT" -qF 'clip(firstRow[col], 500)' /opt/sm75-workbench/extensions/sql/sql_benchmark.html \
  || { echo "FAIL: truncation marker" >&2; exit 14; }

docker image inspect "$FINAL" --format 'FINAL_ID={{.Id}} SIZE={{.Size}}'
docker image inspect "$AUDIT" --format 'AUDIT_ID={{.Id}} SIZE={{.Size}}'
echo "ZERO_TO_LATEST_OK final=$FINAL audit=$AUDIT"
