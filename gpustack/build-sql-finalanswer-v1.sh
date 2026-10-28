#!/usr/bin/env bash
# 构建 SQL 判分扩展 sm75-finalanswer-v1 薄层（upstream 2aa49062 + 本地补丁），只覆盖 panel sql 扩展。
# 逻辑审查：已过（2026-10-28）；实机跑过：未跑（本阶段只交付配方，不构建镜像）。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
EXT=$REPO/gpustack/layers/panel/extensions/sql
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-sqljudge-v1}
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT

for f in "$EXT/sql_benchmark-v20261028.html" "$EXT/manifest-v20261028.json" "$EXT/UPSTREAM-v20261028.md" \
         "$REPO/gpustack/Dockerfile.sql-finalanswer-v20261028"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null

cp "$EXT/sql_benchmark-v20261028.html" "$CTX/sql_benchmark-v20261028.html"
cp "$EXT/manifest-v20261028.json" "$CTX/manifest-v20261028.json"
cp "$EXT/UPSTREAM-v20261028.md" "$CTX/UPSTREAM-v20261028.md"
cp "$REPO/gpustack/Dockerfile.sql-finalanswer-v20261028" "$CTX/Dockerfile"

docker build --pull=false --build-arg PARENT="$PARENT" -f "$CTX/Dockerfile" -t "$IMAGE" "$CTX"

want=$(sha256sum "$CTX/sql_benchmark-v20261028.html" | cut -d' ' -f1)
got=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/extensions/sql/sql_benchmark.html | cut -d' ' -f1)
test "$got" = "$want" || { echo "FAIL: entry html sha" >&2; exit 7; }
got_ver=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/extensions/sql/sql_benchmark-v20261028.html | cut -d' ' -f1)
test "$got_ver" = "$want" || { echo "FAIL: versioned html sha" >&2; exit 8; }
docker run --rm --entrypoint cat "$IMAGE" /opt/sm75-workbench/extensions/sql/manifest.json | grep -q 'sm75-finalanswer-v1' || { echo FAIL_manifest >&2; exit 9; }
docker run --rm --entrypoint grep "$IMAGE" -q 'sm75-finalanswer-v1' /opt/sm75-workbench/extensions/sql/sql_benchmark.html || { echo FAIL_marker >&2; exit 10; }
docker run --rm --entrypoint grep "$IMAGE" -q 'checkedFirstRowColumns' /opt/sm75-workbench/extensions/sql/sql_benchmark.html || { echo FAIL_firstrow_patch >&2; exit 11; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "SQL_FINALANSWER_LAYER_OK image=$IMAGE"
