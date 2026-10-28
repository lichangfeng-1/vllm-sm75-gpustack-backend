#!/usr/bin/env bash
# 构建测速扩展 v3.5 薄层（upstream 07895cff），只覆盖 panel 扩展文件。
# 逻辑审查：已过（2026-10-28）；实机跑过：2026-10-28（父 opt15，HTML/后端/h2/标记四项验装通过）。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
EXT=$REPO/gpustack/layers/panel/extensions/speedtest
RESP=$REPO/gpustack/layers/panel-responsive/public
PARENT=${PARENT:-vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15}
IMAGE=${IMAGE:-vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-st35-v1}
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT

for f in "$RESP/speedtest-v20261021.html" "$EXT/LLM_Speed_Test_v3_Leaderboard-v20261021.html" \
         "$EXT/llm_test_backend-v20261021.py" "$EXT/manifest-v20261021.json" "$EXT/UPSTREAM.md" \
         "$REPO/gpustack/Dockerfile.speedtest-v35-v1"; do
  test -s "$f" || { echo "BLOCK: missing $f" >&2; exit 3; }
done
docker image inspect "$PARENT" >/dev/null

cp "$RESP/speedtest-v20261021.html" "$CTX/LLM_Speed_Test_v3_Python_Backend.html"
cp "$EXT/LLM_Speed_Test_v3_Leaderboard-v20261021.html" "$CTX/LLM_Speed_Test_v3_Leaderboard.html"
cp "$EXT/llm_test_backend-v20261021.py" "$CTX/llm_test_backend.py"
cp -r "$EXT/deps" "$CTX/deps"
cp "$EXT/manifest-v20261021.json" "$CTX/manifest.json"
cp "$EXT/UPSTREAM.md" "$CTX/UPSTREAM.md"
cp "$REPO/gpustack/Dockerfile.speedtest-v35-v1" "$CTX/Dockerfile"

docker build --pull=false --build-arg PARENT="$PARENT" -f "$CTX/Dockerfile" -t "$IMAGE" "$CTX"

want_html=$(sha256sum "$CTX/LLM_Speed_Test_v3_Python_Backend.html" | cut -d' ' -f1)
got_html=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html | cut -d' ' -f1)
test "$got_html" = "$want_html" || { echo "FAIL: html sha" >&2; exit 7; }

want_bak=$(sha256sum "$CTX/llm_test_backend.py" | cut -d' ' -f1)
got_bak=$(docker run --rm --entrypoint sha256sum "$IMAGE" /opt/sm75-workbench/extensions/speedtest/llm_test_backend.py | cut -d' ' -f1)
test "$got_bak" = "$want_bak" || { echo "FAIL: backend sha" >&2; exit 8; }

docker run --rm --entrypoint python3 "$IMAGE" -c "import sys;sys.path.insert(0,'/opt/sm75-workbench/extensions/speedtest/deps');import h2;print('h2',h2.__version__)" | grep -q '^h2 4.4.1$' || { echo FAIL_h2 >&2; exit 9; }
docker run --rm --entrypoint grep "$IMAGE" -q 'min(1400px, 96vw)' /opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html || { echo FAIL_responsive >&2; exit 10; }
docker run --rm --entrypoint cat "$IMAGE" /opt/sm75-workbench/extensions/speedtest/manifest.json | grep -q '07895cff' || { echo FAIL_manifest >&2; exit 11; }

docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}} SIZE={{.Size}}'
echo "SPEEDTEST_V35_LAYER_OK image=$IMAGE"
