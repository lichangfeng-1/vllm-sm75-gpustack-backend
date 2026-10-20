#!/usr/bin/env bash
# build-backend-v1.sh —— 一条命令从上游 vLLM 基座构建到自定义后端镜像（默认最快档）
#
# 用法：
#   bash gpustack/build-backend-v1.sh                 # 默认：基座复用/缺失时重建，产出钉 5.15.1/0.22.2 的最快档
#   bash gpustack/build-backend-v1.sh --no-tf-pin     # 不钉运行期包（基座原版 5.17.0/0.23.2，回退/对照用）
#   bash gpustack/build-backend-v1.sh --rebuild-base  # 强制重建 SM75 基座（上游 vLLM → SM75）
#   bash gpustack/build-backend-v1.sh --drop-base     # 构建成功后删掉中间基座 tag（省盘，但下次要重建）
#   BASE_TAG=... IMAGE=... MAX_JOBS=2 TF_VERSION=5.15.1 TOKENIZERS_VERSION=0.22.2 bash gpustack/build-backend-v1.sh
#
# 说明：Docker 没有“无中间层”的构建；本脚本把 SM75 基座当作内部构建缓存，
#       用户只需要认末端镜像 vllm-sm75-gpustack:v<VERSION>-v6。
#       基座 tag 默认是 vllm-sm75:v<VERSION>（取 docker/VERSION），可用 BASE_TAG 覆盖。
#       同一枚末端镜像既做 GPUStack 后端（入口 panel-sidecar.py vllm serve），
#       也可单独起面板（入口 runtime-entrypoint.py，见 run-book §11）。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VERSION=$(tr -d ' \r\n' < "$REPO_ROOT/docker/VERSION")
[ -n "$VERSION" ] || { echo "红：docker/VERSION 为空" >&2; exit 2; }
BASE_TAG=${BASE_TAG:-vllm-sm75:v$VERSION}
IMAGE_EXPLICIT=${IMAGE:-}
TF_VERSION=${TF_VERSION-5.15.1}
TOKENIZERS_VERSION=${TOKENIZERS_VERSION-0.22.2}
REBUILD_BASE=0
DROP_BASE=0
TF_PIN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild-base) REBUILD_BASE=1; shift ;;
    --drop-base) DROP_BASE=1; shift ;;
    --no-tf-pin) TF_PIN=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [ -n "$IMAGE_EXPLICIT" ]; then
  IMAGE=$IMAGE_EXPLICIT
elif [ "$TF_PIN" = 1 ]; then
  IMAGE=vllm-sm75-gpustack:v$VERSION-v6
else
  IMAGE=vllm-sm75-gpustack:v$VERSION-v4
fi

command -v docker >/dev/null 2>&1 || { echo "红：PATH 上没有 docker" >&2; exit 2; }
[ -f "$SCRIPT_DIR/build-gpustack-v1.sh" ] || { echo "红：缺少 build-gpustack-v1.sh" >&2; exit 2; }
[ -f "$SCRIPT_DIR/Dockerfile.v1" ] || { echo "红：缺少 Dockerfile.v1" >&2; exit 2; }

if [ "$REBUILD_BASE" = 1 ] || ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
  echo "== 第一段：上游 vLLM → SM75 基座（$BASE_TAG）=="
  # 基座脚本用 IMAGE 作为产物 tag；这里必须显式给它，不能让它继承末端镜像名。
  IMAGE="$BASE_TAG" bash "$SCRIPT_DIR/build-gpustack-v1.sh" --build
else
  echo "== 第一段：复用本地基座 $BASE_TAG（要强制重建加 --rebuild-base）=="
fi

if [ "$TF_PIN" = 1 ]; then
  TARGET=runtime-pin-layer
  PIN_ARGS=(--build-arg "TF_VERSION=$TF_VERSION" --build-arg "TOKENIZERS_VERSION=$TOKENIZERS_VERSION")
  echo "== 第二段：SM75 基座 → 自定义后端（$IMAGE，运行期钉 transformers $TF_VERSION / tokenizers $TOKENIZERS_VERSION）=="
else
  TARGET=panel-layer
  PIN_ARGS=(--build-arg TF_VERSION= --build-arg TOKENIZERS_VERSION=)
  echo "== 第二段：SM75 基座 → 自定义后端（$IMAGE，不钉运行期包）=="
fi

docker build --file "$SCRIPT_DIR/Dockerfile.v1" --target "$TARGET" \
  --build-arg BASE_TAG="$BASE_TAG" "${PIN_ARGS[@]}" \
  -t "$IMAGE" "$REPO_ROOT"

if [ "$DROP_BASE" = 1 ]; then
  echo "== 删除中间基座 tag：$BASE_TAG =="
  docker image rm "$BASE_TAG" >/dev/null 2>&1 || true
fi
echo "BUILD_BACKEND_OK base=$BASE_TAG image=$IMAGE target=$TARGET"
