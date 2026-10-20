#!/usr/bin/env bash
# build-backend-v1.sh —— 一条命令从上游 vLLM 基座构建到自定义后端镜像（默认最快档，单次构建）
#
# 默认（吸收式）：由 gpustack/gen-allinone-dockerfile-v1.py 生成 gpustack/Dockerfile.allinone
#   （上游 docker/Dockerfile 原文＋本仓五层），一次 docker build 从 vllm/vllm-openai 基座走到末端，
#   不产生中间基座 tag。上游原文仍是冻结文件；生成物由仓库掌握，可审计。
#
# 用法：
#   bash gpustack/build-backend-v1.sh                 # 默认：两段构建，产出钉 5.15.1/0.22.2 的 -v7
#   bash gpustack/build-backend-v1.sh --no-tf-pin     # 不钉运行期包（基座原版 5.17.0/0.23.2，产出 -v4）
#   bash gpustack/build-backend-v1.sh --allinone      # 可选：单次 allinone 构建（需网络拉上游件，待完整验收）
#   bash gpustack/build-backend-v1.sh --two-stage     # 默认：两段（先建 vllm-sm75 基座，再叠 Dockerfile.v1；已实跑）
#   bash gpustack/build-backend-v1.sh --rebuild-base  # 仅 --two-stage 下有效：强制重建基座
#   bash gpustack/build-backend-v1.sh --drop-base     # 仅 --two-stage 下有效：成功后删基座 tag
#   BASE_TAG=... IMAGE=... MAX_JOBS=2 TF_VERSION=5.15.1 TOKENIZERS_VERSION=0.22.2 bash gpustack/build-backend-v1.sh
#
# 说明：默认产物 tag vllm-sm75-gpustack:v<VERSION>-v7；同一枚镜像既做 GPUStack 后端
#       （入口 panel-sidecar.py vllm serve），也可单独起面板（入口 runtime-entrypoint.py）。
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
TWO_STAGE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --two-stage) TWO_STAGE=1; shift ;;
    --allinone) TWO_STAGE=0; shift ;;
    --rebuild-base) REBUILD_BASE=1; shift ;;
    --drop-base) DROP_BASE=1; shift ;;
    --no-tf-pin) TF_PIN=0; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [ -n "$IMAGE_EXPLICIT" ]; then
  IMAGE=$IMAGE_EXPLICIT
elif [ "$TF_PIN" = 1 ]; then
  if [ "$TWO_STAGE" = 1 ]; then
    IMAGE=vllm-sm75-gpustack:v$VERSION-v7
  else
    IMAGE=vllm-sm75-gpustack:v$VERSION-v7-allinone
  fi
else
  # 不覆盖已有 -v4（两段构建的回退锚）；不钉版产物用独立 tag。
  IMAGE=vllm-sm75-gpustack:v$VERSION-v4-nopin
fi

command -v docker >/dev/null 2>&1 || { echo "红：PATH 上没有 docker" >&2; exit 2; }

if [ "$TF_PIN" = 1 ]; then
  TARGET=runtime-pin-layer
  PIN_ARGS=(--build-arg "TF_VERSION=$TF_VERSION" --build-arg "TOKENIZERS_VERSION=$TOKENIZERS_VERSION")
else
  TARGET=panel-layer
  PIN_ARGS=(--build-arg TF_VERSION= --build-arg TOKENIZERS_VERSION=)
fi

if [ "$TWO_STAGE" = 1 ]; then
  [ -f "$SCRIPT_DIR/build-gpustack-v1.sh" ] || { echo "红：缺少 build-gpustack-v1.sh" >&2; exit 2; }
  [ -f "$SCRIPT_DIR/Dockerfile.v1" ] || { echo "红：缺少 Dockerfile.v1" >&2; exit 2; }
  if [ "$REBUILD_BASE" = 1 ] || ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
    echo "== 第一段：上游 vLLM → SM75 基座（$BASE_TAG）=="
    IMAGE="$BASE_TAG" bash "$SCRIPT_DIR/build-gpustack-v1.sh" --build
  else
    echo "== 第一段：复用本地基座 $BASE_TAG =="
  fi
  echo "== 第二段：SM75 基座 → 自定义后端（$IMAGE，target=$TARGET）=="
  docker build --file "$SCRIPT_DIR/Dockerfile.v1" --target "$TARGET" \
    --build-arg BASE_TAG="$BASE_TAG" "${PIN_ARGS[@]}" \
    -t "$IMAGE" "$REPO_ROOT"
  if [ "$DROP_BASE" = 1 ]; then
    echo "== 删除中间基座 tag：$BASE_TAG =="
    docker image rm "$BASE_TAG" >/dev/null 2>&1 || true
  fi
else
  echo "== 单次构建：上游 vLLM 基座 → 自定义后端（$IMAGE，target=$TARGET）=="
  python3 "$SCRIPT_DIR/gen-allinone-dockerfile-v1.py"
  docker build --file "$SCRIPT_DIR/Dockerfile.allinone" --target "$TARGET" \
    "${PIN_ARGS[@]}" -t "$IMAGE" "$REPO_ROOT"
fi

echo "BUILD_BACKEND_OK image=$IMAGE target=$TARGET mode=$([ "$TWO_STAGE" = 1 ] && echo two-stage || echo allinone)"
