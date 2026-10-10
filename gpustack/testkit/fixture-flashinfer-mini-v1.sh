#!/usr/bin/env bash
# fixture-flashinfer-mini-v1.sh —— 本地干跑与负控用的最小 flashinfer 0.6.18 形状夹具
#
# 做什么：在给定目录下搭一棵 site-packages/flashinfer 最小树，逐个提供 obligations.txt
#         要求的顶层名字，并在同一 site-packages 下搭一个 vllm 占位包（供 install 的定位探测用）。
# 为什么：层 1 的装机链（install → gate-check → evidence）判的是"装完的终态"，本地既无 GPU
#         也无镜像，只能靠一棵形状正确的假树把整条链跑通；真树核验属阶段二任务 14。
# 预期输出：末行 FIXTURE_OK root=<目录>，退出码 0；随后 install/gate/evidence 三条命令都能吃这棵树。
# 失败怎么办：非零即夹具形状不完整（缺模块或缺名字），按脚本点名的路径补；不得反过来放宽门禁判据。
#
# 用法：
#   bash fixture-flashinfer-mini-v1.sh <site-packages目录>
# 之后：
#   PYTHONPATH=<site-packages目录> 让 python 的 find_spec 找到 flashinfer 与 vllm。
#
# 退出码：0 通过 / 2 用法或写盘失败（本夹具不参与镜像构建，不占交付线保留码位）。
#
# 注意：这是测试脚手架，不是镜像内容——Dockerfile.v1 只 COPY gpustack/layers/pcieipc，
#       本文件与 testkit 下其余件都不会进镜像。

set -uo pipefail
export LC_ALL=C

usage() {
  echo "用法：bash fixture-flashinfer-mini-v1.sh <site-packages目录>"
  echo "说明：搭一棵最小 flashinfer 0.6.18 形状树＋同目录 vllm 占位包，供层 1 装机链本地干跑。"
  echo "约束：目录必须落在仓外沙箱（本脚本会在里面写盘，仓内一律拒绝）。"
}

if [ $# -eq 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  usage
  exit 0
fi
if [ $# -ne 1 ]; then
  echo "错误：本脚本只收一个沙箱目录参数（本次得到 $# 个）。" >&2
  usage >&2
  exit 2
fi
case "$1" in
  -*) echo "错误：参数「$1」看着像选项，本脚本只收一个沙箱目录（无选项）。" >&2; exit 2 ;;
esac
[ -n "$1" ] || { echo "错误：目录参数是空串。" >&2; exit 2; }

# 夹具会往里写文件，落进仓里就等于往交付面里塞测试垃圾（还会被上游护栏按新增报红）。
# 判死要排在任何 mkdir 之前：先建后判会把空目录留在仓里。
command -v realpath >/dev/null 2>&1 || { echo "错误：缺少必要工具 realpath，无法判定目录归属。" >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# realpath -m 会保留输入里的盘符形态（K:/… 与 /k/… 两种写法互不相等），
# 所以两侧都再过一道 cygpath -u（Linux 上没有这个工具，即恒等）。
canon() {
  local p=$1 c=""
  if command -v cygpath >/dev/null 2>&1; then
    c=$(cygpath -u -- "$p" 2>/dev/null)
    [ -n "$c" ] && p=$c
  fi
  printf '%s' "${p%/}"
}
REQ_ABS=$(canon "$(realpath -m -- "$1")")
[ -n "$REQ_ABS" ] || { echo "错误：目录参数绝对化得到空串：$1" >&2; exit 2; }
REPO_ABS=$(canon "$(realpath -m -- "$REPO_ROOT")")
[ -n "$REPO_ABS" ] || { echo "错误：仓根绝对化得到空串，无法判定目录归属。" >&2; exit 2; }
case "$REQ_ABS" in
  "$REPO_ABS"|"$REPO_ABS"/*)
    echo "错误：夹具目录不得在仓内（$REQ_ABS 落在 $REPO_ABS 之下），请指到沙箱目录。" >&2
    exit 2 ;;
esac

SP=$(cd -- "$1" 2>/dev/null && pwd)
if [ -z "$SP" ]; then
  mkdir -p -- "$1" || { echo "错误：无法创建 $1" >&2; exit 2; }
  SP=$(cd -- "$1" && pwd) || { echo "错误：目录创建后仍进不去：$1" >&2; exit 2; }
fi
[ -n "$SP" ] || { echo "错误：目录绝对化得到空串：$1" >&2; exit 2; }
[ -d "$SP" ] || { echo "错误：目录不存在：$SP" >&2; exit 2; }

FI="$SP/flashinfer"
for d in "$FI" "$FI/comm" "$FI/jit" "$FI/autotuner" "$FI/trace" "$FI/trace/templates" "$FI/data/csrc" "$SP/vllm"; do
  mkdir -p -- "$d" || { echo "错误：无法创建 $d" >&2; exit 2; }
done
# 安装会往 data/csrc 与 data/include 落两件，这里只保证 data 在位，include 由装机脚本自己建。

write() {
  local path=$1 content=$2
  printf '%s\n' "$content" > "$path" || { echo "错误：写不出 $path" >&2; exit 2; }
}

# 顶层名字与 obligations.txt 逐条对齐（引用方 → 目标模块 → 名字）。
write "$FI/__init__.py" '__version__ = "0.6.18"'
write "$FI/api_logging.py" 'flashinfer_api = None'
write "$FI/autotuner/__init__.py" 'class AutoTuner:
    pass


class DynamicTensorSpec:
    pass


class TunableRunner:
    pass


class TuningConfig:
    pass


def autotune(*args, **kwargs):
    return None


def get_autotune_process_group():
    return None


def set_autotune_process_group(*args, **kwargs):
    return None


def make_bucket_mapper(*args, **kwargs):
    return None'
write "$FI/comm/__init__.py" '# 干净树：这里没有任何 PCIe 回填相关的导入，三处追加块落进来才是打过补丁的终态。
'
write "$FI/comm/cuda_ipc.py" 'def create_shared_buffer(*args, **kwargs):
    return None


def free_shared_buffer(*args, **kwargs):
    return None'
write "$FI/jit/comm.py" 'class JitSpec:
    pass


def gen_jit_spec(*args, **kwargs):
    return JitSpec()'
write "$FI/jit/env.py" 'import os

FLASHINFER_WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_workspace")'
write "$FI/jit/core.py" 'def run(*args, **kwargs):
    return None'
write "$FI/trace/__init__.py" '__all__ = []'
write "$FI/trace/templates/__init__.py" '__all__ = []'
write "$FI/trace/templates/comm.py" 'class TraceTemplate:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def allreduce_fusion(*args, **kwargs):
    return None'
write "$FI/utils.py" 'def register_custom_op(*args, **kwargs):
    return None'
write "$FI/version.py" '__version__ = "0.6.18"'
write "$SP/vllm/__init__.py" '__version__ = "0.30.0"
'

[ -f "$FI/comm/__init__.py" ] || { echo "错误：夹具的 comm/__init__.py 没落地" >&2; exit 2; }
if grep -q "pcie_ipc" "$FI/comm/__init__.py"; then
  echo "错误：夹具的 comm/__init__.py 已含 pcie_ipc marker，不是干净树" >&2
  exit 2
fi
echo "FIXTURE_OK root=$SP flashinfer=$FI vllm=$SP/vllm"
exit 0
