#!/bin/sh
# install-pcieipc.sh —— 在镜像里落 T11a backport（由 gpustack/Dockerfile.v1 层 1 的 RUN 调用，也可手工在容器里跑）
#
# 用法: sh install-pcieipc.sh [CTX目录] [--defer-import-gate]
#   默认**硬跑**运行时 import 门禁（退出码 11a）；只有构建机无 GPU、基座镜像里
#   `import flashinfer.comm` 本身不可用时，才允许显式传 --defer-import-gate。此时本脚本打印
#   IMPORT_GATE_SKIPPED 与 DEFERRED_LABEL 两行，Dockerfile 必须同时带
#   LABEL sm75.gpustack.pcieipc-import-gate=deferred（层 1 有 PCIEIPC_DEFER 与 LABEL 的成对校验），
#   并由任务 14 在带 GPU 的容器里补跑同一句判据——不静默放宽。
#   旧部署包的位参写法 skip-import-gate 仍接受，语义等同 --defer-import-gate。
#
# 本仓适配（相对旧部署包 docker/pcieipc/install-pcieipc.sh 的原件，只动三处，其余逐字保留）：
#   ① 目标路径不再写死 /usr/local/lib/python3.12/dist-packages：默认在构建期用
#      python 的 importlib.util.find_spec 探测 flashinfer 与 vllm 的 site-packages 实位。
#      find_spec 只按 sys.path 定位、不执行包内代码，所以无 GPU 构建机上探测不会失败，
#      也不会和运行期 import 门禁的语义纠缠在一起。SM75_SITE_PACKAGES 仍是显式覆盖口
#      （本地干跑/夹具用），给出即照用，但仍校验 flashinfer 树在位。
#   ② import 门禁开关改为 --defer-import-gate；并把本次实跑解析出的路径与门禁形态写进
#      $SM75_EVIDENCE_DIR/pcieipc-install-env.env，供 gate-evidence-v1.sh 出终态证据时
#      直接引用（避免二次探测与安装期结果不一致）。
#   ③ 探测结果统一成斜杠形（posix()）：Linux 镜像里是恒等变换，Windows 本地干跑时 python 给
#      反斜杠形会让 GNU sha256sum 给摘要加 "\" 前缀、污染逐件对拍。
#
# 退出码（本脚本自有；在镜像构建里 docker 会把非零统一收进 20＝构建失败，日志仍可见本码）：
#   0   通过
#   2   基本目录/前置物料缺失（含 site-packages 探测失败、证据目录不可写）
#   3   comm/__init__.py 已含 marker（拒绝二次追加）
#   4   comm/ 下 pcie_ipc_*.py 已存在
#   5   data/ 下 csrc 或 cuh 已存在
#   6   jit/comm.py 已含 gen_pcie_ipc_comm_module
#   7   trace/templates/comm.py 已含 pcie_ipc_all_reduce_trace
#   8   manifest 解析/落盘/sha 对拍异常
#   9   追加块缺失或行数增量不符
#   11a 运行时 import 门禁失败（简报登记的码位）
#
# 复核要点（写之前定的规矩，逐条对应）：
#   1) 幂等门放在最前：任何"已打过"的迹象都直接退出，绝不重复追加（追加型补丁跑两次就是双份代码）。
#   2) 每条否定检查用 if/fi 而不是 `grep && exit`（set -e 与 AND-OR 列表的交互不该赌）。
#   3) 物料先 sha256 对拍再落盘；落盘后由 gate 脚本按"装完的终态"再对拍一次。
#   4) 追加三处各自记账：行数增量必须等于块行数，否则说明追加写歪。
#   5) 任一断言不过 => 非零退出，构建直接失败（不做静默放宽、不留半成品）。
#   6) 定位类失败（探测不到实位）与门禁类失败分开：前者退 2 并打印探测原文，不猜路径。
set -eu

CTX=""
IMPORT_GATE=1
DEFER_REQUESTED=0
for arg in "$@"; do
  case "$arg" in
    --defer-import-gate|skip-import-gate)
      IMPORT_GATE=0
      DEFER_REQUESTED=1 ;;
    -h|--help)
      echo "用法: sh install-pcieipc.sh [CTX目录] [--defer-import-gate]"
      exit 0 ;;
    *)
      if [ -n "$CTX" ]; then
        echo "ASSERT_FAIL 参数里的目录只允许一个，多出来的是：$arg"
        exit 2
      fi
      CTX="$arg" ;;
  esac
done
CTX="${CTX:-/tmp/pcieipc}"
if [ "$DEFER_REQUESTED" = "1" ]; then
  echo "WARN 运行时 import 门禁被显式跳过（--defer-import-gate；应由构建机预检决定，不得默认跳过）"
fi
PYBIN="${SM75_PY:-python3}"
EVIDENCE_DIR="${SM75_EVIDENCE_DIR:-/opt/vllm-sm75/evidence}"

echo "=== 0a) 定位 site-packages 实位（构建期 python 探测，不写死版本路径） ==="
command -v "$PYBIN" >/dev/null 2>&1 || { echo "ASSERT_FAIL 找不到解释器 $PYBIN（可用 SM75_PY 指定）"; exit 2; }
probe_raw="$("$PYBIN" - <<'PY'
import importlib.util, os, sys

def posix(path):
    # 镜像里（Linux）这是恒等变换；Windows 本地干跑时 python 给的是反斜杠形，
    # 而 GNU sha256sum 对含反斜杠的文件名会给摘要加 "\" 前缀，污染逐件对拍。
    return path.replace("\\", "/")

def locate(mod):
    try:
        spec = importlib.util.find_spec(mod)
    except Exception as exc:
        return "ERR:%s" % (type(exc).__name__,)
    if spec is None:
        return "MISS:不在 sys.path 上"
    origin = getattr(spec, "origin", None)
    if origin and origin != "namespace" and os.path.isfile(origin):
        return "OK:" + posix(os.path.dirname(os.path.dirname(os.path.abspath(origin))))
    for d in (list(getattr(spec, "submodule_search_locations", None) or [])):
        d = os.path.abspath(d)
        if os.path.basename(d) == mod:
            return "OK:" + posix(os.path.dirname(d))
        return "OK:" + posix(d)
    return "NOLOC:既无 origin 也无 search_locations"

for mod in ("flashinfer", "vllm"):
    sys.stdout.write("%s\t%s\n" % (mod, locate(mod)))
PY
)" || { echo "ASSERT_FAIL site-packages 探测进程非零退出（解释器 $PYBIN 异常）"; exit 2; }
probe_fi=$(printf '%s\n' "$probe_raw" | awk -F'\t' '$1=="flashinfer"{ print $2; exit }')
probe_vl=$(printf '%s\n' "$probe_raw" | awk -F'\t' '$1=="vllm"{ print $2; exit }')
case "$probe_fi" in
  OK:*) probe_fi_sp=${probe_fi#OK:} ;;
  *)    echo "ASSERT_FAIL flashinfer 定位失败：${probe_fi:-空串}"; printf '%s\n' "$probe_raw" | sed 's|^|    探测原文 |'; exit 2 ;;
esac
case "$probe_vl" in
  OK:*) VLLM_SP=${probe_vl#OK:} ;;
  *)    VLLM_SP="" ;;
esac

if [ -n "${SM75_SITE_PACKAGES:-}" ]; then
  SP="$SM75_SITE_PACKAGES"
  SP_SRC="env:SM75_SITE_PACKAGES"
  [ "$SP" = "$probe_fi_sp" ] || echo "NOTE 显式覆盖的 SM75_SITE_PACKAGES=$SP 与探测实位 $probe_fi_sp 不同，按覆盖值安装"
else
  SP="$probe_fi_sp"
  SP_SRC="probe:$PYBIN"
  # 非覆盖路径＝真实构建期：vllm 必须能定位，它是“这是 vLLM 镜像”的最低证词。
  [ -n "$VLLM_SP" ] || { echo "ASSERT_FAIL vllm 定位失败：${probe_vl:-空串}（层 1 只在 vLLM 镜像里装）"; printf '%s\n' "$probe_raw" | sed 's|^|    探测原文 |'; exit 2; }
fi
FI="$SP/flashinfer"
[ -n "$SP" ] || { echo "ASSERT_FAIL site-packages 定位得到空串"; exit 2; }
[ -n "$FI" ] || { echo "ASSERT_FAIL flashinfer 树路径拼接得到空串"; exit 2; }
export SM75_SITE_PACKAGES="$SP"
echo "SITE_PACKAGES=$SP（来源=$SP_SRC）"
echo "VLLM_SITE_PACKAGES=${VLLM_SP:-未定位}"
mkdir -p "$EVIDENCE_DIR" 2>/dev/null || { echo "ASSERT_FAIL 证据目录不可创建：$EVIDENCE_DIR"; exit 2; }
[ -d "$EVIDENCE_DIR" ] || { echo "ASSERT_FAIL 证据目录不存在：$EVIDENCE_DIR"; exit 2; }
[ -w "$EVIDENCE_DIR" ] || { echo "ASSERT_FAIL 证据目录不可写：$EVIDENCE_DIR"; exit 2; }
echo "EVIDENCE_DIR=$EVIDENCE_DIR"

echo "=== 0b) 基本目录 ==="
[ -d "$FI" ] || { echo "ASSERT_FAIL 没有 $FI"; exit 2; }
[ -f "$CTX/manifest.txt" ] || { echo "ASSERT_FAIL 缺 $CTX/manifest.txt"; exit 2; }
[ -f "$CTX/obligations.txt" ] || { echo "ASSERT_FAIL 缺 $CTX/obligations.txt"; exit 2; }
[ -f "$CTX/gate-check-v3.py" ] || { echo "ASSERT_FAIL 缺 $CTX/gate-check-v3.py"; exit 2; }
echo "FLASHINFER_TREE=$FI"
"$PYBIN" -V

echo "=== 1) 幂等 / 前置门 ==="
# 用"精确 marker"而不是宽松的 pcie_ipc 子串：宽松串可能命中 0.6.18 里无关文本，把干净构建误判成"打过两次"。
if grep -q "from .pcie_ipc_ar import" "$FI/comm/__init__.py"; then
  echo "ASSERT_FAIL comm/__init__.py 已含 'from .pcie_ipc_ar import'，拒绝重复追加"; exit 3
fi
echo "  参考信息：整棵树里含 pcie_ipc 的文件数=$(grep -rl pcie_ipc "$FI" 2>/dev/null | wc -l | tr -dc '0-9')（0.6.18 应为 0）"
for f in pcie_ipc_ar.py pcie_ipc_policy.py pcie_ipc_topology.py pcie_ipc_tuning.py; do
  if [ -e "$FI/comm/$f" ]; then echo "ASSERT_FAIL 已存在 $FI/comm/$f"; exit 4; fi
done
if [ -e "$FI/data/csrc/pcie_ipc_all_reduce.cu" ]; then echo "ASSERT_FAIL 已存在 csrc/pcie_ipc_all_reduce.cu"; exit 5; fi
if [ -e "$FI/data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh" ]; then
  echo "ASSERT_FAIL 已存在 include/flashinfer/comm/pcie_ipc_all_reduce.cuh"; exit 5
fi
if grep -q "gen_pcie_ipc_comm_module" "$FI/jit/comm.py"; then
  echo "ASSERT_FAIL jit/comm.py 已有 gen_pcie_ipc_comm_module"; exit 6
fi
if grep -q "pcie_ipc_all_reduce_trace" "$FI/trace/templates/comm.py"; then
  echo "ASSERT_FAIL trace/templates/comm.py 已有 pcie_ipc_all_reduce_trace"; exit 7
fi
echo "IDEMPOTENCE_OK 六个目标文件均不存在、三处追加点均未打过"

echo "=== 2) 安装前门禁（0.6.18 是否喂饱待落文件的 import） ==="
"$PYBIN" "$CTX/gate-check-v3.py" "$CTX" pre

echo "=== 3) 落六个逐字节文件（先对拍 sha256） ==="
awk -F'\t' '$5=="verbatim"{print $1"\t"$4}' "$CTX/manifest.txt" > /tmp/pcieipc.verbatim
n=0
while IFS="$(printf '\t')" read -r rel want; do
  case "$rel" in newfiles/*) : ;; *) echo "ASSERT_FAIL manifest 路径异常 $rel"; exit 8 ;; esac
  src="$CTX/$rel"
  [ -f "$src" ] || { echo "ASSERT_FAIL 物料缺失 $src"; exit 8; }
  got=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "ASSERT_FAIL sha 不符 $rel 期望 $want 实为 $got"; exit 8
  fi
  dst="$FI/${rel#newfiles/flashinfer/}"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  got2=$(sha256sum "$dst" | cut -d' ' -f1)
  [ "$got2" = "$want" ] || { echo "ASSERT_FAIL 落盘后 sha 变了 $dst"; exit 8; }
  n=$((n + 1))
  echo "  PLACED $dst"
done < /tmp/pcieipc.verbatim
[ "$n" = "6" ] || { echo "ASSERT_FAIL 落盘文件数=$n，期望 6"; exit 8; }
echo "PLACED_COUNT=$n"

echo "=== 4) 三处追加（逐处记账行数增量） ==="
append_block() {
  target="$1"; block="$2"; label="$3"
  [ -f "$block" ] || { echo "ASSERT_FAIL 缺追加块 $block"; exit 9; }
  before=$(wc -l < "$target" | tr -dc '0-9')
  blk=$(wc -l < "$block" | tr -dc '0-9')
  cat "$block" >> "$target"
  after=$(wc -l < "$target" | tr -dc '0-9')
  delta=$((after - before))
  # 块末行没有换行符时 cat >> 会让块尾与目标原有内容并到同一行 ⇒ wc -l 少计 1。
  # 包内三个块都以换行结尾（delta 恒等于 blk），但万一物料换了形状，这里要出声、不要假失败也不要静默放宽。
  if [ "$(tail -c 1 "$block" | wc -l | tr -dc '0-9')" = "0" ]; then allow=$((blk - 1)); else allow="$blk"; fi
  if [ "$delta" != "$blk" ] && [ "$delta" != "$allow" ]; then
    echo "ASSERT_FAIL $label 行数增量 $delta != 块行数 $blk（允许下限 $allow，追加写歪）"; exit 9
  fi
  [ "$delta" = "$blk" ] || echo "  NOTE $label 块末行无换行 ⇒ 增量 $delta 比块行数 $blk 少 1（按合法形状放行）"
  echo "  APPEND $label before=$before blk=$blk after=$after delta=$delta"
}
append_block "$FI/jit/comm.py"                "$CTX/append/jit_comm.txt"   "jit/comm.py"
append_block "$FI/trace/templates/comm.py"    "$CTX/append/trace_comm.txt" "trace/templates/comm.py"
append_block "$FI/comm/__init__.py"           "$CTX/append/comm_init.txt"  "comm/__init__.py"

echo "=== 5) 语法必须仍可编译 ==="
"$PYBIN" -m py_compile \
  "$FI/comm/pcie_ipc_ar.py" "$FI/comm/pcie_ipc_policy.py" \
  "$FI/comm/pcie_ipc_topology.py" "$FI/comm/pcie_ipc_tuning.py" \
  "$FI/jit/comm.py" "$FI/trace/templates/comm.py" "$FI/comm/__init__.py"
echo "PY_COMPILE_OK 7 个文件"

echo "=== 6) 安装后门禁（终态对拍） ==="
"$PYBIN" "$CTX/gate-check-v3.py" "$CTX" post

echo "=== 7) 运行时门禁（vLLM 判据就是这一句 hasattr） ==="
if [ "$IMPORT_GATE" = "1" ]; then
  if "$PYBIN" - <<'PY'
import sys
import flashinfer.comm as c
ok = hasattr(c, "PcieIpcAllReduceWorkspace")
print("  hasattr(flashinfer.comm,'PcieIpcAllReduceWorkspace') =", ok)
if not ok:
    sys.exit(11)
import flashinfer.comm.pcie_ipc_ar as ar
print("  pcie_ipc_ar 已导入，类 =", ar.PcieIpcAllReduceWorkspace.__name__)
import flashinfer.trace.templates.comm as tc
print("  trace 模板已导入，符号 =", type(tc.pcie_ipc_all_reduce_trace).__name__)
from flashinfer.jit.comm import gen_pcie_ipc_comm_module as g
print("  jit 生成器可解析 =", callable(g))
PY
then
  echo "IMPORT_GATE_OK"
else
  echo "ASSERT_FAIL 运行时 import 门禁未过（看上方 traceback 或 hasattr=False）"
  exit 11
fi
else
  echo "IMPORT_GATE_SKIPPED 由验收步骤在带 GPU 的容器里补跑（任务 14 的探针见交付报告门禁项）"
  echo "DEFERRED_LABEL sm75.gpustack.pcieipc-import-gate=deferred"
fi

echo "=== 8) 交接给终态证据脚本（本轮实跑解析出的路径与门禁形态） ==="
ENV_FILE="$EVIDENCE_DIR/pcieipc-install-env.env"
ENV_TMP="$ENV_FILE.tmp.$$"
IMPORT_GATE_STATE=hard
[ "$IMPORT_GATE" = "1" ] || IMPORT_GATE_STATE=deferred
{
  printf 'SM75_SITE_PACKAGES=%s\n' "$SP"
  printf 'SM75_SITE_PACKAGES_SOURCE=%s\n' "$SP_SRC"
  printf 'SM75_VLLM_SITE_PACKAGES=%s\n' "$VLLM_SP"
  printf 'SM75_PCIEIPC_IMPORT_GATE=%s\n' "$IMPORT_GATE_STATE"
  printf 'SM75_PCIEIPC_CTX=%s\n' "$CTX"
  printf 'SM75_PCIEIPC_PYBIN=%s\n' "$PYBIN"
  printf 'SM75_PCIEIPC_PY_VERSION=%s\n' "$("$PYBIN" -V 2>&1)"
} > "$ENV_TMP" || { echo "ASSERT_FAIL 证据交接文件写入失败：$ENV_TMP"; rm -f "$ENV_TMP"; exit 2; }
[ -s "$ENV_TMP" ] || { echo "ASSERT_FAIL 证据交接文件是空的：$ENV_TMP"; rm -f "$ENV_TMP"; exit 2; }
mv -f "$ENV_TMP" "$ENV_FILE" || { echo "ASSERT_FAIL 证据交接文件改名失败：$ENV_FILE"; rm -f "$ENV_TMP"; exit 2; }
echo "EVIDENCE_HANDOFF $ENV_FILE"

echo "INSTALL_PCIEIPC_OK"
