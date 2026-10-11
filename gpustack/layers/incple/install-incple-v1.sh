#!/bin/sh
# install-incple-v1.sh —— incple 层（AutoRound 量化白名单）装机与构建期断言
# （由 gpustack/Dockerfile.v1 层 2 的 RUN 调用，也可手工在容器里跑）
#
# 用法: sh install-incple-v1.sh [CTX目录]
#   CTX 目录须含 patch_ple_inc.py；默认 /opt/vllm-sm75/incple。
#   环境变量：
#     PLE_INC_TARGET  显式指定目标件（镜像内 site-packages 的 ngram_embedding.py，
#                     本地夹具/负控也走这个口子指 /tmp 沙箱假现场）；缺省时构建期探测。
#     SM75_PY         指定解释器（默认 python3）。
#
# 本层做什么：stage0 装完 docker/next 后，镜像内 site-packages 的
#   vllm/models/qwen4_exp/nvidia/ngram_embedding.py 白名单只认 None/ModelOpt*/Fp8Config；
#   Intel/Qwen3.8-Flash-Next-W4A16-AutoRound 的 quant_config 是 INCConfig，引擎启动即抛
#   NotImplementedError。本脚本用锚点法插一支 INCConfig 分支＋一行 import（不整文件覆盖），
#   并打印改前/改后 sha、做 INCConfig 计数断言——门红即构建红。
#   D-04 注记：补丁只作用 site-packages 安装件；镜像内 /opt/vllm-sm75/source 源码副本
#   保持原样属预期（运行期加载的是前者）。
#
# 落点守卫（抄任务 3 修复轮 3 定形的范式，交接约束第 8 条）：
#   目标件绝对路径先过"`..` 段确定性红"，再按词法形＋`cd && pwd -P` 物理形两条途各自向上
#   遍历仓标记（.git 或 gpustack/SHA256SUMS-upstream.txt），任一命中即退 2 且零写盘——
#   补丁绝不允许写进交付仓（上游树冻结，红码 7 由 gate-tree-unchanged 兜底，这里在源头拦截）。
#   bind-mount/改名 .git 拦不住这两条途的剩余边界由任务 8 的 --out 守卫收口（见交接约束）。
#
# 退出码（本脚本自有；构建期 docker 把非零统一收进 20＝构建失败，日志仍可见本码；
#   注意与任务 3 的 11a 分属不同脚本——11a 在 install-pcieipc.sh 的 import 门，11b 在本链的锚点门）：
#   0   通过（含幂等再跑：改前改后 sha 一致）
#   2   定位/前置失败（CTX 缺补丁件、目标探测失败、目标不存在、`..` 段、落点守卫判死、
#       目标落在镜像源码副本 /opt/vllm-sm75/source）＋ patch 透传的解析失败
#   11  patch_ple_inc.py 锚点缺失（11b；从 patch 原样透传）
#   12  INCConfig 计数断言失败 / 终态 py_compile 失败（本任务登记码位）
set -eu

command -v sha256sum >/dev/null 2>&1 || { echo "ASSERT_FAIL 缺少必要工具 sha256sum"; exit 2; }
command -v grep >/dev/null 2>&1 || { echo "ASSERT_FAIL 缺少必要工具 grep"; exit 2; }

CTX=""
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "用法: sh install-incple-v1.sh [CTX目录]"
      exit 0 ;;
    *)
      if [ -n "$CTX" ]; then
        echo "ASSERT_FAIL 参数里的目录只允许一个，多出来的是：$arg"
        exit 2
      fi
      CTX="$arg" ;;
  esac
done
CTX="${CTX:-/opt/vllm-sm75/incple}"
PYBIN="${SM75_PY:-python3}"

echo "=== 0) 前置物料 ==="
[ -f "$CTX/patch_ple_inc.py" ] || { echo "ASSERT_FAIL 缺 $CTX/patch_ple_inc.py"; exit 2; }
command -v "$PYBIN" >/dev/null 2>&1 || { echo "ASSERT_FAIL 找不到解释器 $PYBIN（可用 SM75_PY 指定）"; exit 2; }

echo "=== 1) 定位目标件（PLE_INC_TARGET 显式覆盖，否则构建期 find_spec 探测 site-packages 实位，不写死路径） ==="
if [ -n "${PLE_INC_TARGET:-}" ]; then
  TARGET="$PLE_INC_TARGET"
  TARGET_SRC="env:PLE_INC_TARGET"
else
  probe_raw="$("$PYBIN" - <<'PY'
import importlib.util, os, sys

def posix(path):
    # 镜像里（Linux）这是恒等变换；Windows 本地干跑时 python 给的是反斜杠形，
    # 后续 sha256sum 与路径守卫都按斜杠形处理，反斜杠会污染比对。
    return path.replace("\\", "/")

try:
    spec = importlib.util.find_spec("vllm")
except Exception as exc:  # noqa: BLE001 - 诊断要的是"为什么定位不了"
    print("ERR:%s" % (type(exc).__name__,))
    sys.exit(0)
if spec is None:
    print("MISS:vllm 不在 sys.path 上")
    sys.exit(0)
origin = getattr(spec, "origin", None)
if origin and origin != "namespace" and os.path.isfile(origin):
    print("OK:" + posix(os.path.dirname(os.path.abspath(origin))))
else:
    hit = ""
    for d in (list(getattr(spec, "submodule_search_locations", None) or [])):
        d = os.path.abspath(d)
        if os.path.basename(d) == "vllm":
            hit = posix(d)
            break
    print(("OK:" + hit) if hit else "NOLOC:既无 origin 也无 search_locations")
PY
)" || { echo "ASSERT_FAIL site-packages 探测进程非零退出（解释器 $PYBIN 异常）"; exit 2; }
  case "$probe_raw" in
    OK:*) VLLM_PKG=${probe_raw#OK:} ;;
    *)    echo "ASSERT_FAIL vllm 定位失败：${probe_raw:-空串}（层 2 只在 vLLM 镜像里装）"; exit 2 ;;
  esac
  TARGET="$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py"
  TARGET_SRC="probe:$PYBIN find_spec(vllm)"
fi
[ -f "$TARGET" ] || { echo "ASSERT_FAIL 目标件不存在：$TARGET（来源=$TARGET_SRC）"; exit 2; }
# 反斜杠→斜杠归一后判源码副本落点（不用 ${x//} 形参数替换：镜像 /bin/sh 可能是 dash，那只是一身 bashism）。
case "$(printf '%s' "$TARGET" | sed 's|\\|/|g')" in
  */vllm-sm75/source/*)
    echo "ASSERT_FAIL 目标落在镜像源码副本 /opt/vllm-sm75/source（D-04：补丁只作用 site-packages 安装件）：$TARGET"
    exit 2 ;;
esac
echo "TARGET=$TARGET（来源=$TARGET_SRC）"

# ---------- 落点守卫（判死排在任何写盘之前；与 install-pcieipc.sh 第 ④ 处同一条判据） ----------
# 绝对化＋反斜杠→斜杠＋驱动器形 X:/… → /x/…：全部是显式 ASCII 变换，不经 cygpath、不吃 locale。
inc_norm_form() {
  _q=$(printf '%s' "$1" | sed -e 's|\\|/|g')     # 先归分隔符与驱动器形，再谈绝对化（顺序错会把 X:/ 当相对段）
  case "$_q" in
    [A-Za-z]:*)
      _d=$(printf '%s\n' "$_q" | cut -c1 | tr 'A-Z' 'a-z')
      _q="/$_d$(printf '%s\n' "$_q" | cut -c3-)" ;;
  esac
  case "$_q" in
    /*) : ;;
    *) _q="$PWD/$_q" ;;
  esac
  _q=${_q%/}
  [ -n "$_q" ] || _q=/
  printf '%s' "$_q"
}

# 含 .. 段的路径无法可靠判定真实落点，直接判红（确定性红，不猜）。
inc_has_dotdot() {
  case "$1" in
    ..|../*|*/..|*/../*) return 0 ;;
  esac
  return 1
}

# 自给定路径逐级向上找仓标记（与 install-pcieipc.sh / gate-evidence-v1.sh 同一标记集、同一收敛口径）。
inc_marker_root() {
  _p=$1
  while :; do
    if [ -e "$_p/.git" ] || [ -f "$_p/gpustack/SHA256SUMS-upstream.txt" ]; then
      printf '%s' "$_p"
      return 0
    fi
    [ "$_p" != "/" ] || break
    _n=${_p%/*}
    [ -n "$_n" ] || _n=/
    [ "$_n" != "$_p" ] || break
    _p=$_n
  done
  return 1
}

# 物理形：目录已在盘上就 cd && pwd -P；不在就退化为词法形（目标件本身不存在时上面已退 2）。
inc_phys_form() {
  _t=$1
  _ph=$(cd -- "$_t" 2>/dev/null && pwd -P 2>/dev/null) || return 1
  [ -n "$_ph" ] || return 1
  printf '%s' "$_ph"
}

# 两条解析途（词法＋物理）各自向上遍历；任一命中仓标记即打印原因并返回 0。
inc_guard_reason=""
inc_check_target() {
  _lex=$1
  _phys=$(inc_phys_form "$_lex") || _phys=""
  for _cand in "$_lex" "$_phys"; do
    [ -n "$_cand" ] || continue
    _hit=$(inc_marker_root "$_cand") || _hit=""
    if [ -n "$_hit" ]; then
      inc_guard_reason="解析途「$_cand」向上遍历命中仓标记，仓根 $_hit"
      return 0
    fi
  done
  return 1
}

echo "=== 2) 落点守卫（目标件不得落在任何仓根及其子树之下） ==="
TARGET_DIR=$(dirname "$TARGET")
T_LEX=$(inc_norm_form "$TARGET_DIR")
[ -n "$T_LEX" ] || { echo "ASSERT_FAIL 目标目录绝对化得到空串：$TARGET_DIR"; exit 2; }
if inc_has_dotdot "$T_LEX"; then
  echo "ASSERT_FAIL 目标路径含 .. 段，无法可靠判定真实落点：$TARGET（请给不含 .. 的绝对路径）"
  exit 2
fi
if inc_check_target "$T_LEX"; then
  echo "ASSERT_FAIL 目标件不得落在任何仓根及其子树之下（$inc_guard_reason；PLE_INC_TARGET=$TARGET）——补丁会写穿交付树，上游冻结红线不允许；请指到镜像内 site-packages 或仓外沙箱"
  exit 2
fi
echo "TARGET_GUARDED $T_LEX（词法＋物理两条向上遍历均无仓标记）"

echo "=== 3) 改前 sha ==="
BEFORE_SHA=$(sha256sum "$TARGET" | cut -d' ' -f1)
[ -n "$BEFORE_SHA" ] || { echo "ASSERT_FAIL 改前 sha 取到空串：$TARGET"; exit 2; }
echo "BEFORE_SHA256=$BEFORE_SHA $TARGET"

echo "=== 4) 锚点法插入（patch_ple_inc.py；缺锚点退 11b，回读契约/编译失败退 12，均原样透传） ==="
set +e
PLE_INC_TARGET="$TARGET" "$PYBIN" "$CTX/patch_ple_inc.py"
patch_rc=$?
set -e
if [ "$patch_rc" != "0" ]; then
  echo "ASSERT_FAIL patch_ple_inc.py 非零退出 rc=$patch_rc（11＝锚点缺失；2＝目标解析；12＝写盘契约）"
  exit "$patch_rc"
fi

echo "=== 5) 改后 sha（幂等再跑时改前改后一致＝没重复插） ==="
AFTER_SHA=$(sha256sum "$TARGET" | cut -d' ' -f1)
[ -n "$AFTER_SHA" ] || { echo "ASSERT_FAIL 改后 sha 取到空串：$TARGET"; exit 2; }
echo "AFTER_SHA256=$AFTER_SHA $TARGET"
if [ "$BEFORE_SHA" = "$AFTER_SHA" ]; then
  echo "IDEMPOTENT_NO_CHANGE 改前改后 sha 一致（已应用形态，幂等不再插入）"
fi

echo "=== 6) INCConfig 计数断言（预期常量：1 处 import＋1 处白名单分支） ==="
EXPECTED_IMPORT_COUNT=1
EXPECTED_WHITELIST_COUNT=1
count_lines() {  # $1=固定串 $2=文件；grep -c 数"匹配行数"，rc=0/1 都合法（0 命中输出 0），rc>=2 是工具/文件级错误
  _rc=0
  _c=$(grep -c -F -- "$1" "$2" 2>/dev/null) || _rc=$?
  if [ "$_rc" -ge 2 ]; then echo "ASSERT_FAIL grep 对 $2 执行异常（rc=$_rc）"; exit 2; fi
  case "$_c" in
    ''|*[!0-9]*) echo "ASSERT_FAIL 计数结果不是非负整数：'$_c'（pattern=$1）"; exit 2 ;;
  esac
  printf '%s' "$_c"
}
imp_count=$(count_lines "from vllm.model_executor.layers.quantization.inc import INCConfig" "$TARGET")
wl_count=$(count_lines "isinstance(quant_config, INCConfig)" "$TARGET")
if [ "$imp_count" != "$EXPECTED_IMPORT_COUNT" ]; then
  echo "ASSERT_FAIL import 行数断言红：期望 $EXPECTED_IMPORT_COUNT 实为 $imp_count（$TARGET）"
  exit 12
fi
if [ "$wl_count" != "$EXPECTED_WHITELIST_COUNT" ]; then
  echo "ASSERT_FAIL 白名单行数断言红：期望 $EXPECTED_WHITELIST_COUNT 实为 $wl_count（$TARGET）"
  exit 12
fi
echo "ASSERT_OK INCConfig import=$imp_count 白名单=$wl_count（均等于预期常量）"

echo "=== 7) 终态语法门（site-packages 里的目标件必须仍可编译） ==="
"$PYBIN" -m py_compile "$TARGET" || { echo "ASSERT_FAIL 终态 py_compile 未过：$TARGET"; exit 12; }
echo "PY_COMPILE_OK $TARGET"

echo "INSTALL_INCPLE_OK target=$TARGET before=$BEFORE_SHA after=$AFTER_SHA"
