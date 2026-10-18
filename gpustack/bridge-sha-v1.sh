#!/bin/sh
# bridge-sha-v1.sh —— 两容器速度层指纹桥（历史枚）
# ⚠ 现行枚＝同目录 bridge-sha-v2.sh：本件的容器模式探测缺 docker exec -i，
#   在真容器上会因 stdin 为空而返回空串、假红退 2。留盘不覆盖只作历史。
#
# 做什么：把 stage1 成品容器（GPUStack 线，vllm-sm75-gpustack:v0.1.7-beta2-v1）与
#   bench 测速容器（gpustack/Dockerfile.bench-v1 产物）之间的 10 件层产物
#   （pcieipc 9 件＋incple 目标件 1 件）逐件 sha256 对拍，证明两边是同一层产物；
#   阶段一没有容器，用 --fixture 模式在本地沙箱对两个"site-packages 仿真根"跑同一套判据。
#
# 用法：
#   sh bridge-sha-v1.sh <容器A> <容器B>       容器模式（docker exec 进容器取 sha256sum）
#   sh bridge-sha-v1.sh --fixture <目录A> <目录B>   fixture 模式（目录＝site-packages 仿真根，
#                                             目录内形状为 <dir>/flashinfer/… 与 <dir>/vllm/…）
#
# 退出码（只用已登记码位，本任务不新增）：
#   0   10 件全一致——打印逐件对照表＋BRIDGE_OK 计数行
#   2   用法/前置红——参数形状不对、docker/sha256sum 不在 PATH、site-packages 定位失败、
#       fixture 目录不存在、清单常量自证失败（清单行数≠预期件数，或含重复条目＝唯一件数≠总条数）
#   13  桥差集非空——任一件两边哈希不等（DIFF）或任一边缺件/不可读（MISSING_A/MISSING_B）；
#       打印差集表，每件一行带类型；缺件与改字节是两类行，文案可区分（负控 c 的判据）
#
# 清单出处（与任务 3/4 落盘脚本的安装路径逐字对齐——读脚本取路径，不凭记忆；逐处可核对）：
#   pcieipc 六件：gpustack/layers/pcieipc/install-pcieipc.sh :282 从 manifest.txt 取
#     verbatim 行、:292 dst="$FI/${rel#newfiles/flashinfer/}" 落盘 ⇒ site-packages 相对形＝
#       flashinfer/comm/pcie_ipc_ar.py
#       flashinfer/comm/pcie_ipc_policy.py
#       flashinfer/comm/pcie_ipc_topology.py
#       flashinfer/comm/pcie_ipc_tuning.py
#       flashinfer/data/csrc/pcie_ipc_all_reduce.cu
#       flashinfer/data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh
#   pcieipc 追加三处的目标件：同脚本 :321-323 append_block 的三个 target ⇒
#       flashinfer/jit/comm.py
#       flashinfer/trace/templates/comm.py
#       flashinfer/comm/__init__.py
#   incple 目标件：gpustack/layers/incple/install-incple-v1.sh :101
#     TARGET="$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py" ⇒
#       vllm/models/qwen4_exp/nvidia/ngram_embedding.py
#
# 根解析：flashinfer/ 前缀件挂 flashinfer 的 site-packages 实位，vllm/ 前缀件挂 vllm 实位；
#   容器模式的定位复用 install-pcieipc.sh :173-201 的同一 find_spec 逻辑（不写死路径；
#   两包若分属不同目录也天然正确）。fixture 模式把 <目录> 直接当 site-packages 仿真根，
#   两类前缀件都从同一个目录拼——这是两模式唯一的形状差异，判据（清单、比对、差集）同源。
#
# 设计注记：
#   1) 空判据前先有存在性断言：清单行数先对 BRIDGE_EXPECTED_FILES 自证，继而断言
#      "去重后唯一件数==总条数"（重复键边界——清单含重复条目时总条数不变但实比对件数变少，
#      只看行数的三层断言全会放行，故重复即红 2 并点名重复件）；比对完成后
#      绿面必须断言"逐件表行数==清单件数"才报 BRIDGE_OK（防"零件比对"假绿）。
#   2) 单件取哈希失败（文件不存在/不可读）归 MISSING 类而不是 DIFF 类：容器模式下
#      docker exec sha256sum 非零退出的输出经管道后取到空串，空串即缺件证词；
#      两边哈希都取到且不等才 DIFF。
#   3) 本脚本对容器只读（docker exec sha256sum / find_spec 探测），不写容器任何路径，
#      因此不设落点守卫（守卫义务在写盘的 install-* 脚本侧，已按交接约束第 8 条定形）。
set -eu

# ---------- 清单常量（10 件；顺序与上节出处对应） ----------
BRIDGE_EXPECTED_FILES=10
BRIDGE_FILES="flashinfer/comm/pcie_ipc_ar.py
flashinfer/comm/pcie_ipc_policy.py
flashinfer/comm/pcie_ipc_topology.py
flashinfer/comm/pcie_ipc_tuning.py
flashinfer/data/csrc/pcie_ipc_all_reduce.cu
flashinfer/data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh
flashinfer/jit/comm.py
flashinfer/trace/templates/comm.py
flashinfer/comm/__init__.py
vllm/models/qwen4_exp/nvidia/ngram_embedding.py"

usage() {
  echo "用法: sh bridge-sha-v1.sh <容器A> <容器B>"
  echo "      sh bridge-sha-v1.sh --fixture <目录A> <目录B>"
  echo "退出码: 0 全一致 / 2 用法与前置 / 13 桥差集（DIFF 或 MISSING）"
}

# ---------- 参数形状 ----------
MODE="container"
case "${1:-}" in
  --fixture) MODE="fixture"; shift ;;
  -h|--help) usage; exit 0 ;;
esac
if [ "$#" -ne 2 ]; then
  echo "ASSERT_FAIL 参数个数不对（需要两个比对对象，实到 $# 个）"
  usage
  exit 2
fi
SIDE_A="$1"; SIDE_B="$2"
[ -n "$SIDE_A" ] && [ -n "$SIDE_B" ] || { echo "ASSERT_FAIL 比对对象名/目录为空串"; exit 2; }
if [ "$SIDE_A" = "$SIDE_B" ]; then
  echo "ASSERT_FAIL 两侧同名（$SIDE_A vs $SIDE_B），自己和自己比对没有信息量"
  exit 2
fi

command -v sha256sum >/dev/null 2>&1 || { echo "ASSERT_FAIL 宿主缺少必要工具 sha256sum"; exit 2; }

# ---------- 清单常量自证（存在性断言排在一切比对之前；防清单被改行/吞行/塞重复条目） ----------
manifest_count=$(printf '%s\n' "$BRIDGE_FILES" | sed '/^$/d' | wc -l | tr -dc '0-9')
if [ "$manifest_count" != "$BRIDGE_EXPECTED_FILES" ]; then
  echo "ASSERT_FAIL 清单常量行数自证红：实到 $manifest_count 件，期望 $BRIDGE_EXPECTED_FILES 件（清单被改行/吞行，桥拒绝开工）"
  exit 2
fi
# 重复键边界自证：去重后的唯一件数必须等于总条数——清单含重复条目时总条数不变而实比对件数变少，
# 后面三层行数断言全会放行（假绿洗白），故这里直接红 2 并点名重复件（LC_ALL=C 固定排序域，判据与 locale 无关）
manifest_unique=$(printf '%s\n' "$BRIDGE_FILES" | sed '/^$/d' | LC_ALL=C sort -u | wc -l | tr -dc '0-9')
if [ "$manifest_unique" != "$manifest_count" ]; then
  echo "ASSERT_FAIL 清单重复键自证红：总条数 $manifest_count 条、唯一件只有 $manifest_unique 件（清单含重复条目，实比对会少比 $((manifest_count - manifest_unique)) 件，桥拒绝开工）"
  echo "重复条目清单："
  printf '%s\n' "$BRIDGE_FILES" | sed '/^$/d' | LC_ALL=C sort | uniq -d
  exit 2
fi

# ---------- 根解析与逐件路径拼装 ----------
# 前缀分流：flashinfer/ 前缀件挂 flashinfer 的 site-packages 实位，vllm/ 前缀件挂 vllm 实位；
#   其它前缀＝清单被污染的证词（path_of 里判死）。

FI_SP_A=""; FI_SP_B=""; VL_SP_A=""; VL_SP_B=""

probe_side() { # $1=容器名 → 打印 "<FI根>\t<VL根>"（find_spec 逻辑与 install-pcieipc.sh :173-201 同形）
  raw=$(docker exec "$1" python3 - <<'PY'
import importlib.util, os, sys

def posix(path):
    # 镜像里（Linux）这是恒等变换；与 install-pcieipc.sh 的探测保持同形，
    # 反斜杠形会让 GNU sha256sum 给摘要加 "\" 前缀、污染逐件对拍。
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
  ) || { echo "ASSERT_FAIL 容器 $1 的探测进程非零退出（python3 缺位或异常）"; exit 2; }
  fi_line=$(printf '%s\n' "$raw" | awk -F'\t' '$1=="flashinfer"{ print $2; exit }')
  vl_line=$(printf '%s\n' "$raw" | awk -F'\t' '$1=="vllm"{ print $2; exit }')
  case "$fi_line" in
    OK:*) : ;;
    *) echo "ASSERT_FAIL 容器 $1 定位 flashinfer 失败：[${fi_line:-空}]（桥只在装有层 1 的镜像里工作）"; exit 2 ;;
  esac
  case "$vl_line" in
    OK:*) : ;;
    *) echo "ASSERT_FAIL 容器 $1 定位 vllm 失败：[${vl_line:-空}]（桥只在装有层 2 的镜像里工作）"; exit 2 ;;
  esac
  printf '%s\t%s\n' "${fi_line#OK:}" "${vl_line#OK:}"
}

path_of() { # $1=侧(A|B) $2=rel → 全路径
  _rel=$2
  case "$1" in
    A) _side="$SIDE_A" ;;
    B) _side="$SIDE_B" ;;
  esac
  if [ "$MODE" = "fixture" ]; then
    printf '%s/%s' "${_side%/}" "$_rel"
  else
    case "$_rel" in
      flashinfer/*) rootvar="FI_SP_$1" ;;
      vllm/*) rootvar="VL_SP_$1" ;;
      *) echo "ASSERT_FAIL 清单出现未知前缀件：$_rel"; exit 2 ;;
    esac
    eval "root=\$$rootvar"
    printf '%s/%s' "${root%/}" "$_rel"
  fi
}

# ---------- 逐件取哈希 ----------
hash_A=""; hash_B=""
read_hash() { # $1=侧 $2=rel；结果写全局 hash_$1
  _path=$(path_of "$1" "$2")
  if [ "$MODE" = "fixture" ]; then
    if [ -f "$_path" ]; then
      h=$(sha256sum "$_path" 2>/dev/null | cut -d' ' -f1) || h=""
    else
      h=""
    fi
  else
    # 侧句柄经 eval 取（$SIDE_$1 直写会被解析成 $SIDE_ ＋字面量，是经典坑）
    eval "side=\$SIDE_$1"
    # docker exec 非零（缺件/不可读）经管道后取到空串——空串即 MISSING 证词（设计注记 2）
    h=$(docker exec "$side" sha256sum "$_path" 2>/dev/null | cut -d' ' -f1) || h=""
  fi
  case "$1" in
    A) hash_A="$h" ;;
    B) hash_B="$h" ;;
  esac
}

# ---------- 模式前置 ----------
if [ "$MODE" = "container" ]; then
  command -v docker >/dev/null 2>&1 || { echo "ASSERT_FAIL 容器模式需要 docker，但宿主 PATH 里没有（本地请改用 --fixture 模式）"; exit 2; }
  _pairA=$(probe_side "$SIDE_A"); _pairB=$(probe_side "$SIDE_B")
  FI_SP_A=$(printf '%s' "$_pairA" | cut -f1); VL_SP_A=$(printf '%s' "$_pairA" | cut -f2)
  FI_SP_B=$(printf '%s' "$_pairB" | cut -f1); VL_SP_B=$(printf '%s' "$_pairB" | cut -f2)
  echo "定位完成：A($SIDE_A) fi=$FI_SP_A vl=$VL_SP_A ｜ B($SIDE_B) fi=$FI_SP_B vl=$VL_SP_B"
else
  for _d in "$SIDE_A" "$SIDE_B"; do
    [ -d "$_d" ] || { echo "ASSERT_FAIL fixture 目录不存在：$_d"; exit 2; }
  done
  echo "fixture 模式：A=$SIDE_A B=$SIDE_B（目录＝site-packages 仿真根）"
fi

echo "=== 逐件对拍（清单 $BRIDGE_EXPECTED_FILES 件） ==="
table_lines=""
diff_lines=""
same_count=0
diff_count=0
checked=0
IFS='
'
for rel in $BRIDGE_FILES; do
  [ -n "$rel" ] || continue
  checked=$((checked + 1))
  read_hash A "$rel"
  read_hash B "$rel"
  if [ -z "$hash_A" ] || [ -z "$hash_B" ]; then
    if [ -z "$hash_A" ] && [ -z "$hash_B" ]; then _type="MISSING_A+B"
    elif [ -z "$hash_A" ]; then _type="MISSING_A"
    else _type="MISSING_B"; fi
    diff_lines="$diff_lines$_type	$rel	[${hash_A:-缺}]	[${hash_B:-缺}]
"
    diff_count=$((diff_count + 1))
    table_lines="$table_lines$_type	$rel	${hash_A:--}	${hash_B:--}
"
  elif [ "$hash_A" = "$hash_B" ]; then
    same_count=$((same_count + 1))
    table_lines="$table_lines	SAME	$rel	$hash_A	$hash_B
"
  else
    diff_lines="$diff_lines	DIFF	$rel	$hash_A	$hash_B
"
    diff_count=$((diff_count + 1))
    table_lines="$table_lines	DIFF	$rel	$hash_A	$hash_B
"
  fi
done

# 存在性断言：走到的件数必须等于清单数（循环被吞/清单中途断档都在这里出声，不给空判据留门）
if [ "$checked" != "$BRIDGE_EXPECTED_FILES" ]; then
  echo "ASSERT_FAIL 实际比对件数 $checked != 清单期望 $BRIDGE_EXPECTED_FILES（比对循环断档）"
  exit 2
fi

echo "== 逐件对照表（TYPE | REL | SHA-A | SHA-B）=="
printf '%s' "$table_lines"

if [ "$diff_count" -gt 0 ]; then
  echo "== 桥差集表（$diff_count 件不一致，TYPE：DIFF＝改字节 / MISSING_*＝缺件或不可读）=="
  printf '%s' "$diff_lines"
  echo "BRIDGE_FAIL diff=$diff_count same=$same_count checked=$checked"
  exit 13
fi

# 绿面收口断言：逐件 SAME 行数必须等于清单数（"差集为空"不等于"比对发生过"）
if [ "$same_count" != "$BRIDGE_EXPECTED_FILES" ]; then
  echo "ASSERT_FAIL 绿面收口红：SAME 行数 $same_count != 清单 $BRIDGE_EXPECTED_FILES"
  exit 2
fi
echo "BRIDGE_OK files=$same_count/$BRIDGE_EXPECTED_FILES（两${MODE}侧 $BRIDGE_EXPECTED_FILES 件层产物逐字节一致）"
exit 0
