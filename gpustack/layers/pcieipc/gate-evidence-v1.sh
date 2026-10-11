#!/usr/bin/env bash
# gate-evidence-v1.sh —— pcieipc 层（镜像层 1）终态门禁 ＋ 镜像内证据 JSON
#
# 做什么：安装完成后按"终态"再跑一次 gate-check-v3.py post，并把这次判定的全部依据写成
#         镜像内证据文件（默认 /opt/vllm-sm75/evidence/pcieipc-gate.json）——
#         含 gate-check 的 rc 与原文、六件逐字节文件的"清单期望 sha ↔ 装完实 sha"、
#         三处追加块的尾部一致性、安装期解析出的 site-packages 实位与 import 门禁形态。
# 为什么：交付接口要求镜像内留一份能自证的文件；验收（任务 14/16）只认产物，不认"我跑过命令"。
#         证据引用安装期写下的交接 env（不重新探测），保证 JSON 里的路径就是真装进去的那棵。
# 预期输出：末行 GATE_EVIDENCE_OK out=<证据路径>，退出码 0。
# 失败怎么办：
#   - gate-check 红 → 仍写出 gate_pass=false 的证据，然后按 gate-check 自有红码退出（实测 2）。
#   - 前置件缺失（安装期交接 env 不存在、ctx 清单缺失、探测到的树不存在）→ 退 2 并点名缺什么。
#   - 落点守卫（修复轮 3，两条解析途）：对**脚本目录**与**证据落点目录**各做两条向上遍历——
#     词法途（`cd && pwd`）与物理途（`cd && pwd -P`，会穿过符号链接/目录联接落到真实目标），
#     路径先经大小写/盘符归一（不存在的路径走与 locale 无关的显式字节级归一）再比对；
#     标记集＝.git 或 gpustack/SHA256SUMS-upstream.txt。落点侧任一条途命中仓标记 ⇒ 无条件退 2
#     且零写盘（不看脚本侧；锚点只作记录进 guard 字段，不参与判定）；脚本侧命中且落点（词法形或
#     物理形）在其仓根之下也退 2。四条途到 / 都无标记＝镜像形位，守卫自动跳过并打印 NOTE——
#     NOTE 只在两条途都跑过且都无标记时才说"落点在仓外"，与实判逐字一致。
#     守卫判定结果（mode/skipped/两条途的仓根/落点的词法形与物理形/解析途计数）写进证据 JSON 的
#     guard 字段，事后可审计。
#     拦的是误写（防呆）不是恶意：本地把 .git 改名即可绕过；容器内看不见宿主的 .git，容器档的
#     落点约束另由任务 8 简报补"禁指挂载点"口径。
#   - 不得手工伪造证据文件绕过本步：任务 6 的指纹桥会再比一次镜像内实 sha。
#
# 用法：
#   bash gate-evidence-v1.sh [--ctx DIR] [--out FILE] [--env-file FILE] [--python BIN]
#   默认：ctx＝本脚本所在目录；env-file＝$SM75_EVIDENCE_DIR/pcieipc-install-env.env；
#         out＝$SM75_EVIDENCE_DIR/pcieipc-gate.json；python＝$SM75_PY（缺省 python3）。
#
# 退出码：
#   0   门禁绿且证据已写
#   2   前置输入不可用（本脚本自己判死的用法/缺件类红）
#   其余＝gate-check-v3.py 的自有红码原样透传（该脚本所有 GATE_FAIL 都走 sys.exit(2)，
#         故本档 gate-check 红与前置红同码，靠文案与 JSON 里的 gate_check.rc 区分）

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
CTX=$SCRIPT_DIR
EVIDENCE_DIR=${SM75_EVIDENCE_DIR:-/opt/vllm-sm75/evidence}
OUT_FILE="$EVIDENCE_DIR/pcieipc-gate.json"
ENV_FILE=""
PYBIN=${SM75_PY:-python3}

usage() {
  echo "用法：bash gate-evidence-v1.sh [--ctx DIR] [--out FILE] [--env-file FILE] [--python BIN]"
  echo "说明：跑 gate-check post 并产出镜像内证据 JSON；证据默认落在 /opt/vllm-sm75/evidence/pcieipc-gate.json。"
}

bad_usage() {
  echo "错误：$*" >&2
  usage >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ctx)
      [ $# -ge 2 ] || bad_usage "--ctx 缺少参数值。"
      CTX=$(cd -- "$2" 2>/dev/null && pwd) || bad_usage "ctx 目录不存在或不可进入：$2"
      [ -n "$CTX" ] || bad_usage "ctx 目录绝对化得到空串：$2"
      shift 2 ;;
    --out)
      [ $# -ge 2 ] || bad_usage "--out 缺少参数值。"
      [ -n "$2" ] || bad_usage "--out 的参数值是空串。"
      OUT_FILE=$2; shift 2 ;;
    --env-file)
      [ $# -ge 2 ] || bad_usage "--env-file 缺少参数值。"
      [ -n "$2" ] || bad_usage "--env-file 的参数值是空串。"
      ENV_FILE=$2; shift 2 ;;
    --python)
      [ $# -ge 2 ] || bad_usage "--python 缺少参数值。"
      [ -n "$2" ] || bad_usage "--python 的参数值是空串。"
      PYBIN=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) bad_usage "未识别的参数「$1」。" ;;
  esac
done

[ -n "$CTX" ] || bad_usage "ctx 目录绝对化得到空串。"
[ -d "$CTX" ] || bad_usage "ctx 目录不存在：$CTX"
[ -n "$OUT_FILE" ] || bad_usage "证据输出路径为空串。"
[ -n "$PYBIN" ] || bad_usage "解释器名为空串。"
command -v "$PYBIN" >/dev/null 2>&1 || bad_usage "找不到解释器「$PYBIN」（可用 --python 或 SM75_PY 指定）"
for needed_tool in awk grep sed mktemp rm wc mkdir tr; do
  command -v "$needed_tool" >/dev/null 2>&1 || bad_usage "缺少必要工具「$needed_tool」，拒绝出证据。"
done

# ---------- 存在性断言（先行；空判据不许裸跑） ----------
[ -f "$CTX/manifest.txt" ]    || bad_usage "缺 $CTX/manifest.txt（证据要逐件记 sha，没清单就没判据）。"
[ -s "$CTX/manifest.txt" ]    || bad_usage "manifest.txt 是 0 字节空文件，不予出证据。"
[ -f "$CTX/obligations.txt" ] || bad_usage "缺 $CTX/obligations.txt（gate-check 的义务表判据源）。"
[ -s "$CTX/obligations.txt" ] || bad_usage "obligations.txt 是空表，gate-check 会把它当空门禁，不予出证据。"
[ -f "$CTX/gate-check-v3.py" ] || bad_usage "缺 $CTX/gate-check-v3.py。"
[ -d "$CTX/newfiles" ]        || bad_usage "缺 $CTX/newfiles 目录（证据要比对物料 sha）。"
[ -d "$CTX/append" ]          || bad_usage "缺 $CTX/append 目录（证据要比对追加块尾部）。"

[ -n "$ENV_FILE" ] || ENV_FILE="$EVIDENCE_DIR/pcieipc-install-env.env"
[ -f "$ENV_FILE" ] || bad_usage "缺安装期交接文件 $ENV_FILE —— 证据必须引用安装期解析出的实位，不接受现场重探。"
[ -r "$ENV_FILE" ] || bad_usage "安装期交接文件不可读：$ENV_FILE"
[ -s "$ENV_FILE" ] || bad_usage "安装期交接文件为空：$ENV_FILE"

# KEY=VALUE 逐行读，不 source（避免把数据文件当代码执行）；只认这步要用的键。
read_env() {
  local key=$1 line value
  line=$(awk -F'=' -v k="$key" '$1==k { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE")
  value=$line
  printf '%s' "$value"
}

SP=$(read_env SM75_SITE_PACKAGES)
[ -n "$SP" ] || bad_usage "交接文件里 SM75_SITE_PACKAGES 为空：$ENV_FILE"
FI="$SP/flashinfer"
[ -d "$FI" ] || bad_usage "交接文件指向的 flashinfer 树不存在：$FI"
VLLM_SP=$(read_env SM75_VLLM_SITE_PACKAGES)
SP_SRC=$(read_env SM75_SITE_PACKAGES_SOURCE)
IMPORT_GATE_STATE=$(read_env SM75_PCIEIPC_IMPORT_GATE)
case "$IMPORT_GATE_STATE" in
  hard|deferred) ;;
  *) bad_usage "交接文件里 SM75_PCIEIPC_IMPORT_GATE=「$IMPORT_GATE_STATE」，只接受 hard 或 deferred。" ;;
esac
INSTALL_CTX=$(read_env SM75_PCIEIPC_CTX)
PY_VERSION=$(read_env SM75_PCIEIPC_PY_VERSION)
export SM75_SITE_PACKAGES="$SP"

# 落点守卫（修复轮 3：复审轮 2 的新-1 与遗留-1）——轮 2 的守卫已经"看向落点本身"，但看的是落点的
# **名字**不是落点：判定用 `cd && pwd` 的词法路径，`.git`/清单标记只在这条字符串的祖先链上找。
# 符号链接/目录联接把物理落点换进仓内而词法路径不变 ⇒ 两侧皆"无标记" ⇒ 绿，且 NOTE 谎称"落点在仓外"
# （实测 rc=0 真把证据 JSON 写进仓）。本轮补第二条解析途：同一目录再走一遍 `cd && pwd -P`（物理形，
# 穿过链接点），四条遍历（脚本位×2＋落点位×2）任一命中即判死，NOTE 文案与实判逐字一致。
# 历史三轮病灶（轮 1 只看脚本侧／轮 2 词法判定／轮 0 相对锚点）均已封堵，判死仍排在任何写盘之前、
# 红码仍是 bad_usage 的 2，不新占码位。
# canon() 与 gpustack/testkit 夹具的 canon() 同形同口径（本轮两处一起把"不存在路径"那一支改成与
# locale 无关的显式字节级归一，见 canon() 内注释；复审 Minor-1 的残余就此关闭）。

# 不依赖 locale 的显式字节级归一：只做 ASCII 变换（反斜杠→斜杠、驱动器形 X:/ → /x/），
# 非 ASCII 字节原样透传——用于"路径不存在、无法用盘上真实性验真"那一支。
msys_bytes_form() {
  local p=$1
  p=$(printf '%s' "$p" | sed -e 's|\\|/|g')     # 反斜杠→斜杠（只动 ASCII 字节；写法已用 bs-test 验真）
  case "$p" in
    [A-Za-z]:*) p="/${p:0:1}${p:2}" ;;         # 驱动器形 X:/… → /X/…（盘符大小写随后由 tr 折叠）
  esac
  printf '%s' "$p"
}

canon() {
  local p=$1 c=""
  if command -v cygpath >/dev/null 2>&1; then
    if [ -e "$p" ]; then
      # 存在的路径：LC_ALL=C 下 cygpath 会按 ANSI 码页吃参数，把 UTF-8 非 ASCII 路径折成乱码，故在
      # UTF-8 locale 下跑；采纳条件是"结果非空且结果路径真实存在"——折坏的串不在盘上，不会被采纳。
      c=$(LC_ALL=C.UTF-8 cygpath -u -- "$p" 2>/dev/null) \
        || c=$(LC_ALL=en_US.UTF-8 cygpath -u -- "$p" 2>/dev/null)
      if [ -n "$c" ] && [ -e "$c" ]; then p=$c; fi
    else
      # 不存在的路径（证据文件名就是典型输入，NC-85 的探漢字-probe.json 同形）：轮 2 在这里"结果非空
      # 即采纳"，安全性单点押在 C.UTF-8 生效上——某台机器的 MSYS 不认 C.UTF-8/en_US.UTF-8 而 cygpath
      # 仍 rc=0 时，输出是双重编码的乱码串，且没有任何盘上真实性可验（复审 Minor-1）：非 ASCII 仓路径
      # 会被静默判成"无标记"＝反向漏拦。改走 msys_bytes_form 的显式字节级归一，locale 不再参与判定。
      p=$(msys_bytes_form "$p")
    fi
    # Windows 文件系统大小写不敏感 → 归小写，堵 /TMP 与 /tmp、短名形态的漏拦；
    # Linux 文件系统大小写敏感，绝不可折叠（cygpath 只在 Git Bash/MSYS 上存在）。
    # 折叠必须用 tr 逐字节只动 ASCII A-Z：本脚本 LC_ALL=C，bash 的 ${p,,} 会把 UTF-8 多字节
    # 当 Latin-1 折叠出坏字节（实测把"使用/交付仓"折成乱码，-e 标记测试随即失效＝反向漏拦）。
    p=$(printf '%s' "$p" | tr 'A-Z' 'a-z')
  fi
  p=${p%/}
  [ -n "$p" ] || p=/
  printf '%s' "$p"
}

# 自给定目录逐级向上找仓标记（.git 或 gpustack/SHA256SUMS-upstream.txt）：命中打印该层（最内层命中）并退 0；
# 到 / 仍无 → 打印空并退 1。调用方用"输出非空＝该侧是仓上下文"判定。32 级内必然收敛（复审第四节已验）。
find_repo_root() {
  local p=$1
  while :; do
    if [ -e "$p/.git" ] || [ -f "$p/gpustack/SHA256SUMS-upstream.txt" ]; then
      printf '%s' "$p"
      return 0
    fi
    [ "$p" != "/" ] || break
    p=${p%/*}
    [ -n "$p" ] || p=/
  done
  return 1
}

OUT_DIR_ABS=$(cd -- "$(dirname -- "$OUT_FILE")" 2>/dev/null && pwd)
[ -n "$OUT_DIR_ABS" ] || bad_usage "证据输出目录不存在或不可进入：$(dirname -- "$OUT_FILE")"
# 第二途：同一个目录再取一次**物理形**——pwd -P 会穿过符号链接/目录联接解析到真实目标，
# 词法途绿＋物理途红即判红（复审轮 2 新-1 的绕过面②-a/②-b 就是靠"名字不变、真身进仓"打穿的）。
OUT_DIR_PHYS=$(cd -- "$(dirname -- "$OUT_FILE")" 2>/dev/null && pwd -P) \
  || bad_usage "证据输出目录的物理形解析失败（pwd -P）：$(dirname -- "$OUT_FILE")"
[ -n "$OUT_DIR_PHYS" ] || bad_usage "证据输出目录的物理形解析得到空串：$(dirname -- "$OUT_FILE")"
SCRIPT_DIR_PHYS=$(cd -- "$SCRIPT_DIR" && pwd -P)
[ -n "$SCRIPT_DIR_PHYS" ] || bad_usage "脚本目录的物理形解析得到空串：$SCRIPT_DIR"
OUT_ABS="$OUT_DIR_ABS/$(basename -- "$OUT_FILE")"
OUT_ABS_PHYS="$OUT_DIR_PHYS/$(basename -- "$OUT_FILE")"
SC_DIR_C=$(canon "$SCRIPT_DIR")
SC_DIR_P=$(canon "$SCRIPT_DIR_PHYS")
OUT_DIR_C=$(canon "$OUT_DIR_ABS")
OUT_DIR_P=$(canon "$OUT_DIR_PHYS")
OUT_ABS_C=$(canon "$OUT_ABS")
OUT_ABS_P=$(canon "$OUT_ABS_PHYS")
# 四条遍历＝脚本位×（词法＋物理）＋落点位×（词法＋物理）。
SCRIPT_REPO=$(find_repo_root "$SC_DIR_C")
SCRIPT_REPO_P=$(find_repo_root "$SC_DIR_P")
OUT_REPO=$(find_repo_root "$OUT_DIR_C")
OUT_REPO_P=$(find_repo_root "$OUT_DIR_P")
# lex_phys_differs：某一侧的词法形与物理形不同形＝那次解析经过了链接点或 MSYS 目录别名
# （/tmp 与 C:\Users\...\Temp 就是本机最常见的一对，非链接也不同形），进 JSON 供事后审计。
LINK_RESOLVED=false
[ "$OUT_DIR_C" = "$OUT_DIR_P" ] || LINK_RESOLVED=true
[ "$SC_DIR_C" = "$SC_DIR_P" ] || LINK_RESOLVED=true
# 锚点：取四条遍历里最外层的仓根，只作审计记录、不参与判定（判定＝落点侧任一途命中即无条件退 2）。
GUARD_ANCHOR=""
for cand in "$SCRIPT_REPO" "$SCRIPT_REPO_P" "$OUT_REPO" "$OUT_REPO_P"; do
  [ -n "$cand" ] || continue
  if [ -z "$GUARD_ANCHOR" ]; then GUARD_ANCHOR=$cand; continue; fi
  case "$GUARD_ANCHOR" in
    "$cand"|"$cand"/*) GUARD_ANCHOR=$cand ;;
  esac
done
if [ -n "$OUT_REPO" ] || [ -n "$OUT_REPO_P" ]; then
  # 落点自身在任何仓标记链之下——无论脚本住在哪（真仓/假标记子目录/无标记副本/镜像/经链接点）都判死。
  OUT_LEG=词法途
  OUT_HIT=$OUT_REPO
  if [ -z "$OUT_HIT" ]; then OUT_LEG=物理途; OUT_HIT=$OUT_REPO_P; fi
  bad_usage "证据输出路径不得在仓内（$OUT_ABS_C 落在仓根 $OUT_HIT 之下＝落点侧向上遍历命中仓标记，命中途=$OUT_LEG；落点词法形 $OUT_DIR_C／物理形 $OUT_DIR_P），请指到沙箱或镜像路径。"
fi
SCRIPT_ROOTS=""
for root in "$SCRIPT_REPO" "$SCRIPT_REPO_P"; do
  [ -n "$root" ] || continue
  case " $SCRIPT_ROOTS " in
    *" $root "*) : ;;
    *) SCRIPT_ROOTS="$SCRIPT_ROOTS $root" ;;
  esac
  for cand in "$OUT_ABS_C" "$OUT_ABS_P"; do
    case "$cand" in
      "$root"|"$root"/*)
        bad_usage "证据输出路径不得在仓内（$cand 落在仓根 $root 之下＝脚本侧向上遍历命中仓标记），请指到沙箱或镜像路径。" ;;
    esac
  done
done
SCRIPT_ROOTS=${SCRIPT_ROOTS# }
if [ -n "$SCRIPT_ROOTS" ]; then
  GUARD_MODE=repo-context
  echo "NOTE: 守卫判定=仓上下文（脚本侧向上遍历命中仓标记：$SCRIPT_ROOTS；落点侧词法途 $OUT_DIR_C 与物理途 $OUT_DIR_P 两条遍历均无仓标记），落点在仓外（两条解析途都已跑过且都无标记） → 放行：out=$OUT_ABS_C"
else
  # 走到这里必须落点侧两条途也都无标记（上面命中即判死），NOTE 放行有"四条遍历皆无标记"的前置。
  GUARD_MODE=image-layout-exempt
  echo "NOTE: 脚本位与落点位的四条向上遍历（词法途 cd&&pwd ＋ 物理途 cd&&pwd -P 含符号链接/目录联接解析目标）均未在任何上级目录找到仓标记（.git 或 gpustack/SHA256SUMS-upstream.txt），判定为镜像形位（非仓上下文），跳过仓内落点守卫：out=$OUT_ABS_C"
fi
[ -w "$OUT_DIR_ABS" ] || bad_usage "证据输出目录不可写：$OUT_DIR_ABS"

echo "== 终态门禁 ＋ 证据（gate-evidence-v1）ctx=$CTX =="
echo "  SITE_PACKAGES=$SP（安装期来源=${SP_SRC:-未记}）"
echo "  VLLM_SITE_PACKAGES=${VLLM_SP:-未定位}"
echo "  运行期 import 门禁形态=$IMPORT_GATE_STATE"

# ---------- 跑 gate-check post（终态对拍） ----------
log_dir=$(mktemp -d 2>/dev/null)
[ -n "$log_dir" ] && [ -d "$log_dir" ] || bad_usage "无法创建临时目录承接 gate-check 输出。"
trap 'rm -rf "$log_dir"' EXIT
OUT_LOG="$log_dir/gate-check-post.log"

# 守卫判定结果落盘成小文件（进证据 JSON 的 guard 字段，事后可审计）：两条解析途（词法＋物理）各自
# 命中的仓根都记进去；能在盘上持久化的 JSON 只有绿面，绿面里 output_side_repo_root* 恒为 null＝
# "这两条遍历都跑过且都没命中"的凭证（红面在判死处即退出、不落 JSON，见头注）。
# 为什么走文件不走 env：Windows/Git Bash 上 MSYS 运行时会把以 "/" 开头的 env 值按 ANSI 码页强转成
# Windows 路径形态，中文仓根进 env 必折坏（实测 使用→乱码）；文件字节无损，python 按 UTF-8 读。
# 镜像里（Linux）两种形态都无损，此处差异只影响本地跑样的字符串长相。
GUARD_FILE="$log_dir/guard.json"
GUARD_SKIPPED=false
[ "$GUARD_MODE" = image-layout-exempt ] && GUARD_SKIPPED=true
gstr() {
  if [ -n "$1" ]; then printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"; else printf 'null'; fi
}
{ printf '{"mode":"%s","skipped":%s,"legs":"lexical+physical(pwd -P)","lex_phys_differs":%s,"script_side_repo_root":' "$GUARD_MODE" "$GUARD_SKIPPED" "$LINK_RESOLVED"
  gstr "$SCRIPT_REPO"
  printf ',"script_side_repo_root_physical":'; gstr "$SCRIPT_REPO_P"
  printf ',"output_side_repo_root":'; gstr "$OUT_REPO"
  printf ',"output_side_repo_root_physical":'; gstr "$OUT_REPO_P"
  printf ',"anchor_repo_root":'; gstr "$GUARD_ANCHOR"
  printf ',"out_abs":'; gstr "$OUT_ABS_C"
  printf ',"out_abs_physical":'; gstr "$OUT_ABS_P"
  printf '}\n'
} > "$GUARD_FILE" || bad_usage "守卫判定文件写入失败：$GUARD_FILE"

"$PYBIN" "$CTX/gate-check-v3.py" "$CTX" post > "$OUT_LOG" 2>&1
GATE_RC=$?
sed 's|^|  |' "$OUT_LOG"
if [ "$GATE_RC" -ne 0 ]; then
  echo "红：gate-check post 非零退出（rc=$GATE_RC，该脚本自有红码即 2），仍写出 gate_pass=false 的证据。" >&2
fi

# ---------- 写证据 JSON ----------
mkdir -p "$(dirname -- "$OUT_FILE")" || bad_usage "证据目录不可创建：$(dirname -- "$OUT_FILE")"
GATE_EVIDENCE_TMP="$OUT_FILE.tmp.$$"
SM75_EV_OUT="$GATE_EVIDENCE_TMP" \
SM75_EV_CTX="$CTX" \
SM75_EV_SP="$SP" \
SM75_EV_VLLM_SP="$VLLM_SP" \
SM75_EV_SP_SRC="$SP_SRC" \
SM75_EV_GATE_STATE="$IMPORT_GATE_STATE" \
SM75_EV_INSTALL_CTX="$INSTALL_CTX" \
SM75_EV_PY_VERSION="$PY_VERSION" \
SM75_EV_PYBIN="$PYBIN" \
SM75_EV_GATE_RC=$GATE_RC \
SM75_EV_GATE_LOG="$OUT_LOG" \
SM75_EV_GUARD_FILE="$GUARD_FILE" \
"$PYBIN" - <<'PY'
import hashlib, json, os, sys, datetime

ctx = os.environ["SM75_EV_CTX"]
sp = os.environ["SM75_EV_SP"]
fi = os.path.join(sp, "flashinfer")
out = os.environ["SM75_EV_OUT"]
rc = int(os.environ["SM75_EV_GATE_RC"])
log_path = os.environ["SM75_EV_GATE_LOG"]

def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()

manifest_rows = []
header = {}
with open(os.path.join(ctx, "manifest.txt"), encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            for key, name in (("source_wheel=", "source_wheel"),
                              ("source_wheel_sha256=", "source_wheel_sha256")):
                if body.startswith(key):
                    header[name] = body[len(key):].strip()
            continue
        parts = line.split("\t")
        if len(parts) == 5:
            manifest_rows.append(parts)

verbatim = []
for rel, _lines, _bytes, want, kind in manifest_rows:
    if kind != "verbatim":
        continue
    installed = os.path.join(fi, rel[len("newfiles/flashinfer/"):])
    item = {"manifest_path": rel, "installed_path": installed, "sha256_expected": want}
    if os.path.isfile(installed):
        got = sha256(installed)
        item["sha256_actual"] = got
        item["match"] = (got == want)
        item["exists"] = True
    else:
        item["sha256_actual"] = None
        item["match"] = False
        item["exists"] = False
    verbatim.append(item)

append_targets = {
    "append/jit_comm.txt": "jit/comm.py",
    "append/trace_comm.txt": "trace/templates/comm.py",
    "append/comm_init.txt": "comm/__init__.py",
}
append_rows = [row for row in manifest_rows if row[4] == "append"]
appends = []
for blk_rel, _lines, _bytes, _sha, _kind in sorted(append_rows):
    tgt_rel = append_targets.get(blk_rel)
    blk_path = os.path.join(ctx, blk_rel)
    entry = {"block": blk_rel, "block_sha256": None, "target_file": None,
             "target_exists": False, "tail_verbatim": False}
    appends.append(entry)
    if not os.path.isfile(blk_path):
        entry["missing_block"] = True
        continue
    entry["block_sha256"] = sha256(blk_path)
    if tgt_rel is None:
        # 清单里出现了本机不知道目标模块的追加块：判红而不是猜一个目标。
        entry["unknown_target"] = True
        continue
    tgt_path = os.path.join(fi, tgt_rel)
    entry["target_file"] = tgt_path
    with open(blk_path, encoding="utf-8", errors="replace") as fh:
        block = fh.read()
    entry["target_exists"] = os.path.isfile(tgt_path)
    if entry["target_exists"]:
        with open(tgt_path, encoding="utf-8", errors="replace") as fh:
            entry["tail_verbatim"] = fh.read().endswith(block)

with open(log_path, encoding="utf-8", errors="replace") as fh:
    gate_log = fh.read()

guard_file = os.environ.get("SM75_EV_GUARD_FILE") or ""
if guard_file:
    with open(guard_file, encoding="utf-8") as fh:
        guard = json.load(fh)
else:
    guard = {"mode": None, "skipped": None, "legs": None, "lex_phys_differs": None,
             "script_side_repo_root": None, "script_side_repo_root_physical": None,
             "output_side_repo_root": None, "output_side_repo_root_physical": None,
             "anchor_repo_root": None, "out_abs": None, "out_abs_physical": None}

evidence = {
    "evidence_schema": "sm75.gpustack.pcieipc-gate/v1",
    "layer": "1-pcieipc",
    "component": "flashinfer-pcie-ipc-backport",
    "phase": "post",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "generated_by": "gpustack/layers/pcieipc/gate-evidence-v1.sh",
    "site_packages": sp,
    "site_packages_source": os.environ.get("SM75_EV_SP_SRC") or None,
    "vllm_site_packages": os.environ.get("SM75_EV_VLLM_SP") or None,
    "import_gate": os.environ.get("SM75_EV_GATE_STATE"),
    "guard": guard,
    "install_ctx": os.environ.get("SM75_EV_INSTALL_CTX") or None,
    "python": {
        "invoked_as": os.environ.get("SM75_EV_PYBIN"),
        "version": os.environ.get("SM75_EV_PY_VERSION") or None,
    },
    "ctx": ctx,
    "source_wheel": header.get("source_wheel"),
    "source_wheel_sha256": header.get("source_wheel_sha256"),
    "gate_check": {
        "script": os.path.join(ctx, "gate-check-v3.py"),
        "script_sha256": sha256(os.path.join(ctx, "gate-check-v3.py")),
        "rc": rc,
        "pass": (rc == 0),
        "output": gate_log,
    },
    "counts": {
        "manifest_rows_total": len(manifest_rows),
        "manifest_rows_verbatim": len(verbatim),
        "manifest_rows_append": len(append_rows),
        "obligations_rows": sum(1 for line in open(os.path.join(ctx, "obligations.txt"), encoding="utf-8")
                               if line.strip() and not line.startswith("#")),
    },
    "verbatim_files": verbatim,
    "append_blocks": appends,
    "installed_matches_manifest": (
        len(verbatim) == 6
        and len(append_rows) == 3
        and sorted(row[0] for row in append_rows) == sorted(append_targets)
        and all(item["match"] for item in verbatim)
        and all(entry["tail_verbatim"] for entry in appends)
    ),
    "gate_pass": (rc == 0),
}

with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(evidence, fh, ensure_ascii=False, indent=2, sort_keys=False)
    fh.write("\n")
PY
ev_rc=$?
if [ "$ev_rc" -ne 0 ] || [ ! -s "$GATE_EVIDENCE_TMP" ]; then
  echo "红：证据 JSON 生成失败（python rc=$ev_rc 或产物为空），不落地证据：$OUT_FILE" >&2
  rm -f "$GATE_EVIDENCE_TMP"
  exit 2
fi
mv -f "$GATE_EVIDENCE_TMP" "$OUT_FILE" || { echo "红：证据文件改名失败：$OUT_FILE" >&2; rm -f "$GATE_EVIDENCE_TMP"; exit 2; }
"$PYBIN" - "$OUT_FILE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    json.load(fh)
PY
json_rc=$?
if [ "$json_rc" -ne 0 ]; then
  echo "红：证据文件不是合法 JSON（python json 解析 rc=$json_rc）：$OUT_FILE" >&2
  exit 2
fi
echo "EVIDENCE_WRITTEN $OUT_FILE"
if [ "$GATE_RC" -ne 0 ]; then
  echo "红：终态门禁未过（gate-check rc=$GATE_RC），证据已写但本步失败：$OUT_FILE" >&2
  exit "$GATE_RC"
fi
echo "GATE_EVIDENCE_OK out=$OUT_FILE import_gate=$IMPORT_GATE_STATE"
exit 0
