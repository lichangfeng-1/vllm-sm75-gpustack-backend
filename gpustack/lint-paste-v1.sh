#!/usr/bin/env bash
# lint-paste-v1.sh —— 两枚 GPUStack 粘贴块的结构护栏（任务 7）
#
# 做什么：校验粘贴块能否整块贴进 GPUStack 2.2.3 的模型表单——
#   env 块：逐行 KEY=VALUE、键不重复、恰 24 行（简报钦定 22 条速度 env ＋ TZ ＋ shm）；
#   args 块：逐行 --flag=value、不含 GPUStack 会自己尾补的 --host/--port/--served-model-name
#     （含则同一参数出现两次，引擎按后值取，用户档位被吃掉）、不含字面 ${（GPUStack 对后端
#     参数不做模板渲染，${MODEL} 会字面直达引擎，审计 A-01）、不含 --speculative-config
#     （MTP 本线暂不启用，且 UI 投机注入位在用户参数之前，vllm.py:531-563/627）、
#     --max-model-len 必须在列且值==262144（用户参数里没有它时，平台静默注入 8192，
#     vllm.py:705-716，审计 B-04）。
# 为什么：gen 脚本的门只保护"生成那一次"，表单里的那一份在运行期没有第二道锁；
#   粘贴块又是人手复制粘贴、最容易被手改坏的物件，所以交付时点必须再有一道独立复检。
#
# 用法：
#   bash gpustack/lint-paste-v1.sh                       # 校验仓内两块
#   bash gpustack/lint-paste-v1.sh --env FILE            # 只换 env 夹具（args 用仓内件）
#   bash gpustack/lint-paste-v1.sh --args FILE           # 只换 args 夹具（env 用仓内件）
#   bash gpustack/lint-paste-v1.sh --env A --args B      # 负控夹具（A/B 一律沙箱件）
#
# 落点守卫（交接约束第 8 条）：本脚本只读不写，没有 --out 面，因此不存在写进上游作用域的
#   可能；夹具路径含 .. 段时取确定性红 2，不猜真实落点。
#
# 退出码（只用已登记码位，本任务不新增，见交接约束第 4 条）：
#   0  两块全部通过
#   2  用法/前置红——缺必要工具、文件不存在或不可读、空块、含 CR 字节、末行缺换行、参数含 .. 段
#   14 参数排除红——args 出现 GPUStack 尾补三条之一
#   15 env 键重复
#   16 内容断言红——行形态非法 / env 值含 ${ / env 行数 != 24 / args 含 ${ / args 含
#      --speculative-config / args flag 重复 / 关键 flag 缺失 / --max-model-len 取值不符
#
# 断言顺序（五道负控各自命中自己的码，谁也不许被更早的门抢走）：
#   L0 前置 → L1 env 行形态 → L2 env 键唯一(15) → L3 env 值占位符(16) → L4 env 行数==24(16)
#   → L5 args 行形态 → L6 args 尾补三条(14) → L7 args 占位符(16) → L8 args 投机(16)
#   → L9 args flag 唯一(16) → L10 关键 flag 在列＋max-model-len 取值(16)
#   L1 排在 L2 之前：坏样张里新增的重复键行本身形态合法，形态门不会抢走 15（夹具自检钉死这点）。
#   L2 排在 L4 之前：塞一行重复键后 env 变 25 行，必须命中 15，不能被行数门以 16 抢走。
#   L6 排在 L7/L8/L10 之前：塞 --port 的样张必须先命中 14，不能被占位符/关键项门以 16 抢走。
#   L10 排在最后：抽掉 --max-model-len 行的样张前面九道全绿，只能由 L10 判红。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SELF=${0##*/}
ENV_FILE="$SCRIPT_DIR/env-paste-v1.txt"
ARGS_FILE="$SCRIPT_DIR/args-paste-v1.txt"

usage() {
  echo "用法：bash gpustack/$SELF [--env KEY=VALUE块] [--args 参数块]"
  echo "说明：不带参数＝校验仓内 gpustack/env-paste-v1.txt 与 gpustack/args-paste-v1.txt。"
}

die() { # die 码 文案
  echo "红｜退出码 $1｜$2"
  exit "$1"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --env|--args)
      [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：$1 缺少参数值。"
      case "$2" in
        .|..|../*|*/..|*/../*) die 2 "错误：$1 的夹具路径含 .. 路径段，无法可靠判定落点（拒绝猜测）。" ;;
      esac
      [ -f "$2" ] || die 2 "错误：$1 指向的文件不存在或不是文件：$2"
      [ -r "$2" ] || die 2 "错误：$1 指向的文件不可读：$2"
      if [ "$1" = "--env" ]; then ENV_FILE=$2; else ARGS_FILE=$2; fi
      shift 2
      ;;
    *) echo "错误：未识别的参数「$1」"; usage; die 2 "退出码 2＝用法红，参数只认 --env/--args/-h。" ;;
  esac
done

# 工具哨兵（L0）：本脚本所有判据都来自这些工具的 stdout，而空串恰是各道的"无问题"值。
# 缺件时命令替换得到空串，会把"工具不可用"误报成"校验通过"，故一律先验可执行。
for needed_tool in grep sed cut sort uniq wc head tr; do
  command -v "$needed_tool" >/dev/null 2>&1 || die 2 "错误：缺少必要工具「$needed_tool」，拒绝校验。"
done

num() { # num 值 文案 —— 计数类判据的哨兵守卫：非纯数字即红，不放行空串
  case "$1" in
    ''|*[!0-9]*) die 2 "错误：$2 取到非数字值「$1」，判据不可信（拒绝按 0 放行）。" ;;
  esac
}

# 块前置（L0）：存在性＋字节形态。空块必须在任何计数判据之前判红，
# 否则"0 行、0 键、0 个违规"会让后面每一道门都漂亮地绿过去。
precheck() { # precheck 路径 名称
  local bytes cr last
  bytes=$(wc -c < "$1")
  num "$bytes" "$2 字节数"
  [ "$bytes" -gt 0 ] || die 2 "错误：$2 为空文件（0 字节），拒绝校验：$1"
  cr=$(tr -dc '\r' < "$1" | wc -c)
  num "$cr" "$2 CR 计数"
  [ "$cr" -eq 0 ] || die 2 "错误：$2 含 $cr 个 CR 字节（Windows 行尾），请用 LF 检出：git -c core.autocrlf=false clone"
  last=$(tail -c 1 "$1" | tr -d '\n')
  [ -z "$last" ] || die 2 "错误：$2 末行缺换行符（末字节非 LF）：$1"
  return 0
}

precheck "$ENV_FILE" "env 粘贴块"
precheck "$ARGS_FILE" "args 粘贴块"

# ---------------- env 块 ----------------
env_lines=$(wc -l < "$ENV_FILE")
num "$env_lines" "env 行数"

bad_shape=$(grep -Ev '^[A-Z0-9_]+=.+$' "$ENV_FILE" || true)
if [ -n "$bad_shape" ]; then
  die 16 "断言 L1 env 行形态——存在不匹配 ^[A-Z0-9_]+=.+\$ 的行（每行必须是 KEY=VALUE，整行可贴进表单）：
$bad_shape"
fi

dup_keys=$(cut -d= -f1 "$ENV_FILE" | sort | uniq -d)
if [ -n "$dup_keys" ]; then
  die 15 "断言 L2 env 键唯一——出现重复键（GPUStack 表单里后行覆盖前行，重复即静默漂移）：$(echo "$dup_keys" | tr '\n' ' ')"
fi

if grep -qF -- '${' "$ENV_FILE"; then
  die 16 "断言 L3 env 值占位符——env 值里出现字面 \${（GPUStack 不做模板渲染，会字面直达进程）：
$(grep -nF -- '${' "$ENV_FILE" | tr '\n' ' ')"
fi

# L4：行数是简报对 env 块的钦定契约（22 速度项＋TZ＋shm），钉死它才能挡住"顺手加一行/删一行"。
if [ "$env_lines" -ne 24 ]; then
  die 16 "断言 L4 env 行数——实测 $env_lines 行，简报契约 24 行（前 22 条速度 env 原序＋TZ＋shm）。"
fi

# ---------------- args 块 ----------------
args_lines=$(wc -l < "$ARGS_FILE")
num "$args_lines" "args 行数"

bad_arg_shape=$(grep -Ev '^--[^[:space:]]+$' "$ARGS_FILE" || true)
if [ -n "$bad_arg_shape" ]; then
  die 16 "断言 L5 args 行形态——存在不匹配 ^--[^[:space:]]+\$ 的行（一行一条 flag=value；
  空格形 --flag value 会被 GPUStack 的 shlex 拆行，JSON 值到引擎手里即坏，见 gen-args-paste-v1.py 头注）：
$bad_arg_shape"
fi

tail_three=$(grep -E '^--(host|port|served-model-name)(=|$)' "$ARGS_FILE" || true)
if [ -n "$tail_three" ]; then
  die 14 "断言 L6 参数排除——args 含 GPUStack 自动尾补的三参数之一（含则同参数重复，引擎按后值取）：
$tail_three"
fi

if grep -qF -- '${' "$ARGS_FILE"; then
  die 16 "断言 L7 占位符——args 含字面 \${（GPUStack 对后端参数不做模板渲染，占位符会字面直达引擎，审计 A-01）：
$(grep -nF -- '${' "$ARGS_FILE" | tr '\n' ' ')"
fi

if grep -qF -- '--speculative-config' "$ARGS_FILE"; then
  die 16 "断言 L8 投机参数——args 含 --speculative-config（本线 MTP 暂不启用，用户 2026-10-11 裁决；
  且 UI 投机注入位排在用户参数之前，会造成重复参数，vllm.py:531-563/627）。后续开启走草稿模型
  绝对路径参数行，见 gpustack/backend-version-card-v1.md。"
fi

dup_flags=$(sed 's/=.*$//' "$ARGS_FILE" | sort | uniq -d)
if [ -n "$dup_flags" ]; then
  die 16 "断言 L9 flag 唯一——同一参数出现多次（引擎按后值取，档位即静默漂移）：$(echo "$dup_flags" | tr '\n' ' ')"
fi

missing=""
for key_flag in --max-model-len --tensor-parallel-size --engram-config --chat-template; do
  grep -Eq -- "^${key_flag}(=|\$)" "$ARGS_FILE" || missing="$missing $key_flag"
done
if [ -n "$missing" ]; then
  die 16 "断言 L10 关键项缺失——args 不在列：$missing
  （--max-model-len 缺失时 GPUStack 会静默注入 8192，262144 档被砍且无任何提示，vllm.py:705-716）。"
fi

mml=$(grep -E '^--max-model-len=' "$ARGS_FILE" | head -n 1)
[ -n "$mml" ] || die 16 "断言 L10 关键项取值——找不到 --max-model-len= 行（等号形才受本护栏检查）。"
if [ "$mml" != "--max-model-len=262144" ]; then
  die 16 "断言 L10 关键项取值——实为「$mml」，期望 --max-model-len=262144（防 GPUStack 静默钳 8192）。"
fi

echo "绿｜env $ENV_FILE：$env_lines 行，形态合法、键唯一、无占位符"
echo "绿｜args $ARGS_FILE：$args_lines 行，形态合法、无尾补三条、无占位符、无投机、flag 唯一、关键项在列"
echo "LINT_PASTE_OK env=$env_lines args=$args_lines"
exit 0
