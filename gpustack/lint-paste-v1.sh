#!/usr/bin/env bash
# lint-paste-v1.sh —— 两枚 GPUStack 粘贴块的结构护栏（任务 7）
#
# 做什么：校验粘贴块能否整块贴进 GPUStack 2.2.3 的模型表单——
#   env 块：逐行 KEY=VALUE（键与值两侧、值内都不得含空白或制表）、键不重复、恰 25 行，
#     再与脚本内置的 25 行常量表逐行逐字比顺序（简报钦定 22 条速度 env 原序＋TZ＋shm＋显存申报值）；
#   args 块：逐行 --flag=value、行内不得含空白、恰 24 行、同样与内置常量表逐行逐字比顺序；
#     不含 GPUStack 会自己尾补的 --host/--port/--served-model-name
#     （含则同一参数出现两次，引擎按后值取，用户档位被吃掉）、不含字面 ${（GPUStack 对后端
#     参数不做模板渲染，${MODEL} 会字面直达引擎，审计 A-01）、不含 --speculative-config
#     （MTP 本线暂不启用，且 UI 投机注入位在用户参数之前，vllm.py:531-563/627）、
#     --max-model-len 必须在列且值==262144（用户参数里没有它时，平台静默注入 8192，
#     vllm.py:705-716，审计 B-04）。
#   env 值内空白为什么判死：GPUStack 对模型 env 只做 env.update(self._model.env) 的纯字典
#     更新、零修剪（gpustack/worker/backends/base.py:457-458），`TZ=Asia/Shanghai `（行尾一
#     空格）与 `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 ` 这类值会原样进进程——前者
#     让容器时区退回 UTC，后者让引擎解析整数失败。简报给的 `.+` 是下限，此处按同 L5 的先例加严。
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
#   16 内容断言红——行形态非法 / env 值含 ${ / env 行数 != 25 / args 含 ${ / args 含
#      --speculative-config / args flag 重复 / 关键 flag 缺失 / --max-model-len 取值不符 /
#      args 行数 != 24 / env 或 args 与内置常量表逐字或顺序不符（L11-L13 为修复轮 1 新增门，
#      红面全部归 16，本轮不新增码位）
#
# 断言顺序（五道负控各自命中自己的码，谁也不许被更早的门抢走）：
#   L0 前置 → L1 env 行形态 → L2 env 键唯一(15) → L3 env 值占位符(16) → L4 env 行数==25(16)
#   → L5 args 行形态 → L6 args 尾补三条(14) → L7 args 占位符(16) → L8 args 投机(16)
#   → L9 args flag 唯一(16) → L10 关键 flag 在列＋max-model-len 取值(16)
#   → L11 args 行数==24(16) → L12 env 逐字有序(16) → L13 args 逐字有序(16)
#   L1 排在 L2 之前：坏样张里新增的重复键行本身形态合法，形态门不会抢走 15（夹具自检钉死这点）。
#   L2 排在 L4 之前：塞一行重复键后 env 变 26 行，必须命中 15，不能被行数门以 16 抢走。
#   L6 排在 L7/L8/L10 之前：塞 --port 的样张必须先命中 14，不能被占位符/关键项门以 16 抢走。
#   L10 排在 L11-L13 之前：抽掉 --max-model-len 行的样张（同时只有 23 行）前面十道里由 L10
#     判红，点名必须是「关键项缺失」而不是行数/有序门。
#   L11-L13 排在全部语义门之后（修复轮 1 的硬条件）：有序常量门是兜底门。审查报告第八节疑虑 2
#     实测——把它插到 L0 之后，八条样张全塌成 16、15 与 14 两个码位从整套负控里消失，与本项目
#     10-10 那条「兜底门排在语义检查之前会把所有多改动负控整体抢走」同型；排在 L10 之后则既有
#     七条负控一条不动，且顺带补掉空行、乱序、取值漂移三类放行面。

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

# L1 用计数判据（grep -c），不用 [ -n "$list" ]：命令替换会剥掉结尾换行，当唯一违规是一行
#   空行时 grep 的输出恰是单个换行符，取回空串＝门静默放行（修复轮 1 关闭的 C-1／审查 I-1）。
#   判据与文案共用同一个模式常量，杜绝"计数改了两处、展示还是老形"的漂移。
#   值侧加严（修复轮 1 的 I-2）：键与值两侧、值内均不得含空白或制表，理由见文件头 env 值段。
ENV_SHAPE='^[A-Z0-9_]+=[^[:space:]]+$'
bad_shape_count=$(grep -cEv "$ENV_SHAPE" "$ENV_FILE" || true)
num "$bad_shape_count" "L1 违规行计数"
if [ "$bad_shape_count" -ne 0 ]; then
  die 16 "断言 L1 env 行形态——$bad_shape_count 行不匹配 $ENV_SHAPE（每行必须是 KEY=VALUE，键与值两侧、值内均不得含空白或制表，整行可贴进表单）。违规行号与内容：
$(grep -nEv "$ENV_SHAPE" "$ENV_FILE")"
fi

dup_keys=$(cut -d= -f1 "$ENV_FILE" | sort | uniq -d)
if [ -n "$dup_keys" ]; then
  die 15 "断言 L2 env 键唯一——出现重复键（GPUStack 表单里后行覆盖前行，重复即静默漂移）：$(echo "$dup_keys" | tr '\n' ' ')"
fi

if grep -qF -- '${' "$ENV_FILE"; then
  die 16 "断言 L3 env 值占位符——env 值里出现字面 \${（GPUStack 不做模板渲染，会字面直达进程）：
$(grep -nF -- '${' "$ENV_FILE" | tr '\n' ' ')"
fi

# L4：行数是简报对 env 块的钦定契约（22 速度项＋TZ＋shm＋显存申报值），钉死它才能挡住"顺手加一行/删一行"。
#   第 25 行 GPUSTACK_MODEL_VRAM_CLAIM 原本只写在正文散文里、没烘进粘贴块，结果"整块粘贴"的人
#   调度器永远排队且不报错（10-14 实撞）——所以它必须进块、并由这道行数门守。
if [ "$env_lines" -ne 25 ]; then
  die 16 "断言 L4 env 行数——实测 $env_lines 行，简报契约 25 行（前 22 条速度 env 原序＋TZ＋shm＋VRAM_CLAIM）。"
fi

# ---------------- args 块 ----------------
args_lines=$(wc -l < "$ARGS_FILE")
num "$args_lines" "args 行数"

# L5 同 L1：计数判据（同型实现两处一起改，修复轮 1 的 C-1）。
ARGS_SHAPE='^--[^[:space:]]+$'
bad_arg_shape_count=$(grep -cEv "$ARGS_SHAPE" "$ARGS_FILE" || true)
num "$bad_arg_shape_count" "L5 违规行计数"
if [ "$bad_arg_shape_count" -ne 0 ]; then
  die 16 "断言 L5 args 行形态——$bad_arg_shape_count 行不匹配 $ARGS_SHAPE（一行一条 flag=value，行内不得含空白；
  空格形 --flag value 会被 GPUStack 的 shlex 拆行，JSON 值到引擎手里即坏，见 gen-args-paste-v1.py 头注）。违规行号与内容：
$(grep -nEv "$ARGS_SHAPE" "$ARGS_FILE")"
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

# ---------------- L11-L13：行数与逐字有序常量门（修复轮 1，关闭 I-3 与疑虑 2） ----------------
# 两张表是粘贴块契约的常驻副本：env＝简报钦定 22 条原序＋TZ＋shm＋显存申报值（25 行），args＝生成器产物的 24 行。
# 有了它们，"顺手改一行值""整块倒序""少贴一行"都会在这里点名到行，而不是只在生成那一次被验。
# 排位是硬条件（见文件头「断言顺序」）：必须全部排在 L10 之后，否则 15 与 14 两个码位会从整套
#   负控里消失，兜底门把所有多改动负控整体抢走。
ENV_EXPECT='NCCL_P2P_LEVEL=SYS
VLLM_USE_MODELSCOPE=false
VLLM_FIREFLY=auto
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
NCCL_DEBUG=WARN
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=0
OMP_NUM_THREADS=1
VLLM_SM75_QWEN38_HC_GEMV=1
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
VLLM_USE_BREAKABLE_CUDAGRAPH=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=16
VLLM_SM75_QSA_FLASHINFER_PREFILL=1
VLLM_SM75_QSA_SHARED_TARGET_SCRATCH=1
VLLM_SM75_QSA_FP8_MAIN_KV=1
VLLM_SM75_QSA_CUBLAS_INDEXER=1
VLLM_SM75_QSA_SELECTED_PAGES_PREFILL=1
VLLM_ENFORCE_STRICT_TOOL_CALLING=1
VLLM_SM75_QSA_MIXED_PREFILL=1
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
TZ=Asia/Shanghai
GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB=64
GPUSTACK_MODEL_VRAM_CLAIM=120779177984'

ARGS_EXPECT='--dtype=float16
--hf-overrides={"dtype":"float16"}
--tensor-parallel-size=8
--enable-expert-parallel
--enable-ep-weight-filter
--language-model-only
--max-model-len=262144
--max-num-seqs=4
--max-num-batched-tokens=4096
--gpu-memory-utilization=0.92
--kv-cache-dtype=fp8_e4m3
--kv-cache-memory-bytes=2415919104
--enable-prefix-caching
--engram-config={"cpu_offload":true}
--gdn-prefill-backend=flashqla_sm75
--generation-config=vllm
--enable-prompt-tokens-details
--compilation-config={"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}
--block-size=16
--scheduler-cls=vllm.v1.core.sched.scheduler_sm75.SM75Scheduler
--enable-auto-tool-choice
--tool-call-parser=qwen3_xml
--reasoning-parser=qwen3
--chat-template=/opt/vllm-sm75/templates/qwen38-next-high-compat.jinja'

# A 路线（默认交付）：保留图像、关闭视频；与纯文本档只差第 6 行。
ARGS_EXPECT_VIDEOOFF='--dtype=float16
--hf-overrides={"dtype":"float16"}
--tensor-parallel-size=8
--enable-expert-parallel
--enable-ep-weight-filter
--limit-mm-per-prompt={"image":16,"video":0}
--max-model-len=262144
--max-num-seqs=4
--max-num-batched-tokens=4096
--gpu-memory-utilization=0.92
--kv-cache-dtype=fp8_e4m3
--kv-cache-memory-bytes=2415919104
--enable-prefix-caching
--engram-config={"cpu_offload":true}
--gdn-prefill-backend=flashqla_sm75
--generation-config=vllm
--enable-prompt-tokens-details
--compilation-config={"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}
--block-size=16
--scheduler-cls=vllm.v1.core.sched.scheduler_sm75.SM75Scheduler
--enable-auto-tool-choice
--tool-call-parser=qwen3_xml
--reasoning-parser=qwen3
--chat-template=/opt/vllm-sm75/templates/qwen38-next-high-compat.jinja'

# L11：args 侧补齐与 env L4 对称的行数门。此前删掉非关键行（24→23）或加一行合法 flag
#   （24→25）都全绿，而注册卡 §四 明写"整块粘贴 24 行"——本项目已经发生过面板静默丢掉 10 个
#   定制项的事故，缺这一道就等于契约无人守。
if [ "$args_lines" -ne 24 ]; then
  die 16 "断言 L11 args 行数——实测 $args_lines 行，契约 24 行（与 L13 常量表等长；env 侧同一契约由 L4 守）。"
fi

ordered_gate() { # ordered_gate 实际件 常量表 门号 名称 契约行数 —— 只读比对，首个差异即红 16 并点名行号
  local file=$1 expect=$2 gate=$3 label=$4 want_lines=$5
  local want i actual expected
  want=$(printf '%s\n' "$expect" | wc -l)
  num "$want" "$gate 常量表行数"
  # 表本身先自证：拿一张被改坏的表去比样张，"全绿"没有任何含义（空判据的镜像面＝错判据）。
  # 契约行数按门传入（env 25／args 24）——两块的行数从 10-14 起不再相等，共用一个常数会让其中一块自证必红。
  [ "$want" -eq "$want_lines" ] || die 2 "错误：$gate 常量表自身不符——表内 $want 行，与 $want_lines 行契约不一致，拒绝校验（护栏脚本被改坏，样张判定作废）。"
  i=1
  while [ "$i" -le "$want_lines" ]; do
    # 逐行取文本比较（POSIX 形）。禁止照抄 cmp -s <(canon)——那是 bash-only 进程替换，
    # 而本脚本现过 dash -n 语法双检。
    actual=$(sed -n "${i}p" "$file")
    expected=$(printf '%s\n' "$expect" | sed -n "${i}p")
    if [ "$actual" != "$expected" ]; then
      die 16 "断言 $gate $label——首个差异在第 $i 行：实测「$actual」≠ 常量「$expected」。本门钉逐字内容与顺序（乱序、改值、增删行都在此点名到行）；确要改档时，粘贴块与本脚本常量表两处一起改，见 gpustack/backend-version-card-v1.md §五。"
    fi
    i=$((i + 1))
  done
}

ordered_gate "$ENV_FILE" "$ENV_EXPECT" L12 "env 逐字有序" 25

# L13 常量表按 args 第 6 行选择：纯文本档（language-model-only）或 A 路线（limit-mm）。
# 两套都是 24 行契约；第 6 行不是这两者即红，避免"改一行就绕过逐字门"。
case "$(sed -n '6p' "$ARGS_FILE")" in
  --language-model-only) ARGS_CANON=$ARGS_EXPECT ;;
  --limit-mm-per-prompt=*) ARGS_CANON=$ARGS_EXPECT_VIDEOOFF ;;
  *) die 16 "断言 L13 args 逐字有序——第 6 行既不是纯文本档的 --language-model-only，也不是 A 路线的 --limit-mm-per-prompt=…；只认这两套 24 行契约。" ;;
esac
ordered_gate "$ARGS_FILE" "$ARGS_CANON" L13 "args 逐字有序" 24

echo "绿｜env $ENV_FILE：$env_lines 行，形态合法、键唯一、无占位符、与常量表逐字同序"
echo "绿｜args $ARGS_FILE：$args_lines 行，形态合法、无尾补三条、无占位符、无投机、flag 唯一、关键项在列、与常量表逐字同序"
echo "LINT_PASTE_OK env=$env_lines args=$args_lines"
exit 0
