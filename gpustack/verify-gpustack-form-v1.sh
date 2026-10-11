#!/usr/bin/env bash
# verify-gpustack-form-v1.sh —— 按 GPUStack 2.2.3 同形态拼装 vLLM 实例，并做拼装/判据双向验证（任务 8）
#
# 做什么：读任务 7 的两枚粘贴块（env-paste / args-paste），拼出一条与 GPUStack worker 拼出来的
#   命令行同形的 docker run。--dry-run 只打印这条命令（整段可粘贴执行）；实跑模式起容器并跑
#   六条运行期断言。本脚本消费的是"粘贴块＋尾补三条"这套拼装语义，不是 GPUStack 进程本身。
# 为什么：交付承诺是"贴进 GPUStack 表单即等价于这条命令"。等价性要有一条可复核的拼装产物；
#   顺序错位（模型路径不在第一、尾补三条跑到用户参数之前）会让引擎按后值取同名参数，
#   表现为档位漂移而不报错，所以顺序本身钉成判据（退 28）。
#
# 拼装语义的事实来源（只读参照，本仓不改上游一行）：
#   副本＝K:\sm75\GPUStack使用SM75\gpustack源码\v2.2.3-正式（GPUStack 正式 v2.2.3）
#   gpustack/worker/backends/vllm.py:171   command = ["vllm", "serve"]
#   gpustack/worker/backends/vllm.py:233/315  restart_policy = ContainerRestartPolicyEnum.NEVER
#   gpustack/worker/backends/vllm.py:235/317  privileged = True
#   gpustack/worker/backends/vllm.py:342   host_network = True
#   gpustack/worker/backends/vllm.py:343   shm_size = int(shm_size_gib * (1 << 30))
#   gpustack/worker/backends/vllm.py:613   arguments: List[str] = [self._model_path]（模型路径位置第一）
#   gpustack/worker/backends/vllm.py:656-659  extend_args_no_exist 尾补 --host/--port/--served-model-name
#   gpustack/worker/backends/base.py:604   def _get_configured_mounts（:617 model_dir = os.path.dirname(
#     self._model_path) 只挂模型目录的父目录；:618-622 ContainerMount(path=model_dir) 宿主与容器同路径）
#
# 打印形态为什么逐词加单引号：args 块里 --compilation-config={"mode":0,...} 含顶层逗号，未加引号时
#   bash 的 brace expansion 会把它拆成 --compilation-config=mode:0 等多个坏词（本机 bash 5.3 实测），
#   命令卡就不可粘贴。逐词 sq 包裹后实测还原成 1 个完整参数。执行路径另走 argv 数组，不经 shell。
#
# 用法（三种模式互斥：--dry-run 打印 / --order-file 只判序 / 默认＝实跑属阶段二）：
#   bash gpustack/verify-gpustack-form-v1.sh --dry-run --model-dir DIR --model NAME --port P --name M
#   bash gpustack/verify-gpustack-form-v1.sh --dry-run --neg-pcieipc ...（负控开关，同上参数）
#   bash gpustack/verify-gpustack-form-v1.sh --order-file FIXTURE ...（对"合法拼装产物人工改序"的夹具判序）
#   SM75_VERIFY_LIVE_OK=1 bash gpustack/verify-gpustack-form-v1.sh ...（实跑，阶段二在服务器执行）
# 夹具入口 --env-file/--args-file 与 lint-paste-v1.sh 同形，供负控替换仓内件。
# 判据自测入口：SM75_SELFCTEST=1 bash gpustack/verify-gpustack-form-v1.sh —— 只跑七条运行期判据的
#   正例＋反例（不碰网络与容器），任一 BAD 退 2。
#
# 落点守卫（交接约束第 8 条）：--out 是启动日志落点，三条边界——①禁指容器内挂载点（挂载根＝模型目录
#   父目录，容器内同路径，日志写进去等于让引擎容器读写自己的日志）；②含 .. 段取确定性红；③词法途
#   （cd && pwd）与物理途（cd && pwd -P，穿过符号链接/目录联接）各自向上遍历仓标记（.git 与
#   gpustack/SHA256SUMS-upstream.txt），任一途命中仓内即红——比 gen-args-paste-v1.py 的"禁写进上游
#   14 顶层项"更严，因为本脚本输出属运行期证据，任何仓内落点都会污染工作树。守卫排在拼装与写盘之前。
#
# 退出码（只用简报登记的 21-28 与既有 2，本任务不新增码位，见交接约束第 4 条）：
#   0  通过
#   2  用法/前置守卫红——参数缺失或形态非法、依赖粘贴块缺失/为空/行数不足、args 含尾补三条、
#      落点守卫红、neg 模式的 env 侧效未变、拼装产物形态异常（固定行 0 命中或多命中）、
#      实跑未经 SM75_VERIFY_LIVE_OK 确认、判据自测出现 BAD
#   21 /health 判负——HTTP 码非 2xx、body 为空、或 body 非 JSON 形态
#   22 一次 chat completion 的 content 缺失或为空
#   23 模型清单未回显 served-model-name（body 为空时先由存在性断言判同一码）
#   24 启动日志出现字面 ${（占位符未回填即字面直达引擎，审计 A-01）
#   25 启动日志出现 --speculative-config（本线 MTP 未启用，须恰 0 次）
#   26 启动日志出现 ray 执行器注入（版本名缺 -custom 后缀即回落 Ray，调研第八节第 1 条）
#   27 负控 --neg-pcieipc 下启动日志出现 FLASHINFER_PCIE_IPC（期望恰 0 次）
#   28 顺序断言红——模型路径 / 用户参数 / 尾补三条三段行号失衡
#
# 断言顺序（每条独立可命中，谁也不许被更早的门抢走）：
#   P0 参数与用法 → P1 依赖件存在性＋非空＋行数下限(2) → P2 args 含尾补三条(2) → P3 落点守卫(2)
#   → P4 neg 侧效自检(2) → A 拼装 → B 顺序断言(28) →〔dry-run／order 到此为止〕
#   → L0 实跑确认门(2) → C 起容器 → C1..C6 运行期断言(21/23/22/24/25/26) → C7 neg 日志(27)
#   P1 排在一切之前：粘贴块缺失时必须红，绝不静默拼出空 --env-file 或空参数块。
#   P2 是任务 7 lint 的双保险：尾补三条由本脚本提供，args 里再有一份即重复参数（lint 用码 14，
#      本脚本按主控增补第 2 条用既有码 2，不新开码位）。
#   B 只在同一次拼装产出上做 grep -n，绝不拼两次输出再拼接；--order-file 复用同一个 assert_order，
#      夹具＝--dry-run 输出人工改序，因此测的是产物而非另一份实现。
#   C4..C7 的"计数==0"判据前一律先挂"日志字节>0"存在性断言（judge_log_present）。
#
# 阶段一／阶段二分界：阶段一只执行 --dry-run / --order-file / SM75_SELFCTEST 三条路径（本任务落码即
#   验证到位）；C 段的取数路径（docker run、curl、docker logs、jq/python3 解析 content）属阶段二待验，
#   其判据逻辑已由自测证红，但"取回来的数据形态是否符合预期"未证——报告里逐条列为门禁项。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SELF=${0##*/}
ENV_FILE_DEFAULT=$SCRIPT_DIR/env-paste-v1.txt
ARGS_FILE_DEFAULT=$SCRIPT_DIR/args-paste-v1.txt
# 镜像默认 tag 取自仓内权威登记：gpustack/backend-version-card-v1.md 第 13 行「镜像名称 image_name」。
IMAGE_DEFAULT=vllm-sm75-gpustack:v0.1.7-beta2-v1
MIN_BLOCK_LINES=24            # 简报契约：两枚粘贴块各 24 行，取此为下限
PCIE_KEY=VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC
REPO_MARKER_GIT=.git
REPO_MARKER_TABLE=gpustack/SHA256SUMS-upstream.txt
WAIT_SECS=900                 # 实跑等 /health 的上限（8 卡 fp16 262K 档加载慢，阶段二再校）

MODE=''
NEG_PCIE=0
IMAGE=$IMAGE_DEFAULT
MODEL_DIR=''
MODEL=''
PORT=''
SERVED_NAME=''
ENV_FILE_IN=''
ARGS_FILE_IN=''
OUT_FILE_IN=''
ORDER_FILE=''

usage() {
  echo "用法：bash gpustack/$SELF (--dry-run | --order-file FIXTURE) --model-dir DIR --model NAME --port P --name M"
  echo "　　　　　　　　　　　　　　　[--neg-pcieipc] [--image TAG] [--env-file F] [--args-file F] [--out LOG]"
  echo "说明：--dry-run＝只打印拼装命令（不碰容器）；--order-file＝对一份拼装产物文本跑顺序断言；"
  echo "　　　两者都不给＝实跑模式（起容器＋运行期断言，属阶段二，需 SM75_VERIFY_LIVE_OK=1）。"
}

die() { # die 码 文案 —— 全脚本唯一的红出口。走 stderr：dry-run 的 stdout 必须只剩可粘贴的命令。
  echo "红｜退出码 $1｜$2" >&2
  exit "$1"
}

num() { # num 值 文案 —— 计数类判据的哨兵守卫：非纯数字即红，绝不把空串当 0 放行
  case "$1" in
    ''|*[!0-9]*) die 2 "错误：$2 取到非数字值「$1」，判据不可信（拒绝按 0 放行）。" ;;
  esac
}

sq() { # sq 词 —— 单引号包裹，内部单引号转义为 '\''；使打印出的每个参数都能原样过 shell
  local s=${1-}
  s=${s//\'/\'\\\'\'}
  printf "'%s'" "$s"
}

# 下面这几个函数会 die，因此一律"把结果写进 _RES、由调用方在父 shell 里赋值"。
# 原因（本轮实测踩到）：函数被 $(...) 命令替换调用时身处子 shell，die 的 exit 只退子 shell，
# 父脚本会带着空返回值继续往下跑——守卫等于没装。调用形态必须是 `f a b; X=$_RES`。
_RES=''

count_of() { # count_of 文本 固定行 —— 整行全等命中数写进 _RES
  local n
  if [ -z "${2-}" ]; then _RES=0; return 0; fi
  n=$(printf '%s\n' "$1" | grep -cxF -- "$2") || true
  [ -n "$n" ] || n=NOT_DIGIT
  _RES=$n
}

line_no_of() { # line_no_of 文本 固定行 —— 恰 1 命中则 _RES=行号并返回 0，否则 _RES=命中数并返回 1
  local n
  count_of "$1" "$2"
  n=$_RES
  num "$n" "固定行「$2」命中数"
  if [ "$n" -ne 1 ]; then _RES=$n; return 1; fi
  _RES=$(printf '%s\n' "$1" | grep -nxF -- "$2" | head -n 1 | cut -d: -f1)
  num "$_RES" "固定行「$2」行号"
  return 0
}

has_dotdot() { # has_dotdot 路径 —— 任一路径段是 .. 即真（正斜杠与反斜杠两种分隔符都判）
  case "$1" in
    .|..|../*|*/..|*/../*|*\\..\\*|*\\..) return 0 ;;
  esac
  return 1
}

abspath() { # abspath 路径 名称 —— 词法绝对化写进 _RES；父目录不在盘上即红（本脚本不代建目录）
  local d b abs
  d=$(dirname -- "$1")
  b=$(basename -- "$1")
  abs=$(cd -- "$d" 2>/dev/null && pwd) \
    || die 2 "错误：$2 的所在目录不存在或不可进入：$d（本脚本不代建目录，请先建好再指落点）。"
  case "$b" in
    .) _RES=$abs ;;
    *) _RES=$abs/$b ;;
  esac
}

hit_repo_upwards() { # hit_repo_upwards 路径 —— 词法＋物理双途向上遍历仓标记；命中则 _RES=仓根返回 0
  local start=$1 cur phys marker parent
  _RES=''
  local -a legs=()
  phys=$(cd -- "$(dirname -- "$start")" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$start")") || phys=''
  legs=("$start")
  [ -n "$phys" ] && legs+=("$phys")
  for cur in "${legs[@]}"; do
    phys=$cur
    while :; do
      for marker in "$REPO_MARKER_GIT" "$REPO_MARKER_TABLE"; do
        if [ -e "$phys/$marker" ]; then _RES=$phys; return 0; fi
      done
      parent=$(dirname -- "$phys")
      [ "$parent" = "$phys" ] && break
      phys=$parent
    done
  done
  return 1
}

# ---------- P3 落点守卫 ----------
guard_out_target() { # guard_out_target 路径 挂载根 —— 三条边界逐条判死，绝对落点写进 _RES，红则零落盘
  local lex root shown
  has_dotdot "$1" && die 2 "错误：--out 含 .. 路径段，无法可靠判定真实落点，本脚本拒绝猜测（退出码 2）：$1"
  abspath "$1" "--out"
  lex=$_RES
  case $lex/ in
    "$2"/*) die 2 "错误：--out 落进容器内挂载点（退出码 2）：$lex 位于挂载根 $2 之下——该目录以同路径 bind 进容器（base.py:604/617），引擎进程会读写自己的日志落点。请改指仓外目录。" ;;
  esac
  if hit_repo_upwards "$lex"; then
    root=$_RES
    shown=${lex#"$root"/}
    die 2 "错误：--out 落点在仓内（退出码 2）：解析为 $lex，位于仓根 $root 之下（词法途 cd&&pwd 或物理途 cd&&pwd -P 向上遍历命中仓标记 .git／gpustack/SHA256SUMS-upstream.txt）。启动日志属运行期证据，任何仓内落点都会污染工作树，请改指仓外（如 /tmp）。落点相对仓根＝$shown"
  fi
  _RES=$lex
}

# ---------- P1 依赖件前置守卫 ----------
precheck_block() { # precheck_block 路径 名称 —— 存在性＋非空＋行数下限；行数写进 _RES
  local bytes lines
  [ -e "$1" ] || die 2 "错误：$2 不存在：$1 —— 本脚本要消费任务 7 的粘贴块，缺件时必须红，绝不静默拼出空 $2（那样拼出来的命令看着完整，实际少了整段 env 或整段参数）。"
  [ -f "$1" ] || die 2 "错误：$2 不是普通文件：$1"
  [ -r "$1" ] || die 2 "错误：$2 不可读：$1"
  bytes=$(wc -c < "$1" | tr -d '[:space:]')
  num "$bytes" "$2 字节数"
  [ "$bytes" -gt 0 ] || die 2 "错误：$2 为空文件（0 字节）：$1"
  lines=$(wc -l < "$1" | tr -d '[:space:]')
  num "$lines" "$2 行数"
  [ "$lines" -ge "$MIN_BLOCK_LINES" ] \
    || die 2 "错误：$2 行数 $lines 低于下限 $MIN_BLOCK_LINES（简报契约：两枚粘贴块各 24 行）：$1"
  _RES=$lines
}

# ---------- A 拼装（argv 数组，执行不经 shell）----------
build_argv() {
  local line
  ARGV=(run --rm -d --name "$CONTAINER_NAME" --network host --privileged --restart no --shm-size 64g)
  ARGV+=(--env-file "$ENV_FILE")
  if [ "$NEG_PCIE" -eq 1 ]; then
    ARGV+=(-e "$PCIE_KEY=0")
  fi
  ARGV+=(-v "$MOUNT_ROOT:$MOUNT_ROOT" "$IMAGE" vllm serve "$MODEL_PATH")
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    ARGV+=("$line")
  done < "$ARGS_FILE"
  ARGV+=(--host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME")
}

print_plan() { # 逐词单引号、每词一行、除末词外带续行符——保证整段可直接粘贴执行
  local i=0 w
  printf 'docker'
  for w in "${ARGV[@]}"; do
    if [ "$i" -eq 0 ]; then
      printf ' %s' "$(sq "$w")"
    else
      printf ' \\\n  %s' "$(sq "$w")"
    fi
    i=$((i + 1))
  done
  printf '\n'
}

# ---------- B 顺序断言（28）：在同一份拼装文本上做 grep -n ----------
assert_order() { # assert_order 文本 —— 三段行号失衡即 28
  local text=$1 ln_serve='' ln_model='' ln_host='' ln_port='' ln_name='' line n min_user='' max_user=''
  local -a want=(
    "serve 基准行|  'serve' \\"
    "模型路径行|  $(sq "$MODEL_PATH") \\"
    "尾补 --host 行|  '--host' \\"
    "尾补 --port 行|  '--port' \\"
    "尾补 --served-model-name 行|  '--served-model-name' \\"
  )
  local item key val n
  for item in "${want[@]}"; do
    key=${item%%|*}; val=${item#*|}
    line_no_of "$text" "$val" || die 2 "错误：拼装产物里「$key」命中 $_RES 次（应为恰 1 次），该文本不是合法拼装产物，顺序判据不予执行（退出码 2）。"
    n=$_RES
    case $key in
      serve*) ln_serve=$n ;;
      模型路径*) ln_model=$n ;;
      *host*) ln_host=$n ;;
      *port*) ln_port=$n ;;
      *served-model-name*) ln_name=$n ;;
    esac
  done
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    line_no_of "$text" "  $(sq "$line") \\" \
      || die 2 "错误：args 参数行「$line」在拼装产物里命中 $_RES 次（应为恰 1 次），无法逐行定序（退出码 2）。"
    n=$_RES
    if [ -z "$min_user" ]; then min_user=$n; max_user=$n; continue; fi
    [ "$n" -lt "$min_user" ] && min_user=$n
    [ "$n" -gt "$max_user" ] && max_user=$n
  done < "$ARGS_FILE"
  [ -n "$min_user" ] || die 2 "错误：args 粘贴块里没有可定序的参数行（空块不该走到这里，P1 应已判红）。"
  local -a chain=(
    "$ln_serve|$ln_model|'vllm serve' 必须早于模型路径"
    "$ln_model|$min_user|模型路径必须早于任一用户参数（vllm.py:613 把它放在 arguments 首位）"
    "$max_user|$ln_host|全部用户参数必须早于尾补三条（vllm.py:656-659 的 extend_args_no_exist 在末尾追加）"
    "$ln_host|$ln_port|尾补 --host 必须早于 --port"
    "$ln_port|$ln_name|尾补 --port 必须早于 --served-model-name"
  )
  local link a b why
  for link in "${chain[@]}"; do
    a=${link%%|*}; b=${link#*|}; b=${b%%|*}; why=${link##*|}
    if [ "$a" -ge "$b" ]; then
      die 28 "断言 B 顺序红——$why（实测行号 $a >= $b；全链 serve=$ln_serve 模型=$ln_model 用户参数=[$min_user,$max_user] host=$ln_host port=$ln_port name=$ln_name）。错位会让引擎按后值取同名参数，表现为档位漂移而不报错。"
    fi
  done
  echo "绿｜B 顺序断言｜serve=$ln_serve < 模型路径=$ln_model < 用户参数[$min_user,$max_user] < host=$ln_host < port=$ln_port < name=$ln_name" >&2
}

# ---------- 运行期判据（C1..C7）：与取数解耦，便于阶段一用正例＋反例证红 ----------
judge_health() { # judge_health HTTP码 body —— 空 body 判负（简报契约），退 21
  local code=$1 body=$2 bytes
  case $code in
    2??) ;; *) die 21 "断言 C1 /health 红——HTTP 码 ${code}（期望 2xx），服务未就绪或端口不符。" ;;
  esac
  bytes=$(printf '%s' "$body" | wc -c | tr -d '[:space:]')
  num "$bytes" "/health body 字节数"
  [ "$bytes" -gt 0 ] || die 21 "断言 C1 /health 红——HTTP $code 但 body 为空（简报契约：空 body 判负）。注意 vLLM 原生 /health 惯例是空 body＋200，本判据以简报为准；阶段二实测若恒空即回来找主控裁定改档（报告门禁项 G-1）。"
  case $body in
    *'{'*|*'['*) ;; *) die 21 "断言 C1 /health 红——body 非 JSON 形态：$body" ;;
  esac
  echo "绿｜C1 /health｜HTTP $code，body $bytes 字节，JSON 形态"
}

judge_chat_content() { # judge_chat_content content body —— content 缺失或为空退 22
  local content=$1 body=$2
  [ -n "$content" ] || die 22 "断言 C2 chat completion 红——choices[0].message.content 缺失或为空。响应片段：$(printf '%.400s' "$body")"
  echo "绿｜C2 chat completion｜content 非空（${#content} 字符）"
}

judge_models() { # judge_models body served-name —— 先挂存在性断言，再判回显，退 23
  local body=$1 name=$2 bytes
  bytes=$(printf '%s' "$body" | wc -c | tr -d '[:space:]')
  num "$bytes" "模型清单 body 字节数"
  [ "$bytes" -gt 0 ] || die 23 "断言 C3 模型清单红——body 为空（0 字节）：'未回显' 与 '没取到' 无法区分，存在性断言先红。"
  printf '%s' "$body" | grep -qF -- "$name" \
    || die 23 "断言 C3 模型清单红——未回显 served-model-name「$name」。响应片段：$(printf '%.400s' "$body")"
  echo "绿｜C3 模型清单｜回显 served-model-name=$name"
}

judge_log_present() { # judge_log_present 日志文件 —— 一切"计数==0"判据之前的存在性断言
  local bytes
  [ -f "$1" ] || die 2 "错误：启动日志不存在：$1（实跑分支须先 docker logs 落盘再断言）。"
  bytes=$(wc -c < "$1")
  num "$bytes" "启动日志字节数"
  [ "$bytes" -gt 0 ] || die 2 "错误：启动日志为空（0 字节）：$1 —— 日志没生成时的『0 次』不构成任何证明，后续计数判据不予执行。"
  echo "INFO｜启动日志 $bytes 字节：$1"
}

judge_log_no_placeholder() { # 24
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -cF -- '${' "$f") || true
  num "$hits" "字面 \${ 命中数"
  [ "$hits" -eq 0 ] || die 24 "断言 C4 占位符红——启动日志出现字面 \${ 共 $hits 次（GPUStack 对后端参数不做模板渲染，占位符未回填即字面直达引擎，审计 A-01）：
$(grep -nF -- '${' "$f" | head -n 5)"
  echo "绿｜C4 占位符｜启动日志无字面 \${"
}

judge_log_no_speculative() { # 25
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -cF -- '--speculative-config' "$f") || true
  num "$hits" "--speculative-config 命中数"
  [ "$hits" -eq 0 ] || die 25 "断言 C5 投机红——启动日志出现 --speculative-config 共 $hits 次（本线 MTP 未启用，用户 2026-10-11 裁决；UI 投机注入位排在用户参数之前，重复参数被后值吃掉，vllm.py:531-563/627）：
$(grep -nF -- '--speculative-config' "$f" | head -n 5)"
  echo "绿｜C5 投机｜--speculative-config 恰 0 次"
}

judge_log_no_ray() { # 26
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -Ec -- 'distributed-executor-backend[= ]+ray|raylet|RayDistributedExecutor|Ray Client' "$f") || true
  num "$hits" "ray 执行器标记命中数"
  [ "$hits" -eq 0 ] || die 26 "断言 C6 ray 红——启动日志出现 ray 执行器注入共 $hits 次（版本名缺 -custom 后缀即默认回落 Ray，标准镜像无 Ray 必失败，utils/command.py:416-457）：
$(grep -nE -- 'distributed-executor-backend[= ]+ray|raylet|RayDistributedExecutor|Ray Client' "$f" | head -n 5)"
  echo "绿｜C6 ray｜无 ray 执行器注入"
}

judge_log_neg_pcie_absent() { # 27
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -cF -- FLASHINFER_PCIE_IPC "$f") || true
  num "$hits" "FLASHINFER_PCIE_IPC 命中数"
  [ "$hits" -eq 0 ] || die 27 "断言 C7 负控红——env 开关 $PCIE_KEY 已置 0，启动日志仍出现 FLASHINFER_PCIE_IPC 共 $hits 次（说明该 env 并非这条日志的唯一控制点，或 -e 覆盖未生效）：
$(grep -nF -- FLASHINFER_PCIE_IPC "$f" | head -n 5)"
  echo "绿｜C7 负控｜开关置 0 后启动日志不再出现 FLASHINFER_PCIE_IPC"
}

# ---------- 判据自测（SM75_SELFCTEST=1）：七条各一正一反，反例必须命中自己的码 ----------
run_case() { # run_case 期望ok|码 名称 命令与参数… —— 子 shell 捕获 rc，BAD 记进 FAIL
  local want=$1 name=$2; shift 2
  local rc=0 out
  out=$("$@" 2>&1) || rc=$?
  if [ "$want" = ok ]; then
    if [ "$rc" -eq 0 ]; then echo "SELFTEST_OK   $name（绿）"; else echo "SELFTEST_BAD  $name 期望绿，实得 rc=$rc：$out"; FAIL=$((FAIL + 1)); fi
  else
    if [ "$rc" -eq "$want" ]; then echo "SELFTEST_OK   $name 反例命中退出码 $want"; else echo "SELFTEST_BAD  $name 期望 $want，实得 rc=$rc：$out"; FAIL=$((FAIL + 1)); fi
  fi
}

selftest_all() {
  local dir good bad
  FAIL=0
  dir=$(mktemp -d 2>/dev/null) || { echo "SELFTEST_BAD 无法建沙箱目录"; exit 2; }
  printf '%s\n' 'Engine launched with args: --dtype float16 --max-model-len 262144' > "$dir/good.log"
  printf '' > "$dir/empty.log"
  printf '%s\n' 'loading ${MODEL} from disk' > "$dir/placeholder.log"
  printf '%s\n' 'speculative decoding enabled: --speculative-config={"method":"mtp"}' > "$dir/spec.log"
  printf '%s\n' 'Using RayDistributedExecutor, raylet started at pid 4242' > "$dir/ray.log"
  printf '%s\n' 'INFO flashinfer pcie ipc: FLASHINFER_PCIE_IPC enabled path=/x' > "$dir/pcie.log"

  run_case ok      "C1 正例 /health 2xx＋非空 JSON" judge_health 200 '{"status":"ok"}'
  run_case 21      "C1 反例 空 body" judge_health 200 ''
  run_case 21      "C1 反例 HTTP 503" judge_health 503 '{"status":"unavailable"}'
  run_case 21      "C1 反例 非 JSON 形态" judge_health 200 'health ok'
  run_case ok      "C2 正例 content 非空" judge_chat_content '好的' '{"choices":[{"message":{"content":"好的"}}]}'
  run_case 22      "C2 反例 content 为空" judge_chat_content '' '{"choices":[{"message":{"content":""}}]}'
  run_case ok      "C3 正例 清单回显" judge_models '{"data":[{"id":"qwen38-next","object":"model"}]}' qwen38-next
  run_case 23      "C3 反例 未回显" judge_models '{"data":[{"id":"other-model"}]}' qwen38-next
  run_case 23      "C3 反例 body 为空（存在性先红）" judge_models '' qwen38-next
  run_case ok      "C4 正例 日志无占位符" judge_log_no_placeholder "$dir/good.log"
  run_case 2       "C4 反例 日志为空（存在性断言先红）" judge_log_present "$dir/empty.log"
  run_case 24      "C4 反例 日志含字面 \${" judge_log_no_placeholder "$dir/placeholder.log"
  run_case ok      "C5 正例 日志无投机" judge_log_no_speculative "$dir/good.log"
  run_case 25      "C5 反例 日志含 --speculative-config" judge_log_no_speculative "$dir/spec.log"
  run_case ok      "C6 正例 日志无 ray" judge_log_no_ray "$dir/good.log"
  run_case 26      "C6 反例 日志含 RayDistributedExecutor" judge_log_no_ray "$dir/ray.log"
  run_case ok      "C7 正例 neg 日志无 FLASHINFER_PCIE_IPC" judge_log_neg_pcie_absent "$dir/good.log"
  run_case 27      "C7 反例 neg 日志仍出现 FLASHINFER_PCIE_IPC" judge_log_neg_pcie_absent "$dir/pcie.log"
  run_case ok      "存在性断言正例 日志非空" judge_log_present "$dir/good.log"
  run_case 2       "存在性断言反例 日志文件不存在" judge_log_present "$dir/missing-file.log"
  rm -rf -- "$dir" 2>/dev/null || true
  if [ "$FAIL" -ne 0 ]; then
    echo "SELFTEST_FAIL 有 $FAIL 条判据未按预期动作"
    exit 2
  fi
  echo "SELFTEST_OK 判据自测全部通过（20 条用例，七条断言各自正反例齐）"
  exit 0
}

if [ "${SM75_SELFCTEST:-0}" = 1 ]; then
  selftest_all
fi

# ---------- P0 参数解析 ----------
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dry-run) [ -z "$MODE" ] || die 2 "错误：模式参数互斥（--dry-run 与 --order-file 只能选一个）。"; MODE=dry; shift ;;
    --order-file)
      [ -z "$MODE" ] || die 2 "错误：模式参数互斥（--dry-run 与 --order-file 只能选一个）。"
      [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--order-file 缺少文件值。"
      has_dotdot "$2" && die 2 "错误：--order-file 含 .. 路径段，无法可靠判定落点（退出码 2）：$2"
      MODE=order; ORDER_FILE=$2; shift 2 ;;
    --neg-pcieipc) NEG_PCIE=1; shift ;;
    --image) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--image 缺少值。"; IMAGE=$2; shift 2 ;;
    --model-dir) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--model-dir 缺少值。"; MODEL_DIR=$2; shift 2 ;;
    --model) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--model 缺少值。"; MODEL=$2; shift 2 ;;
    --port) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--port 缺少值。"; PORT=$2; shift 2 ;;
    --name) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--name 缺少值。"; SERVED_NAME=$2; shift 2 ;;
    --env-file) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--env-file 缺少值。"; has_dotdot "$2" && die 2 "错误：--env-file 含 .. 路径段（退出码 2）：$2"; ENV_FILE_IN=$2; shift 2 ;;
    --args-file) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--args-file 缺少值。"; has_dotdot "$2" && die 2 "错误：--args-file 含 .. 路径段（退出码 2）：$2"; ARGS_FILE_IN=$2; shift 2 ;;
    --out) [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--out 缺少值。"; OUT_FILE_IN=$2; shift 2 ;;
    *) echo "错误：未识别的参数「$1」"; usage; die 2 "退出码 2＝用法红。" ;;
  esac
done
[ -n "$MODE" ] || MODE=live

for pair in "--model-dir|$MODEL_DIR" "--model|$MODEL" "--port|$PORT" "--name|$SERVED_NAME"; do
  [ -n "${pair#*|}" ] || { usage; die 2 "错误：缺少必填参数「${pair%%|*}」——拼一条完整命令所需，本脚本不替你猜默认值（猜出来的命令卡会被当成实测口径）。"; }
done
case $PORT in
  ''|*[!0-9]*) die 2 "错误：--port 必须是纯数字端口，实为「$PORT」。" ;;
esac
case $MODEL_DIR in
  /*) ;; *) die 2 "错误：--model-dir 必须是绝对路径（GPUStack 的 model_path 是绝对路径，同路径 bind 才有意义），实为「$MODEL_DIR」。" ;;
esac
has_dotdot "$MODEL_DIR" && die 2 "错误：--model-dir 含 .. 路径段，无法可靠判定挂载落点（退出码 2）：$MODEL_DIR"
case $MODEL in
  ''|.|..|*/*|*[[:space:]]*) die 2 "错误：--model 只能是模型目录名，不得为空、含路径分隔符或空白：$MODEL" ;;
esac
case $SERVED_NAME in
  *[!a-zA-Z0-9_.-]*) die 2 "错误：--name（served-model-name）只允许字母数字与 _ . -，实为「$SERVED_NAME」（它同时被拼进容器名，docker 对名称字符集有同样限制）。" ;;
esac
[ -n "$IMAGE" ] || die 2 "错误：--image 不得为空。"

# 工具哨兵：所有判据都来自这些工具的 stdout，而空串恰是各道的"无问题"值，缺件必须先把门。
for needed_tool in grep sed cut sort uniq wc head tr basename dirname printf; do
  command -v "$needed_tool" >/dev/null 2>&1 || die 2 "错误：缺少必要工具「$needed_tool」，拒绝拼装。"
done
if [ "$MODE" = live ]; then
  # jq/python3 是二选一的 content 解析器，不在这里判，由下方独立门判"二者皆缺"。
  for needed_tool in docker curl mktemp; do
    command -v "$needed_tool" >/dev/null 2>&1 || die 2 "错误：实跑模式缺少必要工具「$needed_tool」。"
  done
fi

abspath "${ENV_FILE_IN:-$ENV_FILE_DEFAULT}" "--env-file"; ENV_FILE=$_RES
abspath "${ARGS_FILE_IN:-$ARGS_FILE_DEFAULT}" "--args-file"; ARGS_FILE=$_RES
abspath "$MODEL_DIR" "--model-dir"; MODEL_DIR=$_RES
MODEL_PATH=$MODEL_DIR/$MODEL
MOUNT_ROOT=$(dirname -- "$MODEL_DIR")
CONTAINER_NAME=sm75-verify-$SERVED_NAME

# ---------- P1 依赖件存在性（排在一切拼装与写盘之前）----------
precheck_block "$ENV_FILE" "env 粘贴块"; ENV_LINES=$_RES
precheck_block "$ARGS_FILE" "args 粘贴块"; ARGS_LINES=$_RES

# ---------- P2 双保险：args 内不得含尾补三条 ----------
dup_tail=$(grep -E '^--(host|port|served-model-name)(=|$)' "$ARGS_FILE") || true
if [ -n "$dup_tail" ]; then
  die 2 "错误：args 粘贴块含 GPUStack/本脚本尾补的三参数之一（退出码 2＝前置守卫档）：
$dup_tail
这三项由本脚本的尾补三条提供（vllm.py:656-659），重复出现会让命令行出现两份同名参数，后者生效但语义不可预期。任务 7 的 lint-paste-v1.sh 在同一判据上用码 14，两处互为双保险。"
fi

# ---------- P3 落点守卫 ----------
OUT_FILE=${OUT_FILE_IN:-}
if [ -z "$OUT_FILE" ]; then
  OUT_DIR=$(cd -- "${TMPDIR:-/tmp}" 2>/dev/null && pwd) \
    || die 2 "错误：默认日志落点不可用（TMPDIR 与 /tmp 都不可进入），请显式给 --out 指向仓外已存在目录。"
  OUT_FILE=$OUT_DIR/sm75-gpustack-form-verify-v1-bootlog.txt
fi
guard_out_target "$OUT_FILE" "$MOUNT_ROOT"; OUT_FILE=$_RES

# ---------- P4 neg 侧效自检 ----------
if [ "$NEG_PCIE" -eq 1 ]; then
  pcie_hits=$(grep -cE "^$PCIE_KEY=" "$ENV_FILE") || true
  num "$pcie_hits" "$PCIE_KEY 在 env 块的行数"
  [ "$pcie_hits" -eq 1 ] \
    || die 2 "错误：负控前置红（退出码 2）——$PCIE_KEY 在 env 块出现 $pcie_hits 次，期望恰 1 次。键缺失或多行时『置 0』不构成可判定的形态变化，负控即空转。"
  pcie_val=$(grep -E "^$PCIE_KEY=" "$ENV_FILE" | head -n 1)
  pcie_val=${pcie_val#*=}
  [ "$pcie_val" != 0 ] \
    || die 2 "错误：负控侧效未变（退出码 2）——env 块里 $PCIE_KEY 已经是 0，--neg-pcieipc 再置 0 不改变 env 形态，等于没做负控。要么换一份基线值为 1 的 --env-file 夹具，要么去掉 --neg-pcieipc。"
fi

# ---------- A 拼装 + B 顺序断言（同一份输出）----------
build_argv
[ "${#ARGV[@]}" -gt 0 ] || die 2 "错误：拼装数组为空。"
PLAN=$(print_plan)
[ -n "$PLAN" ] || die 2 "错误：拼装产物为空。"
if [ "$NEG_PCIE" -eq 1 ]; then
  # grep 是单行判据，所以逐词两行各自验恰 1 命中（-e 与它的值在打印形态里分占两行）。
  count_of "$PLAN" "  $(sq -e) \\"; num "$_RES" "-e 行命中数"; n_e=$_RES
  count_of "$PLAN" "  $(sq "$PCIE_KEY=0") \\"; num "$_RES" "pcie 值行命中数"; n_v=$_RES
  { [ "$n_e" -eq 1 ] && [ "$n_v" -eq 1 ]; } \
    || die 2 "错误：负控侧效未进命令行（退出码 2）——拼装产物里 '-e' 行命中 $n_e 次、'$PCIE_KEY=0' 行命中 $n_v 次，两者均应恰 1 次。开关没落到命令上，负控结论不可信。"
fi

case $MODE in
  order)
    [ -f "$ORDER_FILE" ] || die 2 "错误：--order-file 指向的文件不存在或不是文件：$ORDER_FILE"
    [ -r "$ORDER_FILE" ] || die 2 "错误：--order-file 不可读：$ORDER_FILE"
    ORDER_TEXT=$(cat -- "$ORDER_FILE")
    [ -n "$ORDER_TEXT" ] || die 2 "错误：--order-file 内容为空，无文本可判序：$ORDER_FILE"
    assert_order "$ORDER_TEXT"
    echo "ORDER_CHECK_OK file=$ORDER_FILE" >&2
    exit 0
    ;;
esac

assert_order "$PLAN"

# ---------- dry-run ----------
if [ "$MODE" = dry ]; then
  printf '%s\n' "# GPUStack 同形态拼装（dry-run，未启动任何容器）—— 下面整段可直接粘贴执行"
  printf '%s\n' "# 容器内命令＝末段 'vllm' 'serve' 起至 '$SERVED_NAME'；env 经 --env-file 注入，其键值不重复打印"
  printf '%s\n' "# 依赖：env 块 $ENV_LINES 行（$ENV_FILE）／args 块 $ARGS_LINES 行（$ARGS_FILE）；挂载根 $MOUNT_ROOT（同路径 bind）"
  printf '%s\n' "# neg-pcieipc=$NEG_PCIE（置 1 时命令行多两词 -e $PCIE_KEY=0，覆盖 env-file 同名键）"
  printf '%s\n' "# 实跑日志落点（阶段二用 --out 改指，禁落挂载根与仓内）：$OUT_FILE"
  printf '%s\n' "$PLAN"
  printf '%s\n' "DRYRUN_OK env_lines=$ENV_LINES args_lines=$ARGS_LINES neg_pcieipc=$NEG_PCIE mode=$MODE" >&2
  exit 0
fi

# ---------- L0 实跑确认门（安传规范第 1 条：实机验证集中在阶段二）----------
if [ "${SM75_VERIFY_LIVE_OK:-0}" != 1 ]; then
  die 2 "本脚本的实跑分支未执行（退出码 2）：阶段一禁止起容器。阶段二在服务器上加前缀 SM75_VERIFY_LIVE_OK=1 再跑同一条命令。"
fi

BODY_DIR=$(mktemp -d 2>/dev/null) || die 2 "错误：无法创建响应体临时目录（mktemp -d）。"
trap 'rm -rf -- "$BODY_DIR" 2>/dev/null' EXIT

same=$(docker ps -aq --filter "name=^${CONTAINER_NAME}$" | wc -l | tr -d '[:space:]')
num "$same" "同名容器数"
[ "$same" -eq 0 ] || die 2 "错误：同名容器已存在（$same 个）：$CONTAINER_NAME —— 本脚本不做删除动作（破坏性操作需另行授权），请改 --name 或人工清理。"

if ! command -v jq >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  die 2 "错误：解析 chat completion 的 content 需要 jq 或 python3，二者皆缺。"
fi
API_KEY=${SM75_VERIFY_API_KEY:-}

# ---------- C 起容器 + 运行期断言 ----------
docker "${ARGV[@]}" || die 2 "错误：docker run 启动失败（原始报错见上，本脚本不改写 docker 的失败语义）。"
echo "INFO｜容器已启动：$CONTAINER_NAME，等待 /health 最多 ${WAIT_SECS}s（每 10s 一次）"
deadline=$((SECONDS + WAIT_SECS))
code=000
while :; do
  code=$(curl -sS -o "$BODY_DIR/health" -w '%{http_code}' --max-time 10 "http://127.0.0.1:$PORT/health" 2>/dev/null) || code=000
  [ "$code" = 200 ] && break
  if [ "$SECONDS" -ge "$deadline" ]; then
    docker logs "$CONTAINER_NAME" > "$OUT_FILE" 2>&1 || true
    judge_log_present "$OUT_FILE"
    die 21 "断言 C1 /health 红——等待 ${WAIT_SECS}s 后仍未 200（最后一次 HTTP 码 $code）；日志已落 $OUT_FILE，先按日志定位。"
  fi
  sleep 10
done
judge_health "$code" "$(cat -- "$BODY_DIR/health")"

models=$(curl -sS --max-time 60 "http://127.0.0.1:$PORT/v1/models") || die 21 "错误：/v1/models 请求失败（C3 的取数路径）。"
judge_models "$models" "$SERVED_NAME"

chat_body=$(curl -sS --max-time 300 -X POST -H 'Content-Type: application/json' \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"只回复两个字：好的\"}],\"max_tokens\":8,\"temperature\":0}" \
  "http://127.0.0.1:$PORT/v1/chat/completions") || die 22 "错误：chat completion 请求失败（C2 的取数路径）。"
content=$( { command -v jq >/dev/null 2>&1 && printf '%s' "$chat_body" | jq -r '.choices[0].message.content // empty'; } \
  || printf '%s' "$chat_body" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception:
    pass' )
judge_chat_content "$content" "$chat_body"

docker logs "$CONTAINER_NAME" > "$OUT_FILE" 2>&1 || die 2 "错误：docker logs 取日志失败：$CONTAINER_NAME"
judge_log_no_placeholder "$OUT_FILE"
judge_log_no_speculative "$OUT_FILE"
judge_log_no_ray "$OUT_FILE"
if [ "$NEG_PCIE" -eq 1 ]; then
  judge_log_neg_pcie_absent "$OUT_FILE"
fi

echo "VERIFY_LIVE_OK name=$SERVED_NAME port=$PORT image=$IMAGE log=$OUT_FILE"
exit 0
