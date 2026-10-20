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
# 拼装语义的事实来源（只读参照，本仓不改上游一行；指针一律"定义行＋生效行"成对写）：
#   副本＝<GPUStack 正式 v2.2.3 源码的本地检出目录>，改成你自己的路径（1139 行的 vllm.py）
#     本行原先写的是制作机上的绝对路径，已换成占位形：写死盘符＝别人照抄第一步就红，还带出内部目录形制；下一位请保持占位形，不要填回真路径。
#   vllm.py:171                     生效行 command = ["vllm", "serve"]
#   vllm.py:233 / :315              生效行 restart_policy=NEVER（单实例分支 / ray 分支）
#   vllm.py:235 / :317              生效行 privileged=True
#   vllm.py:331-338                 生效行 logger.info 打印 image/command/arguments/envs＝GPUStack
#                                   注入参数日志面（worker 侧）。与"本脚本自拼装形态的容器 argv 面"
#                                   是两物，不可互证：前者属阶段二门禁 G-3，后者由 C5a/C5b 判。
#   vllm.py:342                     生效行 host_network=True
#   vllm.py:343                     生效行 shm_size=int(shm_size_gib * (1 << 30))（env 块 64 → 64g）
#   vllm.py:613                     生效行 arguments: List[str] = [self._model_path]（模型路径位置第一）
#   vllm.py:531 / :627              定义行 _get_speculative_arguments / 生效行 arguments.extend(该函数)
#   vllm.py:651 / :652              定义行 user_backend_parameters = _flatten_backend_param() /
#                                   生效行 arguments.extend(user_backend_parameters)（用户参数上界）
#   vllm.py:654-659                 生效段 extend_args_no_exist：656 ("--host", self._worker.ip)、
#                                   657 ("--port", str(ctx.port))、658-659 ("--served-model-name", …)。
#                                   平台尾补的 --host 取 worker IP，不是 127.0.0.1；本脚本默认按简报契约补
#                                   127.0.0.1，属"同形不同值"的已知差异（阶段二门禁 G-2，用 --host 覆写）。
#   vllm.py:705-716                 定义行 _build_max_model_len_arguments / 生效行 714-715
#                                   （用户未给 max-model-len 时平台补 8192；args 块已给 262144 故不触发）
#   base.py:604                     定义行 def _get_configured_mounts
#   base.py:615                     生效门 and not GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
#                                   （镜像部署形下平台整体跳过挂载）
#   base.py:617                     生效行 model_dir = os.path.dirname(self._model_path)
#                                   ＝只挂"模型目录本身"，不是模型目录的父目录
#   base.py:618-622                 生效行 mounts.append(ContainerMount(path=model_dir)) 宿主同路径 bind
#   auto_remove：上述副本全树 0 命中 ⇒ 平台不带 `--rm`，本脚本默认拼装里也不带（见 build_argv 注）。
#
# 打印形态为什么逐词加单引号：args 块里 --compilation-config={"mode":0,...} 含顶层逗号，未加引号时
#   bash 的 brace expansion 会把它拆成 --compilation-config=mode:0 等多个坏词（本机 bash 5.3 实测），
#   命令卡就不可粘贴。逐词 sq 包裹后实测还原成 1 个完整参数。执行路径另走 argv 逐词位置参数，不经 shell。
#
# 为什么全篇不用 bash 数组：交付要求 `bash -n` 与 `dash -n` 双检都过，而 dash 的解析器对 `name=(…)`
#   一律报 `"(" unexpected`（本机 /usr/bin/dash 实测）。argv 因此以"一行一词"的字符串承载，出栈时
#   逐词装进位置参数；代价是词内不得含换行，这条已由 argv_push 判死（红 2），不是假设。
#
# 用法（三种模式互斥：--dry-run 打印 / --order-file 只判序 / 默认＝实跑属阶段二）：
#   bash gpustack/verify-gpustack-form-v1.sh --dry-run --model-dir DIR --model NAME --port P --name M
#   bash gpustack/verify-gpustack-form-v1.sh --dry-run --neg-pcieipc ...（负控开关，同上参数）
#   bash gpustack/verify-gpustack-form-v1.sh --order-file FIXTURE ...（对"合法拼装产物人工改序"的夹具判序）
#   SM75_VERIFY_LIVE_OK=1 bash gpustack/verify-gpustack-form-v1.sh ...（实跑，阶段二在服务器执行）
# 夹具入口 --env-file/--args-file 与 lint-paste-v1.sh 同形，供负控替换仓内件。
# 挂载宽度：默认与平台一致（挂 --model-dir 本身，＝ base.py:617 的 dirname(model_path)）；
#   --mount-parent 才宽一级挂父目录，默认关闭。
# 尾补 host：--host VALUE，默认 127.0.0.1（简报契约）；阶段二跑 G-2 时用它换成平台 worker IP
#   （vllm.py:656 尾补取 self._worker.ip），三条运行期取数（/health、/v1/models、chat）都跟着打这个值。
#   值校验只走码 2：空值、以短横开头、含空白/制表/换行都红；拼装后再由 judge_tail_shape 断末六词
#   恰是尾补三对，词数被改动同样红。此入参只写在本头注与任务 9 映射表说明里，注册卡不动。
# 鉴权键：只从环境变量 SM75_VERIFY_API_KEY 取，实跑时写进 0600 的 curl 配置文件后以 `curl -K` 传入，
#   绝不进命令行（同机 ps 可见＝对外交付物的攻击向量面）；键值也不打印。
# 判据自测入口：SM75_SELFCTEST=1 bash gpustack/verify-gpustack-form-v1.sh —— 只跑运行期判据的
#   正例＋反例（不碰网络与容器），任一 BAD 退 2。
#
# 落点守卫（交接约束第 8 条）：--out 是启动日志落点，三条边界——①禁指容器内挂载点（挂载根默认＝模型目录
#   本身，容器内同路径，日志写进去等于让引擎容器读写自己的日志）。①与③同标准，两侧都各算词法途
#   （cd && pwd）与物理途（cd && pwd -P，穿过符号链接/目录联接/挂载别名），四个组合任一命中即红——
#   只比词法时，同一目录换个名字（如经 /tmp 的挂载别名）就能绕过；②含 .. 段取确定性红；
#   ③两途各自向上遍历仓标记（.git 与 gpustack/SHA256SUMS-upstream.txt），任一途命中仓内即红——
#   比 gen-args-paste-v1.py 的"禁写进上游 14 顶层项"更严，因为本脚本输出属运行期证据，任何仓内落点
#   都会污染工作树。守卫排在拼装与写盘之前。
#   向上遍历拼标记路径时，文件系统根这一级必须拼成「/标记」而不是「//标记」：MSYS 把开头的双斜杠
#   当 UNC（\\主机\共享）去查网络，偶发返回"存在"，会让一切 /tmp 落点假红（本机实测 8 次中 1 次）；
#   POSIX 下开头的 // 亦属实现自定义。abspath／physpath／hit_repo_upwards 三处同型，一并遵守。
#
# 退出码（只用简报登记的 21-28 与既有 2，本任务不新增码位，见交接约束第 4 条）：
#   0  通过
#   2  用法/前置守卫红——参数缺失或形态非法（含端口不在 1-65535、带前导零、--host 值为注入形）、
#      依赖粘贴块缺失/为空/行数不足、
#      模型目录或模型路径不存在不可读、实跑缺 config.json、args 含尾补三条、落点守卫红、
#      neg 模式的 env 侧效未变、拼装产物形态异常（固定行 0 命中或多命中、顺序槽位缺环、词内含换行）、
#      鉴权键字符集非法、实跑未经 SM75_VERIFY_LIVE_OK 确认、判据自测出现 BAD
#   21 健康探针非 200 或未就绪（200＋空 body 合法；容器已退出也落这一码，文案点名退出原因）
#   22 一次 chat completion 的 content 缺失、为空或只有空白字符
#   23 模型清单未回显 served-model-name（body 为空时先由存在性断言判同一码；HTTP 非 2xx 同码但文案分面）
#   24 启动日志出现字面 ${（占位符未回填即字面直达引擎，审计 A-01）
#   25 出现 --speculative-config——C5a 判自拼装 argv 面（本地 dry-run 即可命中），
#      C5b 判本容器启动日志面；GPUStack 平台注入日志面（vllm.py:331-338）是第三物，属阶段二 G-3
#   26 启动日志出现 ray 执行器注入（版本名缺 -custom 后缀即回落 Ray，调研第八节第 1 条）
#   27 负控 --neg-pcieipc 下启动日志出现 FLASHINFER_PCIE_IPC（期望恰 0 次）
#   28 顺序断言红——模型路径 / 用户参数 / 尾补三条三段行号失衡
#
# 断言顺序（每条独立可命中，谁也不许被更早的门抢走）：
#   P0 参数与用法（含端口 1-65535 与字符集）→ P1 依赖件存在性＋非空＋行数下限(2)
#   → P1b 模型目录与模型路径存在可读(2) → P2 args 含尾补三条(2) → P3 落点守卫(2) → P4 neg 侧效自检(2)
#   → A 拼装 → 尾补形状断言(2) → C5a 拼装 argv 面无 --speculative-config(25) → B 顺序断言(28) →〔dry-run／order 到此为止〕
#   → L0 实跑确认门(2) → L1 实跑加严面（config.json 在位）(2) → C 起容器
#   → C0 等待循环（每轮先验容器存活，容器已退出立刻 21）→ C1 健康(21) → C3 清单(23) → C2 对话(22)
#   → C4..C6 日志面(24/25/26) → C7 neg 日志(27)
#   C5a 排在 B 之前：C5a 只吃拼装文本、B 只吃行号顺序，两条样张形态不重叠（改序样张不含投机参数、
#   投机样张不改序），所以谁先跑都不会把对方的红抢走。
#   尾补形状门排在 C5a／B 之前：末六词已经不是尾补三对时，行号顺序的结论没有意义，先按用法档拦下。
#   P1 排在一切之前：粘贴块缺失时必须红，绝不静默拼出空 --env-file 或空参数块。
#   P2 是任务 7 lint 的双保险，且本脚本必须独立成防线（不靠"L5 也会拦"）：判据先剥行首空白、
#      再认 `=`／空白／行尾三种分隔，所以 `   --host=x` 与 `--port 9000` 这类形一样红
#      （lint 用码 14，本脚本按主控增补第 2 条用既有码 2，不新开码位）。
#   P1b 排在起容器之前：模型路径不存在时 docker 会照起，然后引擎在 900 秒超时里以"服务未就绪"
#      的面相报错——归因错且白等，所以存在性判据要在本地就把这一格堵住。
#   B 只在同一次拼装产出上做 grep -n，绝不拼两次输出再拼接；--order-file 复用同一个 assert_order，
#      夹具＝--dry-run 输出人工改序，因此测的是产物而非另一份实现。
#   C4..C7 的"计数==0"判据前一律先挂"日志字节>0"存在性断言（judge_log_present）。
#
# 阶段一／阶段二分界：阶段一只执行 --dry-run / --order-file / SM75_SELFCTEST 三条路径（本任务落码即
#   验证到位）。实跑分支（L0 之后）在阶段一用替身跑通逻辑：docker、curl、jq 三件全是 /tmp 沙箱里的桩，
#   证的是"判据链与取数接线正确"，不是"真引擎给的东西对"——真容器、真 HTTP、jq 或 python3 的真解析、
#   鉴权配置文件的权限位（本机 MSYS 不还原 POSIX 权限位，stat 报 644 是假象）都属阶段二待验，
#   逐条列为门禁项写进报告。本机 python3 是 WindowsApps 存根（rc=0 但不输出，实测），不可当解析器用。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SELF=${0##*/}
ENV_FILE_DEFAULT=$SCRIPT_DIR/env-paste-v1.txt
ARGS_FILE_DEFAULT=$SCRIPT_DIR/args-paste-videooff-v1.txt
# 镜像默认 tag 取自仓内权威登记：gpustack/backend-version-card-v1.md 第 13 行「镜像名称 image_name」。
IMAGE_DEFAULT=vllm-sm75-gpustack:v0.1.7-beta2-v7
MIN_BLOCK_LINES=24            # 粘贴块下限：args 24 行／env 25 行，取较小者 24（精确行数由 lint 的 L4／L11 钉）
PCIE_KEY=VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC
SPEC_FLAG=--speculative-config
REPO_MARKER_GIT=.git
REPO_MARKER_TABLE=gpustack/SHA256SUMS-upstream.txt
WAIT_SECS=900                 # 实跑等 /health 的上限（8 卡 fp16 262K 档加载慢，阶段二再校）
PORT_MIN=1
PORT_MAX=65535
NL='
'                             # argv 一行一词的分隔符；写成引号内真换行（$() 会把尾部换行剥空）

MODE=''
NEG_PCIE=0
MOUNT_PARENT=0                # 0＝与平台同宽（挂模型目录本身）；1＝宽一级挂父目录，默认关
TAIL_HOST=127.0.0.1           # 访问地址（curl 用）＋端口映射绑定地址；默认 127.0.0.1，--host 可覆写走平台 worker IP 形
BIND_HOST=0.0.0.0             # 容器内 vLLM 绑定地址（bridge 形态必须 0.0.0.0，映射才可达）
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
  echo "　　　　　　　　　　　　　　　[--neg-pcieipc] [--mount-parent] [--host VALUE] [--image TAG] [--env-file F] [--args-file F] [--out LOG]"
  echo "说明：--dry-run＝只打印拼装命令（不碰容器）；--order-file＝对一份拼装产物文本跑顺序断言；"
  echo "　　　两者都不给＝实跑模式（起容器＋运行期断言，属阶段二，需 SM75_VERIFY_LIVE_OK=1）。"
  echo "　　　--mount-parent＝挂载根宽一级到模型目录的父目录；默认与 GPUStack 平台同宽（挂模型目录本身）。"
  echo "　　　鉴权键只走环境变量 SM75_VERIFY_API_KEY，实跑写进 0600 的 curl 配置文件，不进命令行。"
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

sq() { # sq 词 —— 单引号包裹（内部单引号转义为 '\''），结果写进 _SQ 而不是走 stdout
  # 不走命令替换：本机（MSYS bash）每次 $(...) 都要 fork，dry-run 一条命令原先因此要 fork 约 150 次、
  # 耗 20 秒以上。写全局后这条热路径不再产生子 shell。结构自检的判据不变＝"会 die 的函数绝不出现在
  # $(...) 里"（print_plan 与 sq 都不 die，允许被命令替换取值）。
  local s=${1-}
  s=${s//\'/\'\\\'\'}
  _SQ="'$s'"
  return 0
}

# 下面这几个函数会 die，因此一律"把结果写进 _RES、由调用方在父 shell 里赋值"。
# 原因（首轮实测踩到）：函数被 $(...) 命令替换调用时身处子 shell，die 的 exit 只退子 shell，
# 父脚本会带着空返回值继续往下跑——守卫等于没装。调用形态必须是 `f a b; X=$_RES`。
_RES=''
_SQ=''
_MARKER=''

count_of() { # count_of 文本 固定行 —— 整行全等命中数写进 _RES
  local n
  if [ -z "${2-}" ]; then _RES=0; return 0; fi
  n=$(printf '%s\n' "$1" | grep -cxF -- "$2") || true
  [ -n "$n" ] || n=NOT_DIGIT
  _RES=$n
}

line_no_of() { # line_no_of 文本 固定行 —— 一次 awk 同时给命中数与首命中行号；恰 1 命中则 _RES=行号返回 0
  local out c l
  # 固定行走 ENVIRON 而不是 -v：-v 的值要做转义处理，args 行尾那个反斜杠会被吃掉。
  out=$(printf '%s\n' "$1" | PAT="$2" awk '$0==ENVIRON["PAT"] {c++; if(c==1) l=NR} END{printf "%d %d\n", c+0, l+0}') \
    || die 2 "错误：行号提取（awk）失败，固定行「$2」的判据不可信（退出码 2）。"
  c=${out% *}; l=${out#* }
  num "$c" "固定行「$2」命中数"
  num "$l" "固定行「$2」行号"
  if [ "$c" -ne 1 ]; then _RES=$c; return 1; fi
  _RES=$l
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
    *) case $abs in
         /) _RES=/$b ;;      # 目标就在文件系统根下时不拼双斜杠（理由见 hit_repo_walk）
         *) _RES=$abs/$b ;;
       esac ;;
  esac
}

physpath() { # physpath 路径 名称 —— 物理绝对化（pwd -P，穿过符号链接/目录联接/挂载别名）写进 _RES
  local d b abs
  d=$(dirname -- "$1")
  b=$(basename -- "$1")
  abs=$(cd -- "$d" 2>/dev/null && pwd -P) \
    || die 2 "错误：$2 的所在目录不存在或不可进入（物理途）：$d"
  case "$b" in
    .) _RES=$abs ;;
    *) case $abs in
         /) _RES=/$b ;;      # 同上：物理根下不拼双斜杠
         *) _RES=$abs/$b ;;
       esac ;;
  esac
}

hit_repo_walk() { # hit_repo_walk 单一路径 —— 从该路径向上遍历仓标记；命中则 _RES=仓根、_MARKER=命中的标记，返回 0
  local cur marker parent probe
  cur=$1
  while :; do
    for marker in "$REPO_MARKER_GIT" "$REPO_MARKER_TABLE"; do
      # 走到文件系统根时不得拼成「//标记」：MSYS 把开头的双斜杠当 UNC（\\主机\共享）去查网络，
      # 实测偶发返回"存在"，守卫于是把一切 /tmp 落点判成仓内（本机 8 次里中 1 次）；POSIX 下
      # 开头的 // 也属实现自定义，同样不该出现。所以根这一级单独拼成「/标记」。
      case $cur in
        /) probe=/$marker ;;
        *) probe=$cur/$marker ;;
      esac
      if [ -e "$probe" ]; then _RES=$cur; _MARKER=$marker; return 0; fi
    done
    parent=$(dirname -- "$cur")
    [ "$parent" = "$cur" ] && break
    cur=$parent
  done
  return 1
}

hit_repo_upwards() { # hit_repo_upwards 路径 —— 词法＋物理双途各走一遍，任一途命中仓内即返回 0（_RES＝仓根，_MARKER＝命中的标记）
  local start=$1 phys base
  _RES=''
  _MARKER=''
  base=${start##*/}
  phys=$(cd -- "$(dirname -- "$start")" 2>/dev/null && pwd -P) || phys=''
  if [ -n "$phys" ]; then
    case $phys in            # 物理根就是 /，同样不拼双斜杠（理由见 hit_repo_walk）
      /) phys=/$base ;;
      *) phys=$phys/$base ;;
    esac
  fi
  if hit_repo_walk "$start"; then return 0; fi
  if [ -n "$phys" ] && [ "$phys" != "$start" ] && hit_repo_walk "$phys"; then return 0; fi
  return 1
}

forbid_in_mount() { # forbid_in_mount 落点 挂载根 途组合 —— 落点位于挂载根之下即红（①的单个组合）
  case "$1/" in
    "$2"/*) die 2 "错误：--out 落进容器内挂载点（退出码 2）：落点 $1 位于挂载根 $2 之下（途组合＝$3）——该目录以同路径 bind 进容器（定义 base.py:604，生效 :617 取 dirname(model_path)、:618-622 挂盘），引擎进程会读写自己的日志落点。请改指仓外目录。" ;;
  esac
  return 0
}

# ---------- P3 落点守卫 ----------
guard_out_target() { # guard_out_target 落点词法 落点物理 挂载根词法 挂载根物理 —— 三条边界逐条判死，红则零落盘
  local lex=$1 phys=$2 mlex=$3 mphys=$4 root shown
  has_dotdot "$lex" && die 2 "错误：--out 含 .. 路径段，无法可靠判定真实落点，本脚本拒绝猜测（退出码 2）：$lex"
  # ①两途对两途，四个组合任一命中即红：只比词法时，同一目录换个名字（挂载别名、符号链接、
  #   目录联接）就能绕过去，而 docker 在宿主上认的是解析后的真实目录。
  forbid_in_mount "$lex" "$mlex" "词法对词法"
  forbid_in_mount "$lex" "$mphys" "词法对物理"
  forbid_in_mount "$phys" "$mlex" "物理对词法"
  forbid_in_mount "$phys" "$mphys" "物理对物理"
  if hit_repo_upwards "$lex"; then
    root=$_RES
    shown=${lex#"$root"/}
    die 2 "错误：--out 落点在仓内（退出码 2）：解析为 $lex，位于仓根 $root 之下，命中标记＝$_MARKER（两条途各走一遍：词法途 cd&&pwd 与物理途 cd&&pwd -P，向上遍历 .git 与 gpustack/SHA256SUMS-upstream.txt）。启动日志属运行期证据，任何仓内落点都会污染工作树，请改指仓外（如 /tmp）。落点相对仓根＝$shown。若 $root 看着不是你的仓（例如它是文件系统根），说明这台机上 $root/$_MARKER 是别处留下的痕迹，请先核清再跑——本守卫宁可错拦也不往疑似仓里写。"
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
    || die 2 "错误：$2 行数 $lines 低于下限 $MIN_BLOCK_LINES（简报契约：args 24 行、env 25 行）：$1"
  cr=$(tr -dc '\r' < "$1" | wc -c | tr -d '[:space:]')
  num "$cr" "$2 CR 字节数"
  [ "$cr" -eq 0 ] \
    || die 2 "错误：$2 含 $cr 个 CR 字节（Windows 行尾）：$1 —— CR 会跟着 flag 一起进 argv，肉眼看不出来，引擎侧报 unrecognized arguments。请用 LF 检出：git -c core.autocrlf=false clone"
  _RES=$lines
}

# ---------- P1b 模型目录前置守卫 ----------
precheck_model_dir() { # precheck_model_dir —— 目录存在性与可读性；缺件时必须红，别把首跑交给 900 秒超时
  [ -d "$MODEL_DIR" ] \
    || die 2 "错误：--model-dir 不是已存在的目录：$MODEL_DIR（本脚本不代建目录）。模型路径不在盘上时 docker 会照常起容器，然后引擎在等待上限 ${WAIT_SECS}s 里以『服务未就绪』的面相报错——归因错且白等一整轮，所以存在性判据必须在本地就把这一格堵住。"
  [ -r "$MODEL_DIR" ] || die 2 "错误：--model-dir 不可读：$MODEL_DIR"
  [ -d "$MODEL_PATH" ] \
    || die 2 "错误：模型路径不存在或不是目录：$MODEL_PATH（GPUStack 的 model_path 是宿主上真实存在的目录，容器按同路径 bind，定义 base.py:604、生效 :617/:618-622）。"
  [ -r "$MODEL_PATH" ] || die 2 "错误：模型路径不可读：$MODEL_PATH"
  _RES=$MODEL_PATH
}

# ---------- A 拼装（argv 一行一词；执行时逐词装进位置参数，不经 shell）----------
ARGV_TEXT=''

argv_push() { # argv_push 词… —— 追加进 ARGV_TEXT；空词与含换行的词先判死（承载形的前提）
  local w
  for w in "$@"; do
    [ -n "$w" ] || die 2 "错误：拼装出现空词，一行一词的承载形会把它静默丢掉（退出码 2）——argv 少一个词等于少一个参数，后果是引擎取错值。"
    case $w in
      *"$NL"*) die 2 "错误：拼装词内含换行，一行一词的承载形不成立，拒绝拼装（退出码 2）：$w" ;;
    esac
    ARGV_TEXT="${ARGV_TEXT:+$ARGV_TEXT$NL}$w"
  done
}

build_argv() {
  local line
  # 不带 --rm：GPUStack v2.2.3 副本里 auto_remove 全树 0 命中（平台不传该字段，本机无法取证它来自谁），
  # 而 --rm 会在容器崩溃时连日志一起收走——C4..C7 与超时归因就只剩 docker 的"No such container"。
  # 默认 bridge ＋ 只映射一个引擎端口（绑 $TAIL_HOST，最小暴露）；容器内 vLLM 绑 BIND_HOST。
  argv_push run -d --name "$CONTAINER_NAME" -p "$TAIL_HOST:$PORT:$PORT" --privileged --restart no --shm-size 64g
  argv_push --env-file "$ENV_FILE"
  if [ "$NEG_PCIE" -eq 1 ]; then
    argv_push -e "$PCIE_KEY=0"
  fi
  argv_push -v "$MOUNT_ROOT:$MOUNT_ROOT" "$IMAGE" vllm serve "$MODEL_PATH"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    argv_push "$line"
  done < "$ARGS_FILE"
  argv_push --host "$BIND_HOST" --port "$PORT" --served-model-name "$SERVED_NAME"
}

judge_tail_shape() { # judge_tail_shape —— 尾补三对必须逐字在位且只占末六词（词数被改动即红 2）
  local want tail
  want=$(printf '%s\n' --host "$TAIL_HOST" --port "$PORT" --served-model-name "$SERVED_NAME" | tr '\n' '|')
  tail=$(printf '%s\n' "$ARGV_TEXT" | tail -n 6 | tr '\n' '|')
  [ "$tail" = "$want" ] \
    || die 2 "错误：尾补形状红（退出码 2）——期望末六词＝$want，实为 $tail。末六词一变就说明有参数被塞进尾补位置或 argv 词数被改动，同名参数会被后值吃掉，正是顺序判据要防的那类漂移。"
}

print_plan() { # 逐词单引号、每词一行、除末词外带续行符——保证整段可直接粘贴执行
  local i=0 w
  printf 'docker'
  printf '%s\n' "$ARGV_TEXT" | while IFS= read -r w; do
    sq "$w"
    if [ "$i" -eq 0 ]; then
      printf ' %s' "$_SQ"
    else
      printf ' \\\n  %s' "$_SQ"
    fi
    i=$((i + 1))
  done
  printf '\n'
}

LN_SERVE=''
LN_MODEL=''
LN_HOST=''
LN_PORT=''
LN_NAME=''

order_slot() { # order_slot 槽名 固定行 —— 恰 1 命中则把行号写进 _RES，否则红；空串绝不出栈
  local key=$1 val=$2
  line_no_of "$ORDER_TEXT" "$val" \
    || die 2 "错误：拼装产物里「$key」命中 $_RES 次（应为恰 1 次），该文本不是合法拼装产物，顺序判据不予执行（退出码 2）。"
  num "$_RES" "槽位「$key」行号"
  return 0
}

# ---------- B 顺序断言（28）：在同一份拼装文本上取五个槽位行号，缺环即红 ----------
# 五个槽位各用一条显式判据取行号，不走 `*port*` 这类通配分派：通配分派的失效形态是"改个文案或
# 加个分组，某一环静默拿到空串，空串参与 -ge 比较时报错被 if 判假"——这一环等于没判还不报错。
# 现在槽位取不到、行号非纯数字、或五环没凑齐，都各自落红（退 2＝拼装产物形态异常档）。
assert_order() { # assert_order 文本 —— 三段行号失衡即 28
  ORDER_TEXT=$1
  local slots=0 line n min_user='' max_user=''
  order_slot "serve 基准行" "  'serve' \\"; LN_SERVE=$_RES; slots=$((slots + 1))
  sq "$MODEL_PATH"; order_slot "模型路径行" "  $_SQ \\"; LN_MODEL=$_RES; slots=$((slots + 1))
  order_slot "尾补 --host 行" "  '--host' \\"; LN_HOST=$_RES; slots=$((slots + 1))
  order_slot "尾补 --port 行" "  '--port' \\"; LN_PORT=$_RES; slots=$((slots + 1))
  order_slot "尾补 --served-model-name 行" "  '--served-model-name' \\"; LN_NAME=$_RES; slots=$((slots + 1))
  [ "$slots" -eq 5 ] \
    || die 2 "错误：顺序判据缺环（退出码 2）——期望取到 5 个槽位行号，实取 $slots 个。缺环的那一环不参与比较，整条链却会照常打印『顺序正确』，所以缺环本身就是红。"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    sq "$line"
    line_no_of "$ORDER_TEXT" "  $_SQ \\" \
      || die 2 "错误：args 参数行「$line」在拼装产物里命中 $_RES 次（应为恰 1 次），无法逐行定序（退出码 2）。"
    n=$_RES
    num "$n" "用户参数行「$line」行号"
    if [ -z "$min_user" ]; then min_user=$n; max_user=$n; continue; fi
    [ "$n" -lt "$min_user" ] && min_user=$n
    [ "$n" -gt "$max_user" ] && max_user=$n
  done < "$ARGS_FILE"
  [ -n "$min_user" ] || die 2 "错误：args 粘贴块里没有可定序的参数行（空块不该走到这里，P1 应已判红）。"
  CHAIN_STATE="serve=$LN_SERVE 模型=$LN_MODEL 用户参数=[$min_user,$max_user] host=$LN_HOST port=$LN_PORT name=$LN_NAME"
  order_link "$LN_SERVE" "$LN_MODEL" "'vllm serve' 必须早于模型路径"
  order_link "$LN_MODEL" "$min_user" "模型路径必须早于任一用户参数（vllm.py:613 生效行把它放在 arguments 首位）"
  order_link "$max_user" "$LN_HOST" "全部用户参数必须早于尾补三条（vllm.py:651/652 放用户参数，:654-659 才尾补）"
  order_link "$LN_HOST" "$LN_PORT" "尾补 --host 必须早于 --port"
  order_link "$LN_PORT" "$LN_NAME" "尾补 --port 必须早于 --served-model-name"
  echo "绿｜B 顺序断言｜serve=$LN_SERVE < 模型路径=$LN_MODEL < 用户参数[$min_user,$max_user] < host=$LN_HOST < port=$LN_PORT < name=$LN_NAME" >&2
}

order_link() { # order_link 左行号 右行号 为什么 —— 两侧先过纯数字哨兵，左>=右即 28
  local a=$1 b=$2 why=$3
  num "$a" "顺序链左端（$why）"
  num "$b" "顺序链右端（$why）"
  [ "$a" -lt "$b" ] || die 28 "断言 B 顺序红——$why（实测行号 $a >= $b；全链 $CHAIN_STATE）。错位会让引擎按后值取同名参数，表现为档位漂移而不报错。"
}

# ---------- C5a 自拼装 argv 面（本地干跑即可命中）----------
judge_argv_no_speculative() { # judge_argv_no_speculative 拼装文本 —— 拼装命令行里 $SPEC_FLAG 须恰 0 次
  local text=$1 hits
  [ -n "$text" ] || die 2 "错误：拼装产物为空，C5a 的『0 次』不构成任何证明（退出码 2）。"
  hits=$(printf '%s\n' "$text" | grep -cF -- "$SPEC_FLAG") || true
  num "$hits" "拼装产物里 $SPEC_FLAG 命中数"
  [ "$hits" -eq 0 ] || die 25 "断言 C5a 投机红——自拼装 argv 面出现 $SPEC_FLAG 共 $hits 次（本线 MTP 未启用，用户 2026-09-13 与 2026-10-11 两次裁决；args 粘贴块里混进投机参数＝档位被换掉）。命中的行：
$(printf '%s\n' "$text" | grep -nF -- "$SPEC_FLAG" | head -n 5)"
  echo "绿｜C5a 拼装 argv 面｜$SPEC_FLAG 恰 0 次" >&2
}

# ---------- 运行期判据（C1..C7）：与取数解耦，便于阶段一用正例＋反例证红 ----------
# C1 判据（2026-10-11 主控裁决增补二第 1 条改契约）：vLLM 原生 /health 就是 200＋空 body，
# 空 body 合法、不得判负（旧判据是简报对 10-10 教训的误引）；"必须真有内容"的证据面在
# C3（清单回显 served-model-name）与 C2（一次对话的 content 非空）。退 21 语义＝"非 200 或未就绪"。
judge_health() { # judge_health HTTP码 body —— 恰 200 才绿；200 时 body 允许为空
  local code=$1 body=$2 bytes
  [ "$code" = 200 ] || die 21 "断言 C1 /health 红——HTTP 码 ${code}（判据＝恰 200），服务未就绪或端口不符。"
  bytes=$(printf '%s' "$body" | wc -c | tr -d '[:space:]')
  num "$bytes" "/health body 字节数"
  echo "绿｜C1 /health｜HTTP $code，body $bytes 字节（空 body 合法；内容证据面见 C2/C3）"
}

judge_chat_content() { # judge_chat_content content body —— content 缺失、为空或只有空白即 22
  local content=$1 body=$2 visible
  visible=$(printf '%s' "$content" | tr -d '[:space:]' | wc -c | tr -d '[:space:]')
  num "$visible" "chat content 非空白字节数"
  [ "$visible" -gt 0 ] || die 22 "断言 C2 chat completion 红——choices[0].message.content 缺失、为空或只含空白字符（实测非空白字节 $visible；只含空白＝模型没吐出可用回答，按 -n 判空会放过去）。响应片段：$(printf '%.400s' "$body")"
  echo "绿｜C2 chat completion｜content 非空白字节 $visible"
}

judge_models() { # judge_models HTTP码 body served-name —— 取数面与回显面分开报，都退 23
  local code=$1 body=$2 name=$3 bytes
  case $code in
    2??) ;; *) die 23 "断言 C3 模型清单红——HTTP 码 ${code}，取数面失败（这不能算『未注册』，两者文案必须分开）。引擎自带鉴权键时检查 SM75_VERIFY_API_KEY 是否给出。响应片段：$(printf '%.400s' "$body")" ;;
  esac
  bytes=$(printf '%s' "$body" | wc -c | tr -d '[:space:]')
  num "$bytes" "模型清单 body 字节数"
  [ "$bytes" -gt 0 ] || die 23 "断言 C3 模型清单红——HTTP $code 但 body 为空（0 字节）：『未回显』与『没取到』无法区分，存在性断言先红。"
  printf '%s' "$body" | grep -qF -- "$name" \
    || die 23 "断言 C3 模型清单红——未回显 served-model-name「$name」。响应片段：$(printf '%.400s' "$body")"
  echo "绿｜C3 模型清单｜HTTP $code，回显 served-model-name=$name"
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

# C5 的两个日志面不可互替（主控裁决增补二第 2 条）：
#   C5a＝自拼装 argv 面（judge_argv_no_speculative，本地 dry-run 就能命中）；
#   C5b＝本容器启动日志面（下面这个函数，实跑时 docker logs 就是本形态的 argv 回显，权威）；
#   第三物＝GPUStack 平台注入参数日志（worker 侧 vllm.py:331-338 的 logger.info），
#   本机拿不到，留阶段二门禁 G-3——三条各判各的，谁也不给谁作证。
judge_log_no_speculative() { # 25 —— C5b 本容器启动日志面
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -cF -- "$SPEC_FLAG" "$f") || true
  num "$hits" "$SPEC_FLAG 命中数"
  [ "$hits" -eq 0 ] || die 25 "断言 C5b 投机红——本容器启动日志出现 $SPEC_FLAG 共 $hits 次（本线 MTP 未启用，用户 2026-09-13 与 2026-10-11 两次裁决；平台注入位排在用户参数之前，定义 vllm.py:531、生效 :627，重复参数被后值吃掉）：
$(grep -nF -- "$SPEC_FLAG" "$f" | head -n 5)"
  echo "绿｜C5b 投机｜本容器启动日志里 $SPEC_FLAG 恰 0 次"
}

judge_log_no_ray() { # 26
  local f=$1 hits
  judge_log_present "$f"
  hits=$(grep -Ec -- 'distributed-executor-backend[= ]+ray|raylet|RayDistributedExecutor|Ray Client' "$f") || true
  num "$hits" "ray 执行器标记命中数"
  [ "$hits" -eq 0 ] || die 26 "断言 C6 ray 红——启动日志出现 ray 执行器注入共 $hits 次（版本名缺 -custom 后缀即默认回落 Ray，标准镜像无 Ray 必失败；定义 utils/command.py:416 should_default_to_ray 与 :436 resolve_executor_backend，生效 :429-433 与 :455）：
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

# ---------- 判据自测（SM75_SELFCTEST=1）：每条判据各一正一反，反例必须命中自己的码 ----------
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

  run_case ok      "C1 正例 /health 恰 200＋JSON" judge_health 200 '{"status":"ok"}'
  run_case ok      "C1 样张① 恰 200＋空 body 必须绿（改契约防改过头）" judge_health 200 ''
  run_case ok      "C1 正例 恰 200＋纯文本 body（内容证据面不在 C1）" judge_health 200 'health ok'
  run_case 21      "C1 反例 HTTP 503" judge_health 503 '{"status":"unavailable"}'
  run_case 21      "C1 反例 HTTP 201（旧判据按 2xx 放行，新契约要恰 200）" judge_health 201 '{"status":"ok"}'
  run_case ok      "C2 正例 content 非空" judge_chat_content '好的' '{"choices":[{"message":{"content":"好的"}}]}'
  run_case 22      "C2 反例 content 为空" judge_chat_content '' '{"choices":[{"message":{"content":""}}]}'
  run_case 22      "C2 反例 content 只有空白（旧 -n 判空会放行）" judge_chat_content ' 	' '{"choices":[{"message":{"content":"  "}}]}'
  run_case ok      "C3 正例 HTTP 200＋清单回显" judge_models 200 '{"data":[{"id":"qwen38-next","object":"model"}]}' qwen38-next
  run_case 23      "C3 反例 200 但未回显" judge_models 200 '{"data":[{"id":"other-model"}]}' qwen38-next
  run_case 23      "C3 反例 200＋空 body（存在性先红）" judge_models 200 '' qwen38-next
  run_case 23      "C3 样张② 401 取数面失败（文案不得讲成『未注册』）" judge_models 401 '{"error":{"type":"authentication_error","message":"invalid api key"}}' qwen38-next
  run_case ok      "C4 正例 日志无占位符" judge_log_no_placeholder "$dir/good.log"
  run_case 2       "C4 反例 日志为空（存在性断言先红）" judge_log_present "$dir/empty.log"
  run_case 24      "C4 反例 日志含字面 \${" judge_log_no_placeholder "$dir/placeholder.log"
  run_case ok      "C5a 正例 拼装 argv 无投机参数" judge_argv_no_speculative "  '--dtype=float16' \\
  '--max-model-len=262144' \\"
  run_case 25      "C5a 反例 拼装 argv 含 --speculative-config" judge_argv_no_speculative "  '--speculative-config={\"method\":\"mtp\"}' \\
  '--dtype=float16' \\"
  run_case 2       "C5a 反例 拼装文本为空（存在性先红）" judge_argv_no_speculative ''
  run_case ok      "C5b 正例 日志无投机" judge_log_no_speculative "$dir/good.log"
  run_case 25      "C5b 反例 日志含 --speculative-config" judge_log_no_speculative "$dir/spec.log"
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
  echo "SELFTEST_OK 判据自测全部通过（26 条用例；原 20 条中 C1 的两条按 2026-10-11 新契约改判、C3 的三条随签名加 HTTP 码，新增 C1×2、C2×1、C3×1、C5a×3）"
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
    --mount-parent) MOUNT_PARENT=1; shift ;;
    --host)
      [ $# -ge 2 ] && [ -n "${2:-}" ] || die 2 "错误：--host 缺少值。"
      TAIL_HOST=$2; shift 2 ;;
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
  0) die 2 "错误：--port 不得为 0（0 不是可监听端口，curl 会直接拒 URL）。" ;;
  0*) die 2 "错误：--port 不得带前导零（超宽写法与引擎侧取值不同形，且 0070000 这类串看着合法实则越界）。" ;;
esac
num "$PORT" "--port"
{ [ "$PORT" -ge "$PORT_MIN" ] && [ "$PORT" -le "$PORT_MAX" ]; } \
  || die 2 "错误：--port 不在 $PORT_MIN-$PORT_MAX 区间，实为「$PORT」。越界值会让 curl 直接拒 URL，而取数失败被等待循环吞成『还没就绪』——一条两秒的用法错会被诊断成等满 ${WAIT_SECS}s 的服务未就绪。"
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
# 尾补 --host 的取值校验（只走码 2＝用法/前置守卫档，不新开码位）。空值在参数解析处已被拦
# （`--host` 必须带非空值），这里再拦两类注入形：以短横开头（会被引擎当成另一个 flag，等于往尾补
# 位置塞参数）、含空白/制表/换行（一个 argv 词被拆成多个）。词数是否真的没变，由 judge_tail_shape
# 在拼装之后再判一次，不靠这里推断。
case $TAIL_HOST in
  -*) die 2 "错误：--host 的值不得以短横开头（会被 vllm 解析成另一个参数，尾补位置就被顶掉了），实为「$TAIL_HOST」。" ;;
esac
case $TAIL_HOST in
  *[[:space:]]*) die 2 "错误：--host 的值不得含空白、制表或换行（一个 argv 词会被拆成多个），实为「$TAIL_HOST」。" ;;
esac
[ -n "$IMAGE" ] || die 2 "错误：--image 不得为空。"

# 工具哨兵：所有判据都来自这些工具的 stdout，而空串恰是各道的"无问题"值，缺件必须先把门。
# 名单按实际调用列（M-6 顺带收口：原名单里的 sed/sort/uniq/cut 本脚本并不调用，虚胖会让"缺件"归因跑偏；
# cut 是本轮把 grep+cut 两跳改成一次 awk 之后才变多余的）。
for needed_tool in grep awk wc tr head basename dirname printf; do
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
# 挂载根：默认与平台同宽。base.py 定义 :604、生效 :617 取 dirname(model_path)、:618-622 挂盘；
# 本脚本的 MODEL_PATH＝MODEL_DIR/MODEL，所以 dirname(MODEL_PATH) 恰是 MODEL_DIR 本身。
# --mount-parent 才宽一级挂父目录（默认关）：宽一级在多露一层宿主目录，--privileged 下是净增攻击面。
if [ "$MOUNT_PARENT" -eq 1 ]; then
  MOUNT_ROOT=$(dirname -- "$MODEL_DIR")
else
  MOUNT_ROOT=$MODEL_DIR
fi
CONTAINER_NAME=sm75-verify-$SERVED_NAME

# ---------- P1 依赖件存在性（排在一切拼装与写盘之前）----------
precheck_block "$ENV_FILE" "env 粘贴块"; ENV_LINES=$_RES
precheck_block "$ARGS_FILE" "args 粘贴块"; ARGS_LINES=$_RES

# ---------- P1b 模型目录存在性（起容器之前就堵，别把首跑交给等待上限）----------
precheck_model_dir

# ---------- P2 双保险：args 内不得含尾补三条 ----------
# 判据先剥行首空白、再认 `=`／空白／行尾三种分隔：锚在行首的旧判据被 `   --host=x` 一行三个空格
# 整体绕过，而 --args-file 是公开入口，单用本脚本即可绕过——本脚本必须独立成防线，
# 不得以"任务 7 的 L5 也会拦"为理由放行（组合面依赖是另一件的判据，换档就失效）。
dup_tail=$(grep -E '^[[:space:]]*--(host|port|served-model-name)([[:space:]]|=|$)' "$ARGS_FILE") || true
if [ -n "$dup_tail" ]; then
  die 2 "错误：args 粘贴块含 GPUStack/本脚本尾补的三参数之一（退出码 2＝前置守卫档）：
$dup_tail
这三项由本脚本的尾补三条提供（vllm.py 生效段 :654-659），重复出现会让命令行出现两份同名参数，后者生效但语义不可预期。任务 7 的 lint-paste-v1.sh 在同一判据上用码 14，两处互为双保险，但本判据不依赖它。"
fi

# ---------- P3 落点守卫 ----------
OUT_FILE=${OUT_FILE_IN:-}
if [ -z "$OUT_FILE" ]; then
  OUT_DIR=$(cd -- "${TMPDIR:-/tmp}" 2>/dev/null && pwd) \
    || die 2 "错误：默认日志落点不可用（TMPDIR 与 /tmp 都不可进入），请显式给 --out 指向仓外已存在目录。"
  OUT_FILE=$OUT_DIR/sm75-gpustack-form-verify-v1-bootlog.txt
fi
# .. 段在解析成绝对路径之前就判死：abspath/physpath 都会把 `..` 吃掉，事后无从还原使用者的猜测。
has_dotdot "$OUT_FILE" \
  && die 2 "错误：--out 含 .. 路径段，无法可靠判定真实落点，本脚本拒绝猜测（退出码 2）：$OUT_FILE"
abspath "$OUT_FILE" "--out"; OUT_LEX=$_RES
physpath "$OUT_FILE" "--out"; OUT_PHYS=$_RES
abspath "$MOUNT_ROOT" "--mount"; MOUNT_LEX=$_RES
physpath "$MOUNT_ROOT" "--mount"; MOUNT_PHYS=$_RES
guard_out_target "$OUT_LEX" "$OUT_PHYS" "$MOUNT_LEX" "$MOUNT_PHYS"; OUT_FILE=$_RES

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

# ---------- A 拼装 + C5a + B 顺序断言（同一份输出）----------
build_argv
[ -n "$ARGV_TEXT" ] || die 2 "错误：拼装词表为空（argv_push 一个词都没追加）。"
judge_tail_shape
PLAN=$(print_plan)
[ -n "$PLAN" ] || die 2 "错误：拼装产物为空。"
if [ "$NEG_PCIE" -eq 1 ]; then
  # grep 是单行判据，所以逐词两行各自验恰 1 命中（-e 与它的值在打印形态里分占两行）。
  sq -e; count_of "$PLAN" "  $_SQ \\"; num "$_RES" "-e 行命中数"; n_e=$_RES
  sq "$PCIE_KEY=0"; count_of "$PLAN" "  $_SQ \\"; num "$_RES" "pcie 值行命中数"; n_v=$_RES
  { [ "$n_e" -eq 1 ] && [ "$n_v" -eq 1 ]; } \
    || die 2 "错误：负控侧效未进命令行（退出码 2）——拼装产物里 '-e' 行命中 $n_e 次、'$PCIE_KEY=0' 行命中 $n_v 次，两者均应恰 1 次。开关没落到命令上，负控结论不可信。"
fi

# ---------- C5a 自拼装 argv 面（dry／order／live 三条路径都过这一道）----------
# 排在 B 之前：这一道只依赖拼装文本，先用它把"拼装里混进投机参数"钉成 25，
# 改序样张（不含投机参数）照样落到 B 的 28——两道各用不重叠的夹具，互不抢走。
judge_argv_no_speculative "$PLAN"

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
  printf '%s\n' "# 挂载宽度＝与平台同宽（挂模型目录本身，定义 base.py:604、生效 :617/:618-622）；宽一级须显式 --mount-parent"
  printf '%s\n' "# 尾补 --host 取 $TAIL_HOST（默认 127.0.0.1＝简报契约；平台尾补取 worker IP，vllm.py:656，本地通过≠平台形态通过＝阶段二门禁 G-2，用 --host 覆写）"
  printf '%s\n' "# 不带 --rm：GPUStack v2.2.3 副本内 auto_remove 全树 0 命中，且 --rm 会在崩溃时连日志一起收走"
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

# ---------- L1 实跑加严面（dry-run 用空夹具目录合法，真起容器不行）----------
[ -f "$MODEL_PATH/config.json" ] \
  || die 2 "错误：实跑要求模型路径含 config.json，实测不存在：$MODEL_PATH/config.json —— 没有它引擎必起不来，而那时等待循环报的是『服务未就绪』，归因会跑偏到显存或端口上去。"

# 观测行（不作判据、不新增码位）：面板改名与"同镜像多容器"之后，第一次上机要先看见同台有哪些容器。
# docker ps 没有 {{.Created}} 字段；这里只取 .Names／.Status，要时间戳得走 docker inspect（回 UTC，面板显示 CST 恒差 8 小时）。
ps_same=$(docker ps -a --filter "ancestor=$IMAGE" --format '{{.Names}} {{.Status}}' 2>/dev/null) || ps_same=''
if [ -n "$ps_same" ]; then
  printf '%s\n' "$ps_same" | while IFS= read -r line; do printf 'INFO｜同台容器｜%s\n' "$line"; done
else
  echo "INFO｜同台容器｜无匹配或 docker 不可读（仅观测，不进判据链）"
fi

BODY_DIR=$(mktemp -d 2>/dev/null) || die 2 "错误：无法创建响应体临时目录（mktemp -d）。"
trap 'rm -rf -- "$BODY_DIR" 2>/dev/null' EXIT

same=$(docker ps -aq --filter "name=^${CONTAINER_NAME}$" | wc -l | tr -d '[:space:]')
num "$same" "同名容器数"
[ "$same" -eq 0 ] || die 2 "错误：同名容器已存在（$same 个）：$CONTAINER_NAME —— 本脚本不做删除动作（破坏性操作需另行授权），请改 --name 或人工清理。"

if ! command -v jq >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  die 2 "错误：解析 chat completion 的 content 需要 jq 或 python3，二者皆缺。"
fi

# ---------- S-1 鉴权键：只进 0600 的 curl 配置文件，绝不进命令行 ----------
# 旧形态 `-H "Authorization: Bearer $API_KEY"` 把键挂在 argv 上，同机任何用户 `ps` 一眼可见——
# 这是对外交付物的攻击向量面，安传规范第 3 条要求它先行关闭。curl 的 -K/--config 是文件形态
# （本机 curl 8.21 实测：含 header 行的配置与纯注释配置都 rc=0），-q 不读 ~/.curlrc；
# 文件写在 BODY_DIR（mktemp -d 建的临时目录）里，退出 trap 随目录一起删除。
umask 077
AUTH_CFG=$BODY_DIR/curl-auth.cfg
API_KEY=${SM75_VERIFY_API_KEY:-}
if [ -n "$API_KEY" ]; then
  case $API_KEY in
    *[!A-Za-z0-9_.:-]*) die 2 "错误：SM75_VERIFY_API_KEY 含白名单外字符（只允许字母数字与 _ . : -）——引号或换行进了 curl 配置文件就变成配置注入面。" ;;
  esac
  printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" > "$AUTH_CFG" \
    || die 2 "错误：写鉴权配置文件失败：$AUTH_CFG"
else
  printf '# 未给出 SM75_VERIFY_API_KEY，本次不带鉴权头\n' > "$AUTH_CFG" \
    || die 2 "错误：写鉴权配置文件失败：$AUTH_CFG"
fi
cfg_bytes=$(wc -c < "$AUTH_CFG" | tr -d '[:space:]')
num "$cfg_bytes" "鉴权配置文件字节数"
[ "$cfg_bytes" -gt 0 ] \
  || die 2 "错误：鉴权配置文件为 0 字节：$AUTH_CFG —— curl -K 读到空文件＝静默降级成不带键，绿面会失去含义。"
# 权限位只作观测不作判据：本机（Git Bash/MSYS）不还原 POSIX 权限位，umask 077 造出的文件 stat 仍报
# 644（实测），拿它当门禁会假红；真判据是 Linux 侧 stat=600，属阶段二门禁 G-5。
cfg_mode=$(stat -c '%a' "$AUTH_CFG" 2>/dev/null || echo NA)
echo "INFO｜鉴权配置文件：$AUTH_CFG $cfg_bytes 字节，stat=$cfg_mode（Linux 侧应为 600；本机 MSYS 不还原权限位，此值不作判据）"

# ---------- C 起容器 + 运行期断言 ----------
# 把 ARGV_TEXT 逐词装进位置参数：IFS 只认换行、glob 先关（词里的 * 与 ? 不能被展开成文件名）。
set -f
OLD_IFS=$IFS
IFS=$NL
set --
for word in $ARGV_TEXT; do
  set -- "$@" "$word"
done
IFS=$OLD_IFS
set +f
[ "$#" -gt 0 ] || die 2 "错误：位置参数装载后为空，拼装词没落进 argv——拒绝把一条没有参数的 docker 命令交出去。"
docker "$@" || die 2 "错误：docker run 启动失败（原始报错见上，本脚本不改写 docker 的失败语义）。"
echo "INFO｜容器已启动：$CONTAINER_NAME，等待 /health 最多 ${WAIT_SECS}s（每 10s 一次）"
deadline=$((SECONDS + WAIT_SECS))
code=000
tick=0
running=true
while :; do
  tick=$((tick + 1))
  # C0 存活短路排在取数之前：容器启动即退出时 curl 永远等不到 200，旧循环会白等满 ${WAIT_SECS}s
  # 再以"服务未就绪"报错——第一次上机就会踩，耗时占满且归因错。
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null) || running=INSPECT_FAIL
  case $running in
    true) ;;
    *)
      docker logs "$CONTAINER_NAME" > "$OUT_FILE" 2>&1 || true
      log_bytes=$(wc -c < "$OUT_FILE" | tr -d '[:space:]')
      num "$log_bytes" "容器退出时的日志字节数"
      die 21 "断言 C1 提前红——第 $tick 格检查到容器已不在运行（docker inspect 回「$running」）。这不是慢，是启动即退出。日志已落 $OUT_FILE（$log_bytes 字节；若内容只是 docker 的『No such container』，说明容器压根没建起来）。按『引擎启动即退出』定位：先看日志末行，再核显存与端口。" ;;
  esac
  code=$(curl -q -sS -K "$AUTH_CFG" -o "$BODY_DIR/health" -w '%{http_code}' --max-time 10 "http://$TAIL_HOST:$PORT/health" 2>/dev/null) || code=000
  [ "$code" = 200 ] && break
  if [ "$SECONDS" -ge "$deadline" ]; then
    docker logs "$CONTAINER_NAME" > "$OUT_FILE" 2>&1 || true
    judge_log_present "$OUT_FILE"
    die 21 "断言 C1 /health 红——等待 ${WAIT_SECS}s（共 $tick 格）后仍未 200（最后一次 HTTP 码 $code，容器状态 $running）；日志已落 $OUT_FILE，先按日志定位。"
  fi
  sleep 10
done
judge_health "$code" "$(cat -- "$BODY_DIR/health")"

models_http=$(curl -q -sS -K "$AUTH_CFG" -o "$BODY_DIR/models" -w '%{http_code}' --max-time 60 "http://$TAIL_HOST:$PORT/v1/models" 2>/dev/null) || models_http=000
judge_models "$models_http" "$(cat -- "$BODY_DIR/models")" "$SERVED_NAME"

chat_body=$(curl -q -sS -K "$AUTH_CFG" --max-time 300 -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"只回复两个字：好的\"}],\"max_tokens\":8,\"temperature\":0}" \
  "http://$TAIL_HOST:$PORT/v1/chat/completions") || die 22 "错误：chat completion 请求失败（C2 的取数路径；鉴权头走 $AUTH_CFG，不经命令行）。"
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

echo "VERIFY_LIVE_OK name=$SERVED_NAME port=$PORT image=$IMAGE log=$OUT_FILE auth_cfg_bytes=$cfg_bytes key_in_argv=0"
exit 0
