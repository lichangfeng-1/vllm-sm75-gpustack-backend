#!/usr/bin/env bash
# build-gpustack-v1.sh —— 国内源补丁门 + GPUStack 交付线构建总入口
#
# 本任务（任务 2）只到"补丁门绿＋脚本静态过"，真构建在阶段二任务 14。
# 上游树按红线保持零改动：补丁只作用于暂存目录里的副本。
#
# 用法：
#   bash gpustack/build-gpustack-v1.sh                     补丁门（默认，不执行 docker build）
#   bash gpustack/build-gpustack-v1.sh --build             补丁门绿后组装并执行 docker build
#   选项：
#     --patch-file PATH   指定补丁文件（默认 本脚本目录/mirror-patch-v1.patch）
#     --stage-dir PATH    指定暂存目录（默认 $SM75_CN_STAGE_DIR 或 $TMPDIR/sm75-gpustack-cn）
#     --no-refresh        不重新复制仓内原件，直接在已有暂存副本上跑门（重复应用负控用）
#     -h | --help         显示用法
#
# 门顺序（简报逐步，不可换）：
#   1 复制仓内 docker/Dockerfile 到暂存目录
#   2 patch --dry-run 预检（hunk 不匹配 rc=1 → 8；patch 工具级错误 rc>=2 → 2）
#   3 实应用（同上分流）
#   4 特征计数门（少或多 → 8）
#   5 对仓内原 docker/Dockerfile 跑任务 1 的上游护栏（漂移红 7 → 本脚本 9）
#
# 退出码（本脚本）：
#   0  通过
#   2  用法错误 / 前置输入不可用（补丁缺失、含 CR、表头不符、缺工具、暂存副本不存在、
#      patch 工具级错误即返回码 >=2、上游护栏因非漂移原因红）
#   8  补丁不匹配（patch --dry-run 返回 1＝hunk 不匹配，或特征计数不符）
#   9  上游文件被改（gate-tree-unchanged-v1.sh 报漂移，其自身红码 7 在此映射为 9）
#   20 构建失败（--build 段 docker build 非零）。原始 rc 写进文案，本脚本一律退 20：
#      透传会把 docker 的 1/2/7/8/9/125 读成"用法/上游/补丁"这些保留语义。
#      码位 20 已登记进交接约束第 4 条滚动表（任务 9 的映射表须按 21 起算套件子门）。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

PATCH_FILE="$SCRIPT_DIR/mirror-patch-v1.patch"
STAGE_DIR="${SM75_CN_STAGE_DIR:-${TMPDIR:-/tmp}/sm75-gpustack-cn}"
REFRESH=1
DO_BUILD=0

usage() {
  awk 'NR>1{ if (/^#/) { sub(/^# ?/,""); print; next } exit }' "$0"
}

# 前置守卫类失败一律走用法码 2；补丁类红（8）、上游类红（9）、构建失败（20）在各门里
# 显式退出，不靠这里的默认值，避免"哪道门红的"被默认码抹平。
fail() { printf '%s\n' "$@" >&2; exit 2; }

# realpath 与 awk 在参数解析阶段就要用（--patch-file 的绝对化、usage 的输出），
# 缺件时命令替换得到空串、空串在下游正是"无问题"值，故哨兵排在 while 之前；
# 其余外部命令的哨兵见下方"前置守卫"清单。
for early_tool in realpath awk; do
  command -v "$early_tool" >/dev/null 2>&1 || \
    fail "错误：缺少必要工具「$early_tool」，拒绝执行补丁门。"
done

while [ $# -gt 0 ]; do
  case "$1" in
    --patch-file)
      [ $# -ge 2 ] || { echo "错误：--patch-file 缺少参数值。" >&2; usage >&2; exit 2; }
      # 一律先解析成绝对形（与暂存目录同法）：门 2/3 用 patch -d "$STAGE_ABS" 跑，
      # 而 -i 按调用者 cwd 解析相对路径——同一个"找不到补丁"会在两道门上被读成
      # "补丁内容与上游不匹配"（假归因），路径在此统一到绝对形即可消除。
      patch_abs=$(realpath -m -- "$2") || \
        { echo "错误：--patch-file 路径解析失败：$2" >&2; usage >&2; exit 2; }
      [ -n "$patch_abs" ] || \
        { echo "错误：--patch-file 路径解析得到空串：$2" >&2; usage >&2; exit 2; }
      PATCH_FILE=$patch_abs; shift 2 ;;
    --stage-dir)
      [ $# -ge 2 ] || { echo "错误：--stage-dir 缺少参数值。" >&2; usage >&2; exit 2; }
      STAGE_DIR=$2; shift 2 ;;
    --no-refresh) REFRESH=0; shift ;;
    --build) DO_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未识别的参数「$1」。" >&2; usage >&2; exit 2 ;;
  esac
done

# grep -c 无匹配时打印 0 且 rc=1（正常），rc>=2 才是真错误；错误与"零匹配"必须分开处理，
# 否则工具报错会被读成"特征缺失"，把环境问题误判成补丁问题。空串即"取不到"，调用方判死。
feat_count() {
  local pattern=$1 file=$2 out rc
  out=$(grep -c -F -- "$pattern" "$file"); rc=$?
  if [ "$rc" -ge 2 ]; then printf ''; return 1; fi
  case "$out" in ''|*[!0-9]*) printf ''; return 1 ;; esac
  printf '%s' "$out"; return 0
}

# ---------- 前置守卫（全部排在自家写盘之前）----------
# 工具哨兵按本脚本实际用到的外部命令列全：缺件时命令替换得到空串，
# 而空串在多处正是"无问题"值，不先验可执行就会把环境问题读成通过。
for needed_tool in patch grep sed tr tail date cp mkdir cmp wc realpath bash awk; do
  command -v "$needed_tool" >/dev/null 2>&1 || \
    fail "错误：缺少必要工具「$needed_tool」，拒绝执行补丁门。"
done

UPSTREAM_DOCKERFILE="$REPO_ROOT/docker/Dockerfile"
GATE_SCRIPT="$SCRIPT_DIR/gate-tree-unchanged-v1.sh"

[ -f "$UPSTREAM_DOCKERFILE" ] && [ -r "$UPSTREAM_DOCKERFILE" ] || \
  fail "错误：仓内原件不可读：docker/Dockerfile（相对仓库根 $REPO_ROOT）。"
[ -f "$GATE_SCRIPT" ] && [ -r "$GATE_SCRIPT" ] || \
  fail "错误：上游护栏脚本不可读：gpustack/gate-tree-unchanged-v1.sh。"
[ -f "$PATCH_FILE" ] && [ -r "$PATCH_FILE" ] || \
  fail "错误：补丁文件不存在或不可读：$PATCH_FILE。"
[ -s "$PATCH_FILE" ] || \
  fail "错误：补丁文件为空（0 字节），无法据此打补丁：$PATCH_FILE。"

# 行尾守卫：Windows 上产出的补丁若混入 CR，Linux 的 patch 会因上下文行尾不符而错位失败，
# 或更糟——静默匹配到错误位置。这里先判死，别把问题留给构建期。
cr_bytes=$(tr -dc '\r' < "$PATCH_FILE" | wc -c)
case "$cr_bytes" in
  ''|*[!0-9]*) fail "错误：无法统计补丁文件的 CR 字节数（wc 异常）。" ;;
esac
[ "$cr_bytes" -eq 0 ] || \
  fail "错误：补丁文件含 $cr_bytes 个 CR 字节，须转为 LF 后重试：$PATCH_FILE。"

first_line=$(sed -n '1p' "$PATCH_FILE")
[ "$first_line" = '--- a/docker/Dockerfile' ] || \
  fail "错误：补丁表头异常，期望「--- a/docker/Dockerfile」，实得「$first_line」。"

# 暂存目录不得落在仓库内：落仓内会让上游护栏把副本当新增件报漂移，也会污染工作树。
# 判定排在 mkdir 之前（realpath -m 允许路径不存在），否则被拒时已在仓库里建出目录。
STAGE_ABS=$(realpath -m -- "$STAGE_DIR") || fail "错误：无法解析暂存目录：$STAGE_DIR。"
case "$STAGE_ABS/" in
  "$REPO_ROOT/"*) fail "错误：暂存目录不能位于仓库内（会污染工作树并触发上游护栏新增红）：$STAGE_ABS。" ;;
esac
mkdir -p "$STAGE_ABS/docker" || fail "错误：无法创建暂存目录：$STAGE_ABS/docker。"
STAGED="$STAGE_ABS/docker/Dockerfile"

echo "== 国内源补丁门 =="
echo "仓库根：$REPO_ROOT"
echo "补丁：  $PATCH_FILE"
echo "暂存：  $STAGED"

# ---------- 门 1：复制原件到构建目录 ----------
if [ "$REFRESH" -eq 1 ]; then
  cp -f -- "$UPSTREAM_DOCKERFILE" "$STAGED" || fail "错误：复制 docker/Dockerfile 到暂存目录失败。"
  cmp -s -- "$UPSTREAM_DOCKERFILE" "$STAGED" || \
    fail "错误：暂存副本与仓内原件字节不一致，复制异常，中止。"
  echo "门 1/5 复制原件：绿（副本已重置为仓内 docker/Dockerfile 逐字节副本）"
else
  [ -f "$STAGED" ] || \
    fail "错误：--no-refresh 要求暂存副本已存在，但 $STAGED 不存在。"
  echo "门 1/5 复制原件：跳过（--no-refresh，直接在已有暂存副本上跑门）"
fi

# ---------- 门 2：patch --dry-run 预检 ----------
# -f 必带：不带时 GNU patch 遇到"已应用/反向"会在终端提问并挂住，非交互环境则静默跳过
# 仍退 1。带 -f 后行为确定：一律按正向尝试，不匹配就红。
dry_out=$(patch -p1 -f --dry-run -i "$PATCH_FILE" -d "$STAGE_ABS" 2>&1)
dry_rc=$?
if [ "$dry_rc" -ne 0 ]; then
  printf '%s\n' "$dry_out" | sed 's|^|    |' >&2
  # 按 patch 的返回码分流，不合并：GNU patch 用 1 表示"hunk 不匹配"（补丁/副本内容问题），
  # 用 >=2 表示工具级错误（补丁或目标打不开、权限、补丁格式坏）。合并成 8 会把环境问题
  # 谎报成"补丁与上游不匹配"——本脚本对 grep 的 rc 已是同一原则（见 feat_count），门 2 不能例外。
  if [ "$dry_rc" -ge 2 ]; then
    printf '红：patch 工具级错误（rc=%s，>=2），属前置输入问题，按用法码 2 退出；上方为 patch 原文，未做补丁内容判定。\n' "$dry_rc" >&2
    exit 2
  fi
  # rc==1：内容不匹配。归因再分两路：先看副本里是否已有补丁特征，有＝重复应用；没有＝上下文与副本不匹配。
  # 不分支就断言"已打过补丁"会把补丁/上游版本漂移误报成用户操作问题。
  already=$(feat_count 'PYTORCH_MIRROR' "$STAGED")
  echo "诊断：" >&2
  if [ -n "$already" ] && [ "$already" -ge 1 ]; then
    echo "    暂存副本已含 PYTORCH_MIRROR（$already 行）＝该副本已打过补丁，补丁不可重复应用；" >&2
    echo "    去掉 --no-refresh 让门 1 重置副本即可。" >&2
  else
    echo "    暂存副本不含补丁特征，说明是上下文不匹配：仓内 docker/Dockerfile 与本补丁" >&2
    echo "    制作时的上游版本不一致（补丁基于 v0.1.7-beta2）。" >&2
  fi
  printf '红：patch --dry-run 预检失败（rc=%s＝hunk 不匹配），退出码 8。\n' "$dry_rc" >&2
  exit 8
fi
echo "门 2/5 dry-run 预检：绿"

# ---------- 门 3：实应用 ----------
apply_out=$(patch -p1 -f -i "$PATCH_FILE" -d "$STAGE_ABS" 2>&1)
apply_rc=$?
if [ "$apply_rc" -ne 0 ]; then
  printf '%s\n' "$apply_out" | sed 's|^|    |' >&2
  # 门 2 已按 rc 分流并通过，这里再出现 >=2 说明是应用阶段新发生的环境问题
  # （副本被别的进程改动/移走、权限），与"补丁内容不匹配"不同类，按同一原则分道。
  if [ "$apply_rc" -ge 2 ]; then
    printf '红：patch 实应用出现工具级错误（rc=%s，>=2），属前置输入问题，按用法码 2 退出；上方为 patch 原文。\n' "$apply_rc" >&2
    exit 2
  fi
  printf '红：dry-run 通过但实应用失败（rc=%s＝hunk 不匹配），补丁或副本状态可疑，退出码 8。\n' "$apply_rc" >&2
  exit 8
fi
[ -f "$STAGED" ] || {
  printf '红：实应用后暂存副本消失：%s，退出码 8。\n' "$STAGED" >&2
  exit 8
}
echo "门 3/5 实应用：绿"

# ---------- 门 4：特征计数门 ----------
# feat_count 在文件头部定义（门 2 的诊断也要用）。
declare -a CHECK_PATTERNS=(
  'PYTORCH_MIRROR'
  'PLACEHOLDER://release/flashinfer_cubin'
  'PLACEHOLDER://release/nvidia-pstated'
  'UV_DEFAULT_INDEX'
  'https://download.pytorch.org/whl/cu129'
  'https://flashinfer.ai/whl/'
  'https://github.com/sasha0552/nvidia-pstated'
)
# 期望：前四项为补丁应留下的特征（>=2 / ==1 / ==1 / >=1，逐字取自简报），
# 后三项是被替换掉的原始锈点，替换后必须归零（防"叠加而非替换"造成的多）。
declare -a CHECK_OP=( '>=' '==' '==' '>=' '==' '==' '==' )
declare -a CHECK_WANT=( 2 1 1 1 0 0 0 )
declare -a CHECK_LABEL=( '特征 PyTorch 索引参数' '特征 cubin 托管直链' '特征 pstated 托管直链' '特征 uv 默认索引' '残留 pytorch 官方索引' '残留 flashinfer 官方索引' '残留 pstated GitHub 直链' )

count_bad=0
echo "门 4/5 特征计数："
printf '%-26s %-6s %-6s %-6s %s\n' "项" "期望" "实测" "判定" "串"
for i in 0 1 2 3 4 5 6; do
  pat=${CHECK_PATTERNS[$i]}
  got=$(feat_count "$pat" "$STAGED")
  if [ -z "$got" ]; then
    printf '%-26s %-6s %-6s %-6s %s\n' "${CHECK_LABEL[$i]}" "${CHECK_OP[$i]}${CHECK_WANT[$i]}" "取不到" "红" "$pat"
    count_bad=$((count_bad + 1))
    continue
  fi
  ok=0
  case "${CHECK_OP[$i]}" in
    '==') [ "$got" -eq "${CHECK_WANT[$i]}" ] && ok=1 ;;
    '>=') [ "$got" -ge "${CHECK_WANT[$i]}" ] && ok=1 ;;
  esac
  if [ "$ok" -eq 1 ]; then
    printf '%-26s %-6s %-6s %-6s %s\n' "${CHECK_LABEL[$i]}" "${CHECK_OP[$i]}${CHECK_WANT[$i]}" "$got" "绿" "$pat"
  else
    printf '%-26s %-6s %-6s %-6s %s\n' "${CHECK_LABEL[$i]}" "${CHECK_OP[$i]}${CHECK_WANT[$i]}" "$got" "红" "$pat"
    count_bad=$((count_bad + 1))
  fi
done
if [ "$count_bad" -ne 0 ]; then
  echo "红：特征计数门有 $count_bad 项不符（少或多都算），退出码 8。" >&2
  echo "    暂存副本：$STAGED（可自行 grep 复核）" >&2
  exit 8
fi
echo "门 4/5 特征计数：绿（4 项特征达标，3 项原锈点归零）"

# ---------- 门 5：仓内原件上游门 ----------
gate_out=$(bash "$GATE_SCRIPT" 2>&1)
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
  printf '%s\n' "$gate_out" | sed 's|^|    |' >&2
  if [ "$gate_rc" -eq 7 ]; then
    echo "红：上游护栏报漂移（其自身码 7），本脚本按约定映射为 9＝上游文件被改。" >&2
    exit 9
  fi
  echo "红：上游护栏非漂移红（码 $gate_rc），属前置输入问题，按用法码 2 退出。" >&2
  exit 2
fi
echo "门 5/5 上游护栏：绿（$(printf '%s' "$gate_out" | tail -n 1)）"

echo "== 补丁门全绿，产物：$STAGED =="

# ---------- 构建调用段（本任务不执行，--build 才执行）----------
# 事实登记：docker/build.sh 无 Dockerfile 路径口子（docker/build.sh:44 硬编码 file=Dockerfile），
# 直调它只会构建仓内未打补丁的原件（静默失败）。所以本段用 docker build --file 消费门 3 的暂存副本，
# 其余逐项对齐 build.sh 的 standard:full 组装——即 EDITION=standard 且 BUILD_MODE=full 时
# build.sh 实际走的那条路径（:44-58）；:7-10 的 unraid 阻断门与 :23-42 的 ultra 分支本段不适用。
# 逐行对应关系（行号指仓内上游件 docker/build.sh，HEAD c79894c 版本）：
#   :44  file=Dockerfile                       → --file 指向暂存副本（本段唯一实质差异，见上）
#   :45  default_image="vllm-sm75:v${VERSION}"  → target_tag 同一算法；:58 的 IMAGE 覆盖口一致
#   :50  --target final                        → 同
#   :21  BASE_IMAGE 默认值（tag＋@sha256 摘要双钉）→ BASE_IMAGE_DEFAULT 逐字同值，env BASE_IMAGE 可覆盖
#   :51  --build-arg BASE_IMAGE="$BASE_IMAGE"   → 恒传（改前只在 env 非空时传，靠 docker/Dockerfile:3 的 ARG 同值兜底）
#   :52  --build-arg MAX_JOBS="${MAX_JOBS:-1}"  → 同
#   :53  --build-arg INSTALL_LMCACHE="${…:-0}"  → 同
#   :54  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" → 同
#   :55  --build-arg SOURCE_REVISION            → 默认值按简报钉成 upstream-v0.1.7-beta2
#        （:22 的默认是 git rev-parse HEAD，会随提交漂移；本线要固定可追溯串，这是有意差异）
#   :56  --build-arg BASE_IMAGE_ID="$BASE_IMAGE" → 恒传，值同 :51
#   :57  --build-arg VLLM_SM75_RELEASE="vllm-sm75-v${VERSION}" → 同
#   :58  --tag "${IMAGE:-$default_image}" "$ROOT" → 同，构建上下文＝$REPO_ROOT
# 本段额外项（补丁新引入的两个 ARG，build.sh 无对应行）：PIP_MIRROR／PYTORCH_MIRROR——
#   env 给出才传 --build-arg，不给就沿用 Dockerfile 内 ARG 的国内默认值，换源地址不必重打补丁。
# 提醒：同一 digest 在 docker/build.sh:21 与 docker/Dockerfile:3 各写死一份，本段以 :21 为准恒传并
#   覆盖 Dockerfile 的 ARG；两处若不一致，生效的是本段传的值（任务 14 的 G-3 一并核 digest）。
# 简报要求的 EDITION=standard BUILD_MODE=full 作为前置 env 一并给出：它们是 build.sh 的入口开关
# （:16-19 校验、:44-48 据此选 file），docker 本身不消费；保留只为"同一份 env 可直接喂 build.sh"
# 的场合复用，本段用 --file 指向打过补丁的暂存副本已达到同一效果。
VERSION_FILE="$REPO_ROOT/docker/VERSION"
BASE_IMAGE_DEFAULT='vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688'
if [ "$DO_BUILD" -eq 1 ]; then
  command -v docker >/dev/null 2>&1 || fail "错误：--build 需要 docker，但当前 PATH 上没有。"
  [ -f "$VERSION_FILE" ] || fail "错误：docker/VERSION 不存在，无法生成 tag。"
  VERSION=$(tr -d ' \r\n' < "$VERSION_FILE")
  [ -n "$VERSION" ] || fail "错误：docker/VERSION 为空，无法生成 tag。"

  SRC_REV="${SOURCE_REVISION:-upstream-v0.1.7-beta2}"
  base_image="${BASE_IMAGE:-$BASE_IMAGE_DEFAULT}"
  build_args=( --file "$STAGED" --target final
               --build-arg BASE_IMAGE="$base_image"
               --build-arg BASE_IMAGE_ID="$base_image"
               --build-arg MAX_JOBS="${MAX_JOBS:-1}"
               --build-arg INSTALL_LMCACHE="${INSTALL_LMCACHE:-0}"
               --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
               --build-arg SOURCE_REVISION="$SRC_REV"
               --build-arg VLLM_SM75_RELEASE="vllm-sm75-v${VERSION}" )
  if [ -n "${PIP_MIRROR:-}" ]; then
    build_args+=( --build-arg PIP_MIRROR="$PIP_MIRROR" )
  fi
  if [ -n "${PYTORCH_MIRROR:-}" ]; then
    build_args+=( --build-arg PYTORCH_MIRROR="$PYTORCH_MIRROR" )
  fi
  target_tag="${IMAGE:-vllm-sm75:v${VERSION}}"
  build_cmd=( docker build "${build_args[@]}" --tag "$target_tag" "$REPO_ROOT" )
  echo "== 构建调用段（EDITION=standard BUILD_MODE=full，逐项对齐 docker/build.sh:44-58）=="
  printf '将执行（前置 env：EDITION=standard BUILD_MODE=full，docker 不消费，仅形态登记）：'
  printf ' %q' "${build_cmd[@]}"
  printf '\n'
  EDITION=standard BUILD_MODE=full "${build_cmd[@]}"
  build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    # 不透传 docker 的码：docker 失败最常见是 1，也可能是 2/7/8/9/125，而 2/7/8/9 在本线各有
    # 保留语义（用法／上游树漂移／补丁不匹配／上游被改），透传会把构建失败读成别的原因。
    # 原始 rc 留在文案里供定位；码位 20＝构建失败，已登记进交接约束第 4 条滚动表。
    echo "红：docker build 失败（原始 rc=$build_rc），本脚本按约定退 20＝构建失败，不报构建完成。" >&2
    exit 20
  fi
  echo "构建完成：$target_tag"
else
  echo "构建调用段：未执行（本任务红线）。加 --build 才会组装并执行 docker build。"
  echo "等价形态：EDITION=standard BUILD_MODE=full SOURCE_REVISION=upstream-v0.1.7-beta2 "
  echo "          [BASE_IMAGE=<国内代理地址>] docker build --file <上面产物> --target final ..."
fi
exit 0
