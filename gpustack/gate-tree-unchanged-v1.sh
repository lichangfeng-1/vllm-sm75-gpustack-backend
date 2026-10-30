#!/usr/bin/env bash
# gate-tree-unchanged-v1.sh —— 上游树（VLLM-SM75 v0.1.7-beta2）字节零改动护栏
#
# 用法：
#   bash gpustack/gate-tree-unchanged-v1.sh                  # 校验：与基线表逐件比对
#   bash gpustack/gate-tree-unchanged-v1.sh --record         # 生成基线表（任务 1 首建用）
#   bash gpustack/gate-tree-unchanged-v1.sh --record --force # 显式覆盖已有基线表（正常流程禁止）
#
# 退出码：
#   0  通过
#   7  检测到漂移 —— 本护栏唯一的“红”码（内容篡改 / 文件缺失 / 上游作用域内新增，含行尾变化）
#   2  用法错误
#   4  基线表格式不合法（非法行 / 重复路径 / 末行缺换行导致行数与条目不符）
#   5  基线表已存在且未加 --force（防止静默重定基线）
#   6  作用域为空 / 文件不可读 / 必要工具缺失
#   10 基线表不存在
#   19 上游作用域内存在编译产物（__pycache__ / *.pyc / *.pyo），须先清理
#
# 码位说明：除 7 以外，本脚本只用 2/4/5/6/10/19。其余码位的归属一律以
# gpustack/exit-code-map-v1.md（任务 9）为权威，本注释不复制码表快照，
# 因为滚动码表每加一个任务就会膨胀，枚举在这里必然腐化。
#
# 作用域定义：上游冻结面 = 全仓库减去 owned 清单（gate-scope-v1.json）与 .git。
# owned 内的路径可自由改动；未列入的一律逐字节比对基线表，任何增删改（含行尾）都红。
# 2026-10-30 v5 起作用域改为显式清单，不再靠“基线表顶层条目”推断——
# 后者会把自研目录也冻进去，改一次自研件就得重录一次基线。
# 清单缺失或格式非法时本脚本拒绝运行（退 6/4），避免静默退化成“只排除 .git”。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
# 2026-10-19：DSH 0.2 与面板修复后重录的基线（v2）；旧 SHA256SUMS-upstream.txt 保留为历史证据。
# 2026-10-23：v3 纳入 10 个自有提交的 8 改+2 新增（C3/速度缓存/external/性能档位/docs），v2 保留历史。
# 2026-10-30：v4 把 ultra/ 与三个缓存目录移出冻结面，并纳入 10-27 之后的上游作用域改动（295 件）。
# 2026-10-30：v5 把冻结面改为 gate-scope-v1.json 显式声明，tests/ tools/ vllm_sm75/ 一并移出（204 件）。
#             旧表全部保留为历史证据，复核旧基线用：
#             SM75_GATE_TABLE=gpustack/SHA256SUMS-upstream-20261023-v3.txt bash gpustack/gate-tree-unchanged-v1.sh
TABLE="${SM75_GATE_TABLE:-$SCRIPT_DIR/SHA256SUMS-upstream-20261030-v5.txt}"
SCOPE="${SM75_GATE_SCOPE:-$SCRIPT_DIR/gate-scope-v1.json}"
TABLE_REL=${TABLE#"$REPO_ROOT"/}
SELF=${0##*/}
cd "$REPO_ROOT" || exit 6

usage() {
  echo "用法：bash gpustack/$SELF [--record [--force]]"
  echo "说明：无参数＝校验上游树与基线表是否逐字节一致；--record＝生成基线表。"
}

MODE=check
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --record) MODE=record ;;
    --force)  FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未识别的参数「$arg」"; usage; exit 2 ;;
  esac
done

# 用法门：--force 单独使用会被静默降级成校验，使用者会以为基线表已被覆盖。
if [ "$FORCE" -ne 0 ] && [ "$MODE" != record ]; then
  echo "错误：--force 只在 --record 模式下有意义，请写成 --record --force。"
  usage
  exit 2
fi

# 工具哨兵：本脚本所有判据都来自这些工具的 stdout，而空串正是各道的“无问题”值。
# 缺件时命令替换得到空串，会把“工具不可用”误报成“校验通过”，故一律先验可执行。
for needed_tool in find awk sed sort uniq grep sha256sum wc; do
  if ! command -v "$needed_tool" >/dev/null 2>&1; then
    echo "错误：缺少必要工具「$needed_tool」，拒绝校验。"
    exit 6
  fi
done

# 上游作用域内所有普通文件（相对路径，按字节序排序）。
#
# 冻结面 = 全仓库 − .git − owned 清单（gate-scope-v1.json）。
# owned 清单是权威：改自研件、加自研目录都不该红，只有上游基座的增删改才红。
# 清单缺失或格式非法时一律拒绝运行，绝不退化成“只排除 .git + gpustack”的旧行为。
if [ ! -f "$SCOPE" ]; then
  echo "错误：作用域清单不存在：${SCOPE#"$REPO_ROOT"/}"
  echo "本脚本拒绝在无显式 owned 清单时运行——否则自研路径会被误钉进上游冻结面。"
  exit 6
fi
scope_owned=$(awk '
  /"owned"[[:space:]]*:/ { inarr=1; next }
  inarr && /^[[:space:]]*\]/ { inarr=0; next }
  inarr {
    if ($0 !~ /^[[:space:]]*"[^"]+"[[:space:]]*,?[[:space:]]*$/) { print "__BADLINE__"; next }
    line=$0
    sub(/^[[:space:]]*"/, "", line)
    sub(/"[[:space:]]*,?[[:space:]]*$/, "", line)
    print line
  }
' "$SCOPE")
if [ -z "$scope_owned" ]; then
  echo "错误：作用域清单里没有解析到任何 owned 条目：${SCOPE#"$REPO_ROOT"/}"
  exit 4
fi
if printf '%s\n' "$scope_owned" | grep -qx '__BADLINE__'; then
  echo "错误：作用域清单格式非法（owned 元素须每行一个、双引号包裹）：${SCOPE#"$REPO_ROOT"/}"
  exit 4
fi
scope_dups=$(printf '%s\n' "$scope_owned" | sort | uniq -d)
if [ -n "$scope_dups" ]; then
  echo "错误：作用域清单存在重复 owned 条目："
  printf '%s\n' "$scope_dups" | sed 's|^|    |'
  exit 4
fi
scope_bad=$(printf '%s\n' "$scope_owned" | awk '/\// || /^\.$/ || /^\.\.$/ {print}')
if [ -n "$scope_bad" ]; then
  echo "错误：owned 条目只能是仓根顶层名，不得含斜杠或指向上级："
  printf '%s\n' "$scope_bad" | sed 's|^|    |'
  exit 4
fi
if ! printf '%s\n' "$scope_owned" | grep -qx 'gpustack'; then
  echo "错误：owned 清单必须包含 gpustack（本护栏自身所在目录），否则脚本会把自己钉进上游面。"
  exit 4
fi

GATE_PRUNE=( -path ./.git )
while IFS= read -r owned; do
  [ -n "$owned" ] || continue
  GATE_PRUNE+=( -o -path "./$owned" )
done <<< "$scope_owned"

list_scope_files() {
  find . \( "${GATE_PRUNE[@]}" \) -prune -o -type f -print |
    sed 's|^\./||' | sort
}

if [ "$MODE" = record ]; then
  # 拒覆盖门排在编译产物门之前：表已存在时无论树脏不脏都不该重录。
  # 反过来的话，脏树会先报 19 并把人引向“清理完就能重录”，而下一步实被 5 拒。
  if [ -f "$TABLE" ] && [ "$FORCE" -ne 1 ]; then
    echo "错误：基线表已存在，拒绝覆盖（重录会抹掉上游改动证据）：$TABLE_REL"
    echo "确实需要重录时加 --force，并在报告里写明理由。"
    exit 5
  fi
  junk=$(find . \( "${GATE_PRUNE[@]}" \) -prune -o \
        \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) -print | sed 's|^\./||')
  if [ -n "$junk" ]; then
    echo "错误：上游作用域内存在编译产物，先清理再生成基线表："
    printf '%s\n' "$junk" | sed 's|^|    |'
    exit 19
  fi
  tmp="$TABLE.tmp.$$"
  trap 'rm -f "$tmp"' EXIT
  count=0
  while IFS= read -r p; do
    out=$(sha256sum -- "$p") || { echo "错误：无法读取 $p"; exit 6; }
    h=${out%% *}
    case "$h" in
      *[!0-9a-f]*|'') echo "错误：$p 的摘要异常：$h"; exit 4 ;;
    esac
    printf '%s  %s\n' "$h" "$p" >> "$tmp"
    count=$((count + 1))
  done < <(list_scope_files)
  if [ "$count" -le 0 ]; then
    echo "错误：上游作用域内没有任何文件，空基线表不予生成。"
    exit 6
  fi
  mv -f "$tmp" "$TABLE"
  trap - EXIT
  echo "已生成基线表：$TABLE_REL（$count 件上游文件）"
  exit 0
fi

# ---------- 校验模式 ----------
if [ ! -f "$TABLE" ]; then
  echo "错误：基线表不存在：$TABLE_REL"
  echo "本仓首次建立基线时可用 --record 生成，之后再跑校验。"
  exit 10
fi

# 逐列切分而非 {64} 量词，避免 mawk 不支持 POSIX 量词时误判整表非法。
badlines=$(awk '{
  h=substr($0,1,64); sep=substr($0,65,2); rest=substr($0,67);
  if (length(h)==64 && h ~ /^[0-9a-f]+$/ && sep=="  " && rest!="" && substr(rest,1,1)!=" ") next;
  print NR": "$0
}' "$TABLE")
if [ -n "$badlines" ]; then
  echo "错误：基线表存在格式非法行，拒绝校验："
  printf '%s\n' "$badlines" | sed 's|^|    |'
  exit 4
fi

declare -A want
declare -A scope
total=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  h=${line%% *}
  p=${line#* }
  p=${p# }
  want["$p"]=$h
  scope["${p%%/*}"]=1
  total=$((total + 1))
done < "$TABLE"
if [ "$total" -le 0 ]; then
  echo "错误：基线表没有任何合法条目，无法校验。"
  exit 6
fi
# 三点互检：唯一键数 ↔ 装载条目数 ↔ 表逻辑行数。
# want 以路径为键，重复路径后写覆盖前写，唯一键数小于条目数即表内存在重复键。
if [ "${#want[@]}" -ne "$total" ]; then
  echo "错误：基线表存在重复路径，拒绝校验（一件上游文件只能有一条基线记录）。"
  exit 4
fi
# 末行缺换行时 read 会静默丢掉最后一条。awk 的 NR 数逻辑行、含未终止的末行，
# 与条目数失衡即可检出；wc -l 数的是换行符个数，与 read 同步少 1，等式恒成立。
tablelines=$(awk 'END{print NR}' "$TABLE")
case "$tablelines" in
  ''|*[!0-9]*)
    echo "错误：无法统计基线表行数（$TABLE_REL 不可读或 awk 异常）。"
    exit 6
    ;;
esac
if [ "$tablelines" -ne "$total" ]; then
  echo "错误：基线表行数 $tablelines 与已装载条目 $total 不符（疑末行缺换行符），拒绝校验。"
  exit 4
fi
echo "基线表条目 $total 件，顶层作用域 ${#scope[@]} 项。"

modified=()
missing=()
extra=()
seen=0
for p in "${!want[@]}"; do
  if [ ! -f "$p" ]; then
    missing+=("$p")
    continue
  fi
  out=$(sha256sum -- "$p") || { echo "错误：无法读取 $p"; exit 6; }
  h=${out%% *}
  case "$h" in
    *[!0-9a-f]*|'') echo "错误：$p 的摘要异常：$h"; exit 4 ;;
  esac
  seen=$((seen + 1))
  [ "${want[$p]}" = "$h" ] || modified+=("$p")
done

while IFS= read -r p; do
  [ -n "${scope[${p%%/*}]+x}" ] || continue
  [ -n "${want[$p]+x}" ] || extra+=("$p")
done < <(list_scope_files)

if [ "${#modified[@]}" -eq 0 ] && [ "${#missing[@]}" -eq 0 ] && [ "${#extra[@]}" -eq 0 ]; then
  echo "通过：上游树 $seen 件文件与基线表逐字节一致（作用域内无新增、无缺失）。"
  exit 0
fi

echo "红：上游树相对基线表出现漂移（本护栏退出码 7）"
if [ "${#modified[@]}" -gt 0 ]; then
  echo "内容被改动（含行尾转换）$(( ${#modified[@]} )) 件："
  printf '%s\n' "${modified[@]}" | sed 's|^|    |'
fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "基线中存在但工作树缺失 $(( ${#missing[@]} )) 件："
  printf '%s\n' "${missing[@]}" | sed 's|^|    |'
fi
if [ "${#extra[@]}" -gt 0 ]; then
  echo "上游作用域内新增 $(( ${#extra[@]} )) 件："
  printf '%s\n' "${extra[@]}" | sed 's|^|    |'
fi
echo "差集统计：篡改 ${#modified[@]}、缺失 ${#missing[@]}、新增 ${#extra[@]}（基线共 $total 件）"
exit 7
