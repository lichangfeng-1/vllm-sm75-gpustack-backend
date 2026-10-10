#!/usr/bin/env bash
# lint-count-v1.sh —— pcieipc 层（镜像层 1）件数门，红码 17
#
# 做什么：核对层 1 物料在 ctx 目录里的“件数形状”——newfiles 恰好 6 件、append 恰好 3 件、
#         manifest.txt 非空且逐行与盘上一一对应（路径、类型、sha256 三样都对得上）。
# 为什么：装机脚本按 manifest 落盘并断言 6 件；物料少一件、多一件或清单被截断，都必须在
#         构建最前期红掉，否则装到一半留下半成品镜像，事后极难归因。
# 预期输出：末行 LINT_COUNT_OK，退出码 0。
# 失败怎么办：退 17 时按点名处置——缺件就重拷物料（源＝旧部署包 docker/pcieipc/），
#         清单为空就回填 manifest.txt；不得改判据凑绿，也不得删清单来绕过本门。
#
# 用法：
#   bash lint-count-v1.sh                      # ctx＝本脚本所在目录
#   bash lint-count-v1.sh --ctx DIR            # 负控夹具目录
#
# 退出码：
#   0   通过
#   2   用法错误 / 前置输入不可用（ctx 不存在或不可进入、必要工具缺失、清单不可读）
#   17  件数门红（存在性断言红、件数不符、清单条目与盘上不符、行格式非法、sha 不符、重复键）
#       简报只给了 17 一个红码，故本门所有“内容红”共用 17，靠点名文案区分命中的是哪个断言。
#
# 断言顺序（存在性断言先行，空判据不许裸跑）：
#   A1 newfiles/ 目录存在        A2 append/ 目录存在
#   A3 manifest.txt 存在且非空   A4 obligations.txt 存在且非空（gate-check 的义务表判据源）
#   只有 A1-A4 全绿，才允许跑“每行对应一件”“实到件数”这类计数判据——
#   否则“0 行 = 0 个问题”会把空清单读成通过。

set -uo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
CTX=$SCRIPT_DIR

usage() {
  echo "用法：bash lint-count-v1.sh [--ctx DIR]"
  echo "说明：核对 pcieipc 层物料的件数与清单一致性；ctx 默认为本脚本所在目录。"
}

red() {
  echo "红：件数门（本护栏退出码 17）—— $*" >&2
  exit 17
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
      [ -n "$2" ] || bad_usage "--ctx 的参数值是空串。"
      CTX=$(cd -- "$2" 2>/dev/null && pwd) || bad_usage "ctx 目录不存在或不可进入：$2"
      [ -n "$CTX" ] || bad_usage "ctx 目录绝对化得到空串：$2"
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) bad_usage "未识别的参数「$1」。" ;;
  esac
done

[ -d "$CTX" ] || bad_usage "ctx 目录不存在：$CTX"

# 工具哨兵：本门所有判据都来自这些工具的 stdout，而空串正是各道的“无问题”值。
for needed_tool in find sed sort awk grep cut uniq comm sha256sum mktemp rm; do
  command -v "$needed_tool" >/dev/null 2>&1 || \
    bad_usage "缺少必要工具「$needed_tool」，拒绝做件数判定。"
done

NEWFILES_WANT=6
APPEND_WANT=3
MANIFEST="$CTX/manifest.txt"
OBLIGATIONS="$CTX/obligations.txt"

echo "== 件数门（lint-count-v1）ctx=$CTX =="

# ---------- A1-A4 存在性断言 ----------
[ -d "$CTX/newfiles" ] || red "A1 存在性断言：$CTX/newfiles 目录不存在（六件逐字节物料的落点）。"
[ -d "$CTX/append" ]   || red "A2 存在性断言：$CTX/append 目录不存在（三处追加块的落点）。"
[ -f "$MANIFEST" ]     || red "A3 存在性断言：清单文件不存在：$CTX/manifest.txt。"
[ -r "$MANIFEST" ]     || bad_usage "清单文件不可读：$MANIFEST"
[ -s "$MANIFEST" ]     || red "A3 存在性断言：manifest.txt 是 0 字节空文件——空清单上「每行对应一件」恒真，不予通过。"
[ -f "$OBLIGATIONS" ]  || red "A4 存在性断言：$CTX/obligations.txt 不存在（gate-check 的义务表判据源）。"
[ -r "$OBLIGATIONS" ]  || bad_usage "obligations.txt 不可读：$OBLIGATIONS"

# 剥掉注释与空行后的义务条目数；取不到数字即判死，不用空串当“0 条也通过”。
oblig_rows=$(awk 'NF>0 && $0 !~ /^#/ { n++ } END { print n + 0 }' "$OBLIGATIONS")
case "$oblig_rows" in
  ''|*[!0-9]*) red "A4 存在性断言：无法统计 obligations.txt 的有效条目数（awk 给出「$oblig_rows」）。" ;;
esac
[ "$oblig_rows" -gt 0 ] || red "A4 存在性断言：obligations.txt 有效条目为 0（只剩注释/空行），空义务表＝空门禁。"
echo "A1-A4 存在性断言：绿（newfiles/、append/、manifest.txt、obligations.txt 均在位，义务表 $oblig_rows 条）"

# ---------- 实到件数（find 在 ctx 内跑，路径天然相对，免 sed 转义） ----------
disk_newfiles=$(cd "$CTX" && find newfiles -type f -print | sort)
disk_append=$(cd "$CTX" && find append -type f -print | sort)
n_newfiles=$(printf '%s\n' "$disk_newfiles" | grep -c . || true)
n_append=$(printf '%s\n' "$disk_append" | grep -c . || true)
for cnt in "$n_newfiles" "$n_append"; do
  case "$cnt" in
    ''|*[!0-9]*) red "件数统计异常：find 管道给出的计数不是数字（「$cnt」），拒绝判定。" ;;
  esac
done
if [ "$n_newfiles" -ne "$NEWFILES_WANT" ]; then
  echo "  实到 newfiles 件数=$n_newfiles（期望 $NEWFILES_WANT），盘上清单："
  printf '%s\n' "$disk_newfiles" | sed '/^$/d;s|^|    |'
  red "newfiles 实到件数 $n_newfiles != $NEWFILES_WANT（缺件或多件都走到这里）。"
fi
if [ "$n_append" -ne "$APPEND_WANT" ]; then
  echo "  实到 append 件数=$n_append（期望 $APPEND_WANT），盘上清单："
  printf '%s\n' "$disk_append" | sed '/^$/d;s|^|    |'
  red "append 实到件数 $n_append != $APPEND_WANT。"
fi
echo "  实到件数：newfiles=$n_newfiles/$NEWFILES_WANT，append=$n_append/$APPEND_WANT"

# ---------- 清单行格式（5 列制表符分隔，类型与路径前缀互钉） ----------
# 不用 {64} 量词：镜像里的 mawk 不支持重复区间，量词失配会让整表被误判非法（同任务 1 的处置）。
# awk 报错与“无非法行”都表现为空 stdout，所以这里必须收 rc：工具级失败一律走 2，不当通过。
badrows=$(awk -F'\t' '
  /^[[:space:]]*$/     { next }
  /^#/                 { next }
  {
    if (NF != 5)       { print NR" 列数="NF": "$0; next }
    for (i = 1; i <= 5; i++) {
      if ($i == "")  { print NR" 第"i"列为空: "$0; next }
    }
    want_kind = ""
    if ($1 ~ /^newfiles\//)    { want_kind = "verbatim" }
    else if ($1 ~ /^append\//) { want_kind = "append" }
    else { print NR" 路径前缀异常: "$1; next }
    if ($5 != want_kind) { print NR" 类型="$5" 与前缀不符（应为 "want_kind"）: "$1; next }
    if (length($4) != 64 || $4 !~ /^[0-9a-fA-F]+$/) { print NR" sha256 形态非法: "$4 }
  }' "$MANIFEST")
badrows_rc=$?
if [ "$badrows_rc" -ne 0 ]; then
  bad_usage "清单行格式检查的 awk 非零退出（rc=$badrows_rc）；工具级失败不读作“无非法行”。"
fi
if [ -n "$badrows" ]; then
  printf '%s\n' "$badrows" | sed 's|^|    |'
  red "manifest.txt 存在格式非法行（列数、空列、路径前缀、类型互钉、sha 形态五检之一不过）。"
fi

# ---------- 清单有效条目（存在性断言，先于任何“逐行对应”计数） ----------
rows=$(awk -F'\t' '!/^[[:space:]]*$/ && !/^#/ { print }' "$MANIFEST")
if [ -z "$rows" ]; then
  red "A3 存在性断言：manifest.txt 非 0 字节但剥掉注释与空行后有效条目为 0，判据本身不可空跑。"
fi
n_rows=$(printf '%s\n' "$rows" | grep -c .)
n_verbatim=$(printf '%s\n' "$rows" | awk -F'\t' '$5=="verbatim"' | grep -c . || true)
n_append_rows=$(printf '%s\n' "$rows" | awk -F'\t' '$5=="append"' | grep -c . || true)
if [ "$n_verbatim" -ne "$NEWFILES_WANT" ]; then
  red "manifest 里 verbatim 行 $n_verbatim 条 != $NEWFILES_WANT。"
fi
if [ "$n_append_rows" -ne "$APPEND_WANT" ]; then
  red "manifest 里 append 行 $n_append_rows 条 != $APPEND_WANT。"
fi
dup_paths=$(printf '%s\n' "$rows" | awk -F'\t' '{print $1}' | sort | uniq -d)
if [ -n "$dup_paths" ]; then
  printf '%s\n' "$dup_paths" | sed 's|^|    |'
  red "manifest.txt 存在重复路径（一件只能有一条清单记录）。"
fi
echo "  清单有效条目=$n_rows（verbatim $n_verbatim＋append $n_append_rows），无重复键"

# ---------- 每行对应一件：盘上存在＋sha256 相符 ----------
missing=0
shabad=0
while IFS= read -r row; do
  [ -n "$row" ] || continue
  rel=$(printf '%s' "$row" | cut -f1)
  want=$(printf '%s' "$row" | cut -f4)
  if [ ! -f "$CTX/$rel" ]; then
    echo "    清单有、盘上无：$rel" >&2
    missing=$((missing + 1))
    continue
  fi
  got=$(sha256sum -- "$CTX/$rel" | cut -d' ' -f1)
  case "$got" in
    ''|*[!0-9a-f]*) echo "    sha 统计异常：$rel（「$got」）" >&2; missing=$((missing + 1)); continue ;;
  esac
  if [ "$got" != "$want" ]; then
    echo "    sha 不符：$rel 期望 $want 实为 $got" >&2
    shabad=$((shabad + 1))
  fi
done <<EOF
$rows
EOF
if [ "$missing" -ne 0 ] || [ "$shabad" -ne 0 ]; then
  red "清单与盘上一一对应不成立（清单有盘上无或统计异常 $missing 件、sha 不符 $shabad 件）。"
fi

# ---------- 反向核对：盘上有、清单无（多件与杂件） ----------
tmpdir=$(mktemp -d 2>/dev/null)
[ -n "$tmpdir" ] && [ -d "$tmpdir" ] || bad_usage "无法创建临时目录做双向比对。"
trap 'rm -rf "$tmpdir"' EXIT
{ printf '%s\n' "$disk_newfiles"; printf '%s\n' "$disk_append"; } | sed '/^$/d' | sort > "$tmpdir/disk"
printf '%s\n' "$rows" | awk -F'\t' '{print $1}' | sort > "$tmpdir/rows"
extra=$(comm -23 "$tmpdir/disk" "$tmpdir/rows")
if [ -n "$extra" ]; then
  printf '%s\n' "$extra" | sed 's|^|    |'
  red "盘上有、清单无的多余文件（件数门按“恰好 6＋3”判定，多余件同样红）："
fi

echo "  清单↔盘 双向一致：$n_rows 件逐件在位、sha256 逐件相符、无多余件"
echo "LINT_COUNT_OK newfiles=$n_newfiles append=$n_append manifest_rows=$n_rows obligations=$oblig_rows"
exit 0
