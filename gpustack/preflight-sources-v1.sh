#!/usr/bin/env bash
# preflight-sources-v1.sh —— 构建期国内源探活（只读 HEAD 请求，不下载任何内容）
#
# 用法：
#   bash gpustack/preflight-sources-v1.sh                 按内置默认清单探活
#   bash gpustack/preflight-sources-v1.sh --list FILE     改用外部清单（自检与负控用）
#   清单格式：每行 源名|类别|URL|切换开关名|分组，# 开头为注释，空行忽略
#   类别：registry（镜像仓库代理）| index（pypi / wheels 索引）| placeholder（待任务 13 回填）
#   分组：互为替代的候选共用一个分组名（两条基础镜像代理同组）；没有候选的源自成一组，
#         组名直接复用源名。源名与分组名都不许为空，源名不许重复（重复会让下面的
#         "点名差额"定位失效，所以整表校验先判死）。
#
# 退出码（本脚本）：
#   0 每个分组组内至少一条真实源绿（PLACEHOLDER 项豁免，不计红）
#   2 用法错误 / 清单不可读或非法（字段数、空源名、空分组名、重复源名、类别非法、
#     占位前缀互锁、真实源非 http(s)）/ 清单不含真实源 / 绿红计数与真实源总数不符
#     （＝有源被静默跳过，逐条点名）/ 缺必要工具
#   3 至少一个分组"全组红"（逐条打印每条红源的 URL＋对应切换开关名＋所属分组）
#
# 判绿口径：
#   index       —— 与 curl -fsI 等价：请求成功且 HTTP 状态码 < 400
#   registry    —— 同上，另把 HTTP 401 单列为绿：Docker Registry v2 规范下未带 token 访问
#                  /v2/ 一律回 401 质询，token 由 docker CLI 换取，HEAD 探活代替不了它。
#   placeholder —— 打印"豁免（待任务 13 回填）"并跳过，不产生红。
#   分组取或    —— 同组候选互为替代：组内有一条绿即满足，全组红才判红退 3。
#                  单条自成一组时与简报的"任一红即退 3"逐字等价。
#   状态码取不到 —— curl 正常退出（rc=0）却拿不到 http_code＝响应形态异常，判红；
#                  不许把"取不到"当"没问题"。
# 局限：HEAD 探活只证明"地址可达、索引在线"，registry 行更只证代理在线（不证该镜像可拉，
#       digest 核验是任务 14 的 G-3），也不证明具体 wheel / 镜像 tag 在该镜像上存在。
# 清单格式变更（修复轮 1·主控裁决②）：4 段改 5 段，第 5 段为分组名。旧的 4 段清单会被
#       字段数门判退 2 并逐行点名，不会被静默按"无分组"接受。

set -uo pipefail
export LC_ALL=C

LIST_FILE=""
MAX_TIME=15

usage() {
  awk 'NR>1{ if (/^#/) { sub(/^# ?/,""); print; next } exit }' "$0"
}

# 工具哨兵排在参数解析之前：--list 的 realpath 与 usage() 的 awk 都在解析阶段就要用，
# 缺件时命令替换得到空串，空串在这里正是"无问题"值——不先验可执行会把环境问题读成通过。
for needed_tool in curl awk sed cat realpath; do
  command -v "$needed_tool" >/dev/null 2>&1 || \
    { echo "错误：缺少必要工具「$needed_tool」，无法探活。" >&2; exit 2; }
done

while [ $# -gt 0 ]; do
  case "$1" in
    --list)
      [ $# -ge 2 ] || { echo "错误：--list 缺少参数值。" >&2; usage >&2; exit 2; }
      # 路径一律先解析成绝对形（与 build 脚本的暂存目录同法）：后续所有报错与可读性判定
      # 都指向同一个确定路径，不随调用者 cwd 二义。
      list_abs=$(realpath -m -- "$2") || \
        { echo "错误：--list 路径解析失败：$2" >&2; exit 2; }
      [ -n "$list_abs" ] || \
        { echo "错误：--list 路径解析得到空串：$2" >&2; exit 2; }
      LIST_FILE=$list_abs; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未识别的参数「$1」。" >&2; usage >&2; exit 2 ;;
  esac
done

read_list() {
  if [ -n "$LIST_FILE" ]; then
    [ -r "$LIST_FILE" ] || { echo "错误：清单文件不可读：$LIST_FILE" >&2; return 1; }
    cat -- "$LIST_FILE"
  else
    cat <<'DEFAULT'
# 基础镜像代理：vllm/vllm-openai:v0.30.0-cu129（上游按 sha256 钉版，代理须支持同一 digest）
# 两条候选互为替代，归同一分组：任一可用即满足，全组红才退 3。
base_image_proxy_daocloud|registry|https://docker.m.daocloud.io/v2/vllm/vllm-openai/manifests/v0.30.0-cu129|BASE_IMAGE|base_image_proxy
base_image_proxy_1ms|registry|https://docker.1ms.run/v2/vllm/vllm-openai/manifests/v0.30.0-cu129|BASE_IMAGE|base_image_proxy
# PyTorch cu129 轮子索引：补丁内 ARG PYTORCH_MIRROR 的默认值（preflight 实测该路径 200）
pytorch_wheels_mirror|index|https://mirror.sjtu.edu.cn/pytorch-wheels/cu129/|PYTORCH_MIRROR|pytorch_wheels_mirror
# pypi 索引：补丁内 ARG PIP_MIRROR 与 ENV UV_DEFAULT_INDEX 的默认值
pypi_mirror|index|https://pypi.tuna.tsinghua.edu.cn/simple/|PIP_MIRROR|pypi_mirror
# 本仓托管件：URL 待任务 13 回填 Release 直链，探活阶段一律豁免
flashinfer_cubin_release|placeholder|PLACEHOLDER://release/flashinfer_cubin-0.6.18-py3-none-any.whl|无 env 开关（任务 13 改补丁）|release_hosted
nvidia_pstated_release|placeholder|PLACEHOLDER://release/nvidia-pstated|无 env 开关（任务 13 改补丁）|release_hosted
DEFAULT
  fi
}

LIST=$(read_list) || exit 2
[ -n "$LIST" ] || { echo "错误：清单内容为空，无源可探。" >&2; exit 2; }

# 整表校验排在任何网络请求之前：非法行若放到循环里才发现，前面的源已经白跑一轮。
badlines=$(printf '%s\n' "$LIST" | awk -F'|' '
  /^#/ { next }
  NF==0 { next }
  NF!=5 { print NR": 字段数 "NF"，应为 5 段「源名|类别|URL|切换开关名|分组」→ "$0; next }
  $1=="" { print NR": 源名不能为空 → "$0; next }
  $5=="" { print NR": 分组名不能为空 → "$0; next }
  $2=="placeholder" && $3 !~ /^PLACEHOLDER:\/\// { print NR": placeholder 类的 URL 必须用 PLACEHOLDER:// 前缀 → "$0; next }
  $2!="placeholder" && $3 ~ /^PLACEHOLDER:\/\// { print NR": URL 用了 PLACEHOLDER:// 前缀但类别不是 placeholder → "$0; next }
  $2!="placeholder" && $3 !~ /^https?:\/\// { print NR": 真实源 URL 必须是 http(s) 地址 → "$0; next }
  $2!="registry" && $2!="index" && $2!="placeholder" { print NR": 类别非法「"$2"」，只允许 registry|index|placeholder → "$0; next }
  { if ($1 in seen) print NR": 源名重复「"$1"」（第 "seen[$1]" 行已用过）→ "$0; else seen[$1]=NR }
')
if [ -n "$badlines" ]; then
  echo "错误：清单存在非法行，拒绝探活：" >&2
  printf '%s\n' "$badlines" | sed 's|^|    |' >&2
  exit 2
fi

real_total=$(printf '%s\n' "$LIST" | awk -F'|' '!/^#/ && NF==5 && $2!="placeholder" { n++ } END { print n+0 }')
case "$real_total" in
  ''|*[!0-9]*) echo "错误：无法统计真实源条数（awk 异常）。" >&2; exit 2 ;;
esac
if [ "$real_total" -eq 0 ]; then
  echo "错误：清单里没有真实源（全是占位或注释），探活无意义。" >&2
  exit 2
fi

echo "== 国内源探活（curl -I --max-time ${MAX_TIME}s，只读 HEAD，不下载）=="
printf '%-26s %-10s %-20s %-6s %-8s %s\n' "源名" "类别" "分组" "HTTP" "耗时s" "判定"

declare -A group_total
declare -A group_green
group_order=()
red_lines=()
seen_ids=()
green=0
exempt=0
while IFS='|' read -r sid skind surl shint sgroup; do
  # 这两条 continue 是"静默跳过"的唯一入口，且只可能命中注释行与空行（源名/分组名/URL
  # 已在整表校验判死）。收尾的一致性断言再兜一层：真有源被跳过就退 2 并点名。
  case "$sid" in ''|'#'*) continue ;; esac
  [ "${#surl}" -gt 0 ] || continue

  if [ "$skind" = "placeholder" ]; then
    printf '%-26s %-10s %-20s %-6s %-8s %s\n' "$sid" "$skind" "$sgroup" "-" "-" "豁免（待任务 13 回填）"
    echo "    URL：$surl"
    exempt=$((exempt + 1))
    continue
  fi

  if [ -z "${group_total[$sgroup]+x}" ]; then group_order+=("$sgroup"); fi
  group_total[$sgroup]=$(( ${group_total[$sgroup]:-0} + 1 ))

  raw=$(curl -sS -o /dev/null -I --max-time "$MAX_TIME" -w '%{http_code} %{time_total}' "$surl" 2>/dev/null)
  curl_rc=$?
  code=${raw%% *}
  secs=${raw##* }
  case "$code" in ''|*[!0-9]*) code="000" ;; esac
  case "$secs" in ''|*[!0-9.]*) secs="n/a" ;; esac

  is_red=0
  reason=""
  if [ "$curl_rc" -ne 0 ]; then
    is_red=1
    reason="curl rc=$curl_rc（网络 / DNS / TLS 失败或超时）"
  elif [ "$code" = "000" ]; then
    # "取不到状态码"不是"没问题"：curl 报告成功却拿不到 http_code，响应形态异常，判红。
    is_red=1
    reason="响应形态异常（curl rc=0，HTTP 状态码取不到）"
  elif [ "$code" = "401" ] && [ "$skind" = "registry" ]; then
    reason="绿（401＝registry 质询，token 由 docker CLI 换取）"
  elif [ "$code" -ge 400 ]; then
    is_red=1
    reason="HTTP $code"
  else
    reason="绿"
  fi
  printf '%-26s %-10s %-20s %-6s %-8s %s\n' "$sid" "$skind" "$sgroup" "$code" "$secs" "$reason"
  seen_ids+=("$sid")
  if [ "$is_red" -eq 1 ]; then
    red_lines+=("$sid|$surl|$shint|$sgroup")
  else
    green=$((green + 1))
    group_green[$sgroup]=$(( ${group_green[$sgroup]:-0} + 1 ))
  fi
done <<EOS
$LIST
EOS

# 一致性断言：真实源要么判绿要么判红，任何"绿＋红 ≠ 总数"都说明有源被静默跳过——
# 那正是本轮点名的假绿形态（清单被手工编辑过、字段数对但内容空等），一律退 2 并点名差额。
tally=$((green + ${#red_lines[@]}))
if [ "$tally" -ne "$real_total" ]; then
  echo "错误：判定计数与真实源总数不符：绿 $green ＋红 ${#red_lines[@]} ＝ $tally，" \
       "清单统计真实源 $real_total，差额 $((real_total - tally)) 条被静默跳过（本脚本退出码 2）。" >&2
  expected_ids=$(printf '%s\n' "$LIST" | awk -F'|' '!/^#/ && NF==5 && $2!="placeholder" { print $1 }')
  nl=$'\n'
  seen_blob=$nl
  if [ "${#seen_ids[@]}" -gt 0 ]; then seen_blob+=$(printf '%s\n' "${seen_ids[@]}"); fi
  seen_blob+=$nl
  skipped=()
  while IFS= read -r one_id; do
    [ -n "$one_id" ] || continue
    case "$nl$seen_blob$nl" in
      *"$nl$one_id$nl"*) : ;;
      *) skipped+=("$one_id") ;;
    esac
  done <<EIE
$expected_ids
EIE
  if [ "${#skipped[@]}" -gt 0 ]; then
    for one_skip in "${skipped[@]}"; do printf '    被跳过：源名「%s」\n' "$one_skip" >&2; done
  else
    printf '    未能按源名定位差额（同名列或计数异常），请逐行核对清单：%s\n' "${LIST_FILE:-内置清单}" >&2
  fi
  exit 2
fi

# 分组取或：组内有一条绿即满足；全组红才判红退 3。
fully_red=()
for gname in "${group_order[@]}"; do
  if [ "${group_green[$gname]:-0}" -eq 0 ]; then fully_red+=("$gname"); fi
done

echo "统计：真实源 $real_total 项（绿 $green、红 ${#red_lines[@]}），占位豁免 $exempt 项；分组 ${#group_order[@]} 个，全组红 ${#fully_red[@]} 个。"

if [ "${#red_lines[@]}" -gt 0 ]; then
  if [ "${#fully_red[@]}" -gt 0 ]; then
    echo "红：以下真实源不可用（其中含全组红的分组，本脚本退出码 3）：" >&2
  else
    echo "提示：以下红源所在分组另有可用候选，按分组取或不判红（本脚本仍退 0）：" >&2
  fi
  for entry in "${red_lines[@]}"; do
    rest=$entry
    printf '    源名：%s\n' "${rest%%|*}" >&2; rest=${rest#*|}
    printf '    URL： %s\n' "${rest%%|*}" >&2; rest=${rest#*|}
    printf '    开关： %s\n' "${rest%%|*}" >&2; rest=${rest#*|}
    printf '    分组： %s\n' "$rest" >&2
    printf '    处置： 换成可用地址后，构建时用同名 env / --build-arg 覆盖，再重跑本脚本。\n' >&2
  done
fi
if [ "${#fully_red[@]}" -gt 0 ]; then
  printf '红：全组红的分组：%s\n' "${fully_red[*]}" >&2
  exit 3
fi
echo "绿：每个分组至少有一条真实源可用；上表即逐源耗时。"
exit 0
