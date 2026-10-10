#!/usr/bin/env bash
# preflight-sources-v1.sh —— 构建期国内源探活（只读 HEAD 请求，不下载任何内容）
#
# 用法：
#   bash gpustack/preflight-sources-v1.sh                 按内置默认清单探活
#   bash gpustack/preflight-sources-v1.sh --list FILE     改用外部清单（自检与负控用）
#   清单格式：每行 源名|类别|URL|切换开关名，# 开头为注释，空行忽略
#   类别：registry（镜像仓库代理）| index（pypi / wheels 索引）| placeholder（待任务 13 回填）
#
# 退出码（本脚本）：
#   0 全部真实源绿（PLACEHOLDER 项豁免，不计红）
#   2 用法错误 / 清单不可读或非法 / 清单不含真实源 / 缺 curl
#   3 至少一个真实源红（逐条打印该源 URL＋对应切换开关名）
#
# 判绿口径：
#   index       —— 与 curl -fsI 等价：请求成功且 HTTP 状态码 < 400
#   registry    —— 同上，另把 HTTP 401 单列为绿：Docker Registry v2 规范下未带 token 访问
#                  /v2/ 一律回 401 质询，token 由 docker CLI 换取，HEAD 探活代替不了它。
#   placeholder —— 打印"豁免（待任务 13 回填）"并跳过，不产生红。
# 局限：HEAD 探活只证明"地址可达、索引在线"，不证明具体 wheel / 镜像 tag 在该镜像上存在，
#       也不证明内容与官方一致；这两件是阶段二任务 14 的实构建门禁项。

set -uo pipefail
export LC_ALL=C

LIST_FILE=""
MAX_TIME=15

usage() {
  awk 'NR>1{ if (/^#/) { sub(/^# ?/,""); print; next } exit }' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --list)
      [ $# -ge 2 ] || { echo "错误：--list 缺少参数值。" >&2; usage >&2; exit 2; }
      LIST_FILE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未识别的参数「$1」。" >&2; usage >&2; exit 2 ;;
  esac
done

for needed_tool in curl awk sed cat grep; do
  command -v "$needed_tool" >/dev/null 2>&1 || \
    { echo "错误：缺少必要工具「$needed_tool」，无法探活。" >&2; exit 2; }
done

read_list() {
  if [ -n "$LIST_FILE" ]; then
    [ -r "$LIST_FILE" ] || { echo "错误：清单文件不可读：$LIST_FILE" >&2; return 1; }
    cat -- "$LIST_FILE"
  else
    cat <<'DEFAULT'
# 基础镜像代理：vllm/vllm-openai:v0.30.0-cu129（上游按 sha256 钉版，代理须支持同一 digest）
base_image_proxy_daocloud|registry|https://docker.m.daocloud.io/v2/vllm/vllm-openai/manifests/v0.30.0-cu129|BASE_IMAGE
base_image_proxy_1ms|registry|https://docker.1ms.run/v2/vllm/vllm-openai/manifests/v0.30.0-cu129|BASE_IMAGE
# PyTorch cu129 轮子索引：补丁内 ARG PYTORCH_MIRROR 的默认值（preflight 实测该路径 200）
pytorch_wheels_mirror|index|https://mirror.sjtu.edu.cn/pytorch-wheels/cu129/|PYTORCH_MIRROR
# pypi 索引：补丁内 ARG PIP_MIRROR 与 ENV UV_DEFAULT_INDEX 的默认值
pypi_mirror|index|https://pypi.tuna.tsinghua.edu.cn/simple/|PIP_MIRROR
# 本仓托管件：URL 待任务 13 回填 Release 直链，探活阶段一律豁免
flashinfer_cubin_release|placeholder|PLACEHOLDER://release/flashinfer_cubin-0.6.18-py3-none-any.whl|无 env 开关（任务 13 改补丁）
nvidia_pstated_release|placeholder|PLACEHOLDER://release/nvidia-pstated|无 env 开关（任务 13 改补丁）
DEFAULT
  fi
}

LIST=$(read_list) || exit 2
[ -n "$LIST" ] || { echo "错误：清单内容为空，无源可探。" >&2; exit 2; }

# 整表校验排在任何网络请求之前：非法行若放到循环里才发现，前面的源已经白跑一轮。
badlines=$(printf '%s\n' "$LIST" | awk -F'|' '
  /^#/ { next }
  NF==0 { next }
  NF!=4 { print NR": 字段数 "NF"，应为 4 段「源名|类别|URL|切换开关名」→ "$0; next }
  $2=="placeholder" && $3 !~ /^PLACEHOLDER:\/\// { print NR": placeholder 类的 URL 必须用 PLACEHOLDER:// 前缀 → "$0; next }
  $2!="placeholder" && $3 ~ /^PLACEHOLDER:\/\// { print NR": URL 用了 PLACEHOLDER:// 前缀但类别不是 placeholder → "$0; next }
  $2!="placeholder" && $3 !~ /^https?:\/\// { print NR": 真实源 URL 必须是 http(s) 地址 → "$0; next }
  $2!="registry" && $2!="index" && $2!="placeholder" { print NR": 类别非法「"$2"」，只允许 registry|index|placeholder → "$0; next }
')
if [ -n "$badlines" ]; then
  echo "错误：清单存在非法行，拒绝探活：" >&2
  printf '%s\n' "$badlines" | sed 's|^|    |' >&2
  exit 2
fi

real_total=$(printf '%s\n' "$LIST" | awk -F'|' '!/^#/ && NF==4 && $2!="placeholder" { n++ } END { print n+0 }')
case "$real_total" in
  ''|*[!0-9]*) echo "错误：无法统计真实源条数（awk 异常）。" >&2; exit 2 ;;
esac
if [ "$real_total" -eq 0 ]; then
  echo "错误：清单里没有真实源（全是占位或注释），探活无意义。" >&2
  exit 2
fi

echo "== 国内源探活（curl -I --max-time ${MAX_TIME}s，只读 HEAD，不下载）=="
printf '%-28s %-11s %-6s %-8s %s\n' "源名" "类别" "HTTP" "耗时s" "判定"

red_lines=()
green=0
exempt=0
while IFS='|' read -r sid skind surl shint; do
  case "$sid" in ''|'#'*) continue ;; esac
  [ "${#surl}" -gt 0 ] || continue

  if [ "$skind" = "placeholder" ]; then
    printf '%-28s %-11s %-6s %-8s %s\n' "$sid" "$skind" "-" "-" "豁免（待任务 13 回填）"
    echo "    URL：$surl"
    exempt=$((exempt + 1))
    continue
  fi

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
  elif [ "$code" = "401" ] && [ "$skind" = "registry" ]; then
    reason="绿（401＝registry 质询，token 由 docker CLI 换取）"
  elif [ "$code" -ge 400 ]; then
    is_red=1
    reason="HTTP $code"
  else
    reason="绿"
  fi
  printf '%-28s %-11s %-6s %-8s %s\n' "$sid" "$skind" "$code" "$secs" "$reason"
  if [ "$is_red" -eq 1 ]; then
    red_lines+=("$sid|$surl|$shint")
  else
    green=$((green + 1))
  fi
done <<EOS
$LIST
EOS

echo "统计：真实源 $real_total 项（绿 $green、红 ${#red_lines[@]}），占位豁免 $exempt 项。"

if [ "${#red_lines[@]}" -gt 0 ]; then
  echo "红：以下真实源不可用（本脚本退出码 3）：" >&2
  for entry in "${red_lines[@]}"; do
    printf '    源名：%s\n' "${entry%%|*}" >&2
    rest=${entry#*|}
    printf '    URL： %s\n' "${rest%%|*}" >&2
    printf '    开关： %s\n' "${rest#*|}" >&2
    printf '    处置： 换成可用地址后，构建时用同名 env / --build-arg 覆盖，再重跑本脚本。\n' >&2
  done
  exit 3
fi
echo "绿：全部真实源可达；上表即逐源耗时。"
exit 0
