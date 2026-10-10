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
for needed_tool in awk grep sed mktemp rm wc mkdir; do
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

# 证据文件要落盘，指向仓内就等于往交付面里塞测试垃圾（还会被上游护栏按新增报红）。
OUT_DIR_ABS=$(cd -- "$(dirname -- "$OUT_FILE")" 2>/dev/null && pwd)
[ -n "$OUT_DIR_ABS" ] || bad_usage "证据输出目录不存在或不可进入：$(dirname -- "$OUT_FILE")"
OUT_ABS="$OUT_DIR_ABS/$(basename -- "$OUT_FILE")"
REPO_ROOT_G=$(cd -- "$SCRIPT_DIR/../.." && pwd)
case "$OUT_ABS" in
  "$REPO_ROOT_G"|"$REPO_ROOT_G"/*)
    bad_usage "证据输出路径不得在仓内（$OUT_ABS 落在 $REPO_ROOT_G 之下），请指到沙箱或镜像路径。" ;;
esac
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
