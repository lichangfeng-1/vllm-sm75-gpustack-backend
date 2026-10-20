#!/bin/bash
# set-panel-autostart-v1.sh —— 一次把"面板随模型自启"接好（默认 dry-run，只打印不改）
#
# 做五件事：
#   ① 后端版本 0.1.7-sm75.1-fast-custom（默认交付档，镜像 -v7）的 entrypoint 指到 panel-sidecar 包装，并把 IMAGE_NAME 写进该版本；
#   ② 模型行 env 追加 SM75_PANEL_AUTOSTART=1 / SM75_CONSOLE_HOST=0.0.0.0 / SM75_CONSOLE_ROOT=/var/lib/gpustack/panel / SM75_CONSOLE_PORT=1615；
#   ③ 模型行 env 追加 SM75_EXTENSIONS（默认 panel,harness,benchlocal,speedtest,sql，可整体覆盖）；
#   ④ 追加 BenchLocal external 三格（v7 镜像生效；v6 镜像会忽略并继续本地校验器）：
#      SM75_BENCH_VERIFIERS_MODE / SM75_BENCH_VERIFIER_CONFIG_FILE / SM75_BENCH_VERIFIER_STATE_DIR。
#   ⑤ 追加编译缓存六格（默认 /var/lib/gpustack/cache；首次填、二启命中；不想要可删这六格）。
#      扩展依赖与两套测试资产已烘进镜像；不需要时把 EXTENSIONS 设为 panel 或删掉该变量。
#
# 用法：
#   GS_KEY=<API key> HOST=<ip:8180> MODEL_ID=<模型 id> bash gpustack/set-panel-autostart-v1.sh          # 只看将改什么
#   GS_KEY=<API key> HOST=<ip:8180> MODEL_ID=<模型 id> bash gpustack/set-panel-autostart-v1.sh --apply  # 真改
#   VERSION_NAME=0.1.7-sm75.1-custom ... bash gpustack/set-panel-autostart-v1.sh --apply               # v4 回退路线
#   EXTENSIONS=panel GS_KEY=... bash gpustack/set-panel-autostart-v1.sh --apply                        # 只起面板
#
# 依赖：curl、python3。改完要重启实例（平台按 replicas 重建），或用 UI 停/起一次。
set -eu
umask 077
WORK=$(mktemp -d)
export WORK
trap 'rm -rf "$WORK"' EXIT
CURL_CFG="$WORK/curl.cfg"
printf 'header = "X-API-Key: %s"
' "$GS_KEY" > "$CURL_CFG"
chmod 600 "$CURL_CFG"
: "${GS_KEY:?需要 GS_KEY（平台 API key，别写进脚本）}"
: "${HOST:?需要 HOST，形如 172.17.100.10:8180}"
MODEL_ID="${MODEL_ID:-4}"
BACKEND_ID="${BACKEND_ID:-1}"
VERSION_NAME="${VERSION_NAME:-0.1.7-sm75.1-fast-custom}"
ENTRYPOINT="${ENTRYPOINT:-python3 /opt/vllm-sm75/panel-sidecar.py vllm serve}"
EXTENSIONS="${EXTENSIONS:-panel,harness,benchlocal,speedtest,sql}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

api() { curl -sS -m 20 -K "$CURL_CFG" "$@"; }

echo "== 读后端版本 =="
api "http://${HOST}/v2/inference-backends/${BACKEND_ID}" -o "$WORK/spa-backend.json"
echo "== 读模型行 =="
api "http://${HOST}/v2/models/${MODEL_ID}" -o "$WORK/spa-model.json"

python3 - "$VERSION_NAME" "$ENTRYPOINT" "$EXTENSIONS" <<'PY'
import json, os, sys
ver, ep, extensions = sys.argv[1], sys.argv[2], sys.argv[3]
b = json.load(open(os.path.join(os.environ["WORK"], "spa-backend.json"), encoding="utf-8"))
if "code" in b:
    sys.exit("读后端失败: %s" % b)
vc = dict(b.get("version_configs") or {})
if ver not in vc:
    sys.exit("版本 %s 不在该后端里（现有：%s）" % (ver, list(vc)))
cfg = dict(vc[ver])
print("  entrypoint: %r -> %r" % (cfg.get("entrypoint"), ep))
cfg["entrypoint"] = ep
if os.environ.get("IMAGE_NAME"):
    old_image = cfg.get("image_name")
    cfg["image_name"] = os.environ["IMAGE_NAME"]
    print("  version image: %r -> %r" % (old_image, cfg["image_name"]))
vc[ver] = cfg
json.dump({"backend_name": b.get("backend_name"), "version_configs": vc},
          open(os.path.join(os.environ["WORK"], "spa-backend-put.json"), "w"), ensure_ascii=False)

m = json.load(open(os.path.join(os.environ["WORK"], "spa-model.json"), encoding="utf-8"))
if "code" in m:
    sys.exit("读模型失败: %s" % m)
env = dict(m.get("env") or {})
want = {"SM75_PANEL_AUTOSTART": "1", "SM75_CONSOLE_HOST": "0.0.0.0",
        "SM75_CONSOLE_ROOT": "/var/lib/gpustack/panel",
        "SM75_CONSOLE_PORT": "1615",
        "SM75_EXTENSIONS": extensions,
        "SM75_BENCH_VERIFIERS_MODE": "external",
        "SM75_BENCH_VERIFIER_CONFIG_FILE": "/var/lib/gpustack/benchlocal/benchlocal.env",
        "SM75_BENCH_VERIFIER_STATE_DIR": "/var/lib/gpustack/benchlocal",
        "FLASHINFER_WORKSPACE_BASE": "/var/lib/gpustack/cache",
        "HUMMING_CACHE_DIR": "/var/lib/gpustack/cache/humming",
        "CUDA_CACHE_PATH": "/var/lib/gpustack/cache/cuda",
        "TRITON_CACHE_DIR": "/var/lib/gpustack/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/var/lib/gpustack/cache/inductor",
        "TORCH_EXTENSIONS_DIR": "/var/lib/gpustack/cache/torch_extensions"}
for k, v in want.items():
    print("  env %-22s %r -> %r" % (k, env.get(k), v))
    env[k] = v
# PUT /v2/models 会清掉未提交的 image_name（10-18 实撞）；必须显式带回。
image_name = (os.environ.get("IMAGE_NAME") or m.get("image_name")
              or (vc.get(ver) or {}).get("image_name"))
if not image_name:
    sys.exit("拒绝提交：模型行与后端版本都没有 image_name，先在后端版本里补")
print("  image_name: %r -> %r" % (m.get("image_name"), image_name))
body = {"name": m.get("name"), "source": m.get("source"), "env": env,
        "image_name": image_name,
        "backend": m.get("backend"), "backend_version": m.get("backend_version")}
for k in ("model_scope_model_id", "huggingface_repo_id", "local_path"):
    if m.get(k):
        body[k] = m[k]
json.dump(body, open(os.path.join(os.environ["WORK"], "spa-model-put.json"), "w"), ensure_ascii=False)
print("  两份 PUT body 已写好：%s、%s" % (os.path.join(os.environ["WORK"], "spa-backend-put.json"), os.path.join(os.environ["WORK"], "spa-model-put.json")))
PY

if [ "$APPLY" != "1" ]; then
  echo "DRY_RUN 未提交。加 --apply 才会 PUT。"
  exit 0
fi

echo "== 提交后端版本 =="
api -X PUT -H "Content-Type: application/json" --data @"$WORK/spa-backend-put.json" \
  "http://${HOST}/v2/inference-backends/${BACKEND_ID}" -o "$WORK/spa-backend-out.json"
echo "== 提交模型行 =="
api -X PUT -H "Content-Type: application/json" --data @"$WORK/spa-model-put.json" \
  "http://${HOST}/v2/models/${MODEL_ID}" -o "$WORK/spa-model-out.json"
python3 - <<'PY'
import json, os
for name in ("backend", "model"):
    d = json.load(open(os.path.join(os.environ["WORK"], "spa-%s-out.json" % name), encoding="utf-8"))
    if "code" in d:
        raise SystemExit("%s PUT 被拒: %s" % (name, str(d)[:300]))
print("APPLIED 后端 entrypoint 与模型 env 已写入；重启实例后生效（面板 :1615）")
PY

# PUT 会清未提交的 image_name；提交后回读断言，防止静默丢镜像名。
api "http://${HOST}/v2/models/${MODEL_ID}" -o "$WORK/spa-model-after.json"
python3 - <<'PY'
import json, os
m = json.load(open(os.path.join(os.environ["WORK"], "spa-model-after.json"), encoding="utf-8"))
image = m.get("image_name")
if not image:
    raise SystemExit("红：PUT 后模型行 image_name 为空（平台会按默认解析，可能拉错镜像）")
print("CHECK image_name=%s" % image)
PY
