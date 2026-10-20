#!/usr/bin/env python3
"""Panel autostart wrapper for the GPUStack backend image.

Purpose: one model row should bring up both the vLLM engine and the SM75 console,
so an operator does not have to run a second docker command. Enabled purely by env:

    SM75_PANEL_AUTOSTART=1        # 1/true/yes/on starts the console before the engine
    SM75_CONSOLE_ROOT=<dir>       # panel state; default /var/lib/gpustack/panel
                                  # (inside the worker data dir, so GPUStack persists it on the host)

Without SM75_PANEL_AUTOSTART this is a transparent passthrough: it execs exactly the
command GPUStack gave the container, so the plain "backend image" behaviour is unchanged.
The console is started detached in its own session; if it dies, the engine is unaffected.

When the engine command is recognisable (vllm serve ... or runtime-entrypoint.py ...),
its --host/--port/--served-model-name/model path/--api-key are exported to the console as
SM75_ENGINE_* so the console can observe the GPUStack-owned engine instead of trying to
start one itself.
"""
import json
import os
import secrets
import shutil
import subprocess
import sys

NODE = "/usr/local/bin/node"
SERVER = "/opt/sm75-workbench/console/server.mjs"
DEFAULT_ROOT = "/var/lib/gpustack/panel"
WORKBENCH = "/opt/sm75-workbench"
VERIFIER_ROOT = WORKBENCH + "/benchlocal-verifiers"
BAKED_EXTENSIONS = WORKBENCH + "/extensions"
VERIFIER_ROLES = {
    "bugfind-15": ("4211", "4011"),
    "cli-40": ("4212", "4012"),
    "hermesagent-20": ("4213", "4013"),
}


def extension_set():
    return {
        item.strip()
        for item in str(os.environ.get("SM75_EXTENSIONS", "")).split(",")
        if item.strip()
    }


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def flag_value(args, name):
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def engine_args(argv):
    """Return (model_args, model_path) from the command GPUStack handed us."""
    args = list(argv)
    if args and ("serve" in args):
        args = args[args.index("serve") + 1:]
    elif args and args[0].endswith("vllm"):
        args = args[1:]
    elif len(args) > 1 and args[0].endswith("runtime-entrypoint.py"):
        args = args[1:]
    model_path = flag_value(args, "--model")
    if not model_path:
        for index, arg in enumerate(args):
            if arg.startswith("-"):
                continue
            # A flag value is not a model path.
            if index > 0 and args[index - 1] in (
                "--host", "--port", "--served-model-name", "--api-key",
                "--dtype", "--max-model-len", "--tensor-parallel-size",
            ):
                continue
            model_path = arg
            break
    return args, model_path


def engine_environment(argv):
    args, model_path = engine_args(argv)
    if not args:
        return {}
    api_key = flag_value(args, "--api-key")
    if api_key:
        stripped = []
        skip = False
        for arg in args:
            if skip:
                skip = False
                continue
            if arg == "--api-key":
                skip = True
                continue
            if arg.startswith("--api-key="):
                continue
            stripped.append(arg)
        args = stripped
    host = flag_value(args, "--host") or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]", ""):
        host = "127.0.0.1"
    port = flag_value(args, "--port") or "8000"
    served = flag_value(args, "--served-model-name")
    model = served or (os.path.basename(model_path.rstrip("/")) if model_path else "model")
    env = {
        "SM75_EXTERNAL_ENGINE": "1",
        "SM75_ENGINE_URL": "http://%s:%s" % (host, port),
        "SM75_ENGINE_ARGS": json.dumps(args, ensure_ascii=False),
        "SM75_ENGINE_MODEL": model,
    }
    if model_path:
        env["SM75_ENGINE_MODEL_PATH"] = model_path
    if api_key:
        env["SM75_ENGINE_KEY"] = api_key
    return env


def copy_baked_extensions(root, extensions):
    """Install test assets baked into the image into the persistent panel root."""
    for kind in ("speedtest", "sql"):
        if kind not in extensions:
            continue
        source = os.path.join(BAKED_EXTENSIONS, kind)
        target = os.path.join(root, "extensions", kind)
        if not os.path.isfile(os.path.join(source, "manifest.json")):
            continue
        if os.path.isfile(os.path.join(target, "manifest.json")):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
            print("EXTENSION_READY %s -> %s" % (kind, target), file=sys.stderr)
        except OSError as exc:
            print("EXTENSION_SKIP %s: %s" % (kind, exc), file=sys.stderr)


VERIFIER_CONFIG_DEFAULT = "/run/sm75-bench/benchlocal.env"
VERIFIER_STATE_DEFAULT = "/run/sm75-bench"


def read_verifier_config(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name in {
                    "SM75_BENCH_VERIFIER_KEY",
                    "SM75_BENCH_VERIFIERS",
                    "SM75_BENCH_PROXY_BASE",
                }:
                    values[name] = value.strip()
    except OSError:
        pass
    return values


def start_external_verifiers(env, root):
    """Use host-managed verifier containers instead of spawning local gateways."""
    config_path = str(
        env.get("SM75_BENCH_VERIFIER_CONFIG_FILE") or VERIFIER_CONFIG_DEFAULT
    )
    values = read_verifier_config(config_path)
    key = values.get("SM75_BENCH_VERIFIER_KEY") or str(
        env.get("SM75_BENCH_VERIFIER_KEY") or ""
    )
    verifiers = values.get("SM75_BENCH_VERIFIERS") or str(
        env.get("SM75_BENCH_VERIFIERS") or ""
    )
    proxy_base = values.get("SM75_BENCH_PROXY_BASE") or str(
        env.get("SM75_BENCH_PROXY_BASE") or ""
    )
    if len(key) < 32 or not verifiers or not proxy_base:
        print(
            "BENCH_VERIFIERS_EXTERNAL_MISSING config=%s" % config_path,
            file=sys.stderr,
        )
        return
    try:
        parsed = json.loads(verifiers)
    except ValueError as exc:
        print("BENCH_VERIFIERS_EXTERNAL_INVALID %s" % exc, file=sys.stderr)
        return
    if not isinstance(parsed, dict) or not all(
        isinstance(value, str) and value.startswith(("http://", "https://"))
        for value in parsed.values()
    ):
        print("BENCH_VERIFIERS_EXTERNAL_INVALID urls", file=sys.stderr)
        return
    state_dir = str(
        env.get("SM75_BENCH_VERIFIER_STATE_DIR")
        or os.path.dirname(config_path)
        or VERIFIER_STATE_DEFAULT
    )
    env["SM75_BENCH_VERIFIER_KEY"] = key
    env["SM75_BENCH_VERIFIERS"] = verifiers
    env["SM75_BENCH_PROXY_BASE"] = proxy_base
    env["SM75_BENCH_VERIFIER_STATE_DIR"] = state_dir
    print("BENCH_VERIFIERS_EXTERNAL %s" % verifiers, file=sys.stderr)


def start_verifiers(env, root, extensions):
    """Run the three BenchLocal gateways inside this container (no GPU access)."""
    if "benchlocal" not in extensions:
        return
    mode = str(env.get("SM75_BENCH_VERIFIERS_MODE") or "local").strip().lower()
    if mode == "external":
        start_external_verifiers(env, root)
        return
    key = str(env.get("SM75_BENCH_VERIFIER_KEY") or "")
    if len(key) < 32:
        key = secrets.token_urlsafe(32)
    console_port = str(env.get("SM75_CONSOLE_PORT") or "1616")
    origin = "http://127.0.0.1:%s" % console_port
    env["SM75_BENCH_VERIFIER_KEY"] = key
    env["SM75_BENCH_PROXY_BASE"] = origin + "/benchlocal-inference/v1"
    os.makedirs("/home/verifier", exist_ok=True)
    try:
        os.chown("/home/verifier", 10001, 10001)
    except OSError:
        pass
    urls = {}
    for role, (port, inner) in VERIFIER_ROLES.items():
        role_root = os.path.join(VERIFIER_ROOT, role)
        if not os.path.isdir(role_root):
            print("VERIFIER_SKIP %s：镜像内缺少角色目录" % role, file=sys.stderr)
            continue
        child_env = dict(env)
        child_env.update(
            SM75_BENCH_VERIFIER_ROOT=role_root,
            SM75_VERIFIER_PORT=port,
            SM75_VERIFIER_INNER_PORT=inner,
            SM75_BENCH_PROXY_ORIGIN=origin,
            NVIDIA_VISIBLE_DEVICES="void",
            CUDA_VISIBLE_DEVICES="",
            PATH="/opt/hermes-venv/bin:/opt/sm75-bench/rust/bin:/usr/local/go/bin:"
                 + child_env.get("PATH", ""),
            AGENT_BROWSER_EXECUTABLE_PATH="/opt/sm75-bench/chrome/chrome-linux64/chrome",
        )
        log_path = os.path.join(root, "verifier-%s.log" % role)
        with open(log_path, "ab", buffering=0) as log:
            subprocess.Popen(
                [NODE, os.path.join(VERIFIER_ROOT, "gateway.mjs")],
                env=child_env, stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        urls[role] = "http://127.0.0.1:%s" % port
    if urls:
        env["SM75_BENCH_VERIFIERS"] = json.dumps(urls)
        print("BENCH_VERIFIERS_STARTED %s" % json.dumps(urls), file=sys.stderr)


def start_panel(argv):
    root = os.environ.get("SM75_CONSOLE_ROOT") or DEFAULT_ROOT
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        print("PANEL_SKIP 无法创建 %s: %s" % (root, exc), file=sys.stderr)
        return
    if not (os.path.isfile(NODE) and os.path.isfile(SERVER)):
        print("PANEL_SKIP 缺少 %s 或 %s" % (NODE, SERVER), file=sys.stderr)
        return
    extensions = extension_set()
    if "panel" in extensions and not truthy(os.environ.get("SM75_PANEL_AUTOSTART")):
        os.environ["SM75_PANEL_AUTOSTART"] = "1"
    copy_baked_extensions(root, extensions)
    env = dict(os.environ)
    env.update(engine_environment(argv))
    env["SM75_CONSOLE_ROOT"] = root
    env.setdefault("SM75_DSH_ROOT", "/var/lib/gpustack/dsh")
    env.setdefault("SM75_EDITION", "ultra")
    env.setdefault("SM75_CONSOLE_PORT", "1616")
    if env["SM75_EDITION"] != "ultra":
        env["SM75_EDITION"] = "ultra"
    start_verifiers(env, root, extensions)
    log_path = os.path.join(root, "panel.log")
    with open(log_path, "ab", buffering=0) as log:
        # 独立会话：面板不随引擎退出而被连带杀掉，也不抢占容器的 stdout。
        subprocess.Popen([NODE, SERVER], env=env, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print("PANEL_STARTED root=%s log=%s engine=%s extensions=%s" % (
        root, log_path, env.get("SM75_ENGINE_URL", "not-detected"),
        ",".join(sorted(extensions)) or "panel"), file=sys.stderr)


def main():
    if truthy(os.environ.get("SM75_PANEL_AUTOSTART")):
        start_panel(sys.argv[1:])
    if len(sys.argv) < 2:
        print("panel-sidecar: nothing to exec（GPUStack 未给命令时不空转）", file=sys.stderr)
        return 2
    os.execvp(sys.argv[1], sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
