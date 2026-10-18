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
"""
import os
import subprocess
import sys

NODE = "/usr/local/bin/node"
SERVER = "/opt/sm75-workbench/console/server.mjs"
DEFAULT_ROOT = "/var/lib/gpustack/panel"


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def start_panel():
    root = os.environ.get("SM75_CONSOLE_ROOT") or DEFAULT_ROOT
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        print("PANEL_SKIP 无法创建 %s: %s" % (root, exc), file=sys.stderr)
        return
    if not (os.path.isfile(NODE) and os.path.isfile(SERVER)):
        print("PANEL_SKIP 缺少 %s 或 %s" % (NODE, SERVER), file=sys.stderr)
        return
    env = dict(os.environ)
    env["SM75_CONSOLE_ROOT"] = root
    env.setdefault("SM75_EDITION", "ultra")
    log_path = os.path.join(root, "panel.log")
    with open(log_path, "ab", buffering=0) as log:
        # 独立会话：面板不随引擎退出而被连带杀掉，也不抢占容器的 stdout。
        subprocess.Popen([NODE, SERVER], env=env, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print("PANEL_STARTED root=%s log=%s" % (root, log_path), file=sys.stderr)


def main():
    if truthy(os.environ.get("SM75_PANEL_AUTOSTART")):
        start_panel()
    if len(sys.argv) < 2:
        print("panel-sidecar: nothing to exec（GPUStack 未给命令时不空转）", file=sys.stderr)
        return 2
    os.execvp(sys.argv[1], sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
