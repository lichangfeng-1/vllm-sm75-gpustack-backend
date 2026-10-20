#!/usr/bin/env python3
"""gen-allinone-dockerfile-v1.py — 把上游 docker/Dockerfile 与我们五层合成一份自包含 Dockerfile。

为什么：仓库要自己掌握构建定义，不依赖"先构建基座镜像再 FROM"的中间产物；
上游原文仍是冻结文件（SHA256SUMS-upstream），本脚本只读它，生成物 gpustack/Dockerfile.allinone
才是实际构建入口。上游破坏性更新时，我们只需要决定是否重新生成；不重新生成也不影响现有可用性。

用法：
  python3 gpustack/gen-allinone-dockerfile-v1.py [--check]
    --check  只比对生成物是否与当前输入一致（不一致 rc=2），不写盘
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "docker" / "Dockerfile"
MIRROR_PATCH = ROOT / "gpustack" / "mirror-patch-v1.patch"
LAYERS = ROOT / "gpustack" / "Dockerfile.v1"
OUT = ROOT / "gpustack" / "Dockerfile.allinone"
MARK_BEGIN = "# ===================== 层 1：pcieipc"
NODE_ARG_PREFIX = "ARG NODE_IMAGE="


def patched_upstream() -> str:
    """上游原文 + mirror-patch（与两段构建的基座一致；补丁只改 docker/Dockerfile）。"""
    raw = UPSTREAM.read_text(encoding="utf-8")
    if not MIRROR_PATCH.exists():
        raise SystemExit("红：缺少 gpustack/mirror-patch-v1.patch")
    if not shutil.which("patch"):
        raise SystemExit("红：PATH 上没有 patch 命令，无法把镜像源补丁应用到上游段")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "docker"
        d.mkdir()
        (d / "Dockerfile").write_text(raw, encoding="utf-8")
        r = subprocess.run(["patch", "-p1", "--batch", "-i", str(MIRROR_PATCH)],
                           cwd=td, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("红：mirror-patch 应用失败：%s%s" % (r.stdout, r.stderr))
        return (d / "Dockerfile").read_text(encoding="utf-8")


def build() -> str:
    up = patched_upstream().rstrip("\n")
    layers = LAYERS.read_text(encoding="utf-8").rstrip("\n")

    if "# syntax=docker/dockerfile:1.7" not in up.splitlines()[0]:
        raise SystemExit("红：上游 Dockerfile 首行不是 syntax 指令，生成器需要复核")
    if "FROM verified-runtime AS final" not in up:
        raise SystemExit("红：上游最终 stage 不是 final，生成器需要复核")
    if MARK_BEGIN not in layers:
        raise SystemExit("红：我们的层定义缺少层 1 标记")

    # 只取我们的层段（从层 1 开始）；去掉重复 syntax 行与 BASE_TAG 依赖。
    idx = layers.index(MARK_BEGIN)
    seg = layers[idx:]

    # NODE_IMAGE 必须作为全局 ARG 出现在第一个 FROM 之前；从层段里提出来，插到上游 ARG BASE_IMAGE 之后。
    node_arg = None
    kept = []
    for line in seg.splitlines():
        if line.startswith(NODE_ARG_PREFIX):
            node_arg = line
            continue
        if line.strip() == "ARG BASE_TAG":
            continue
        kept.append(line)
    seg = "\n".join(kept)
    seg = seg.replace("FROM ${BASE_TAG} AS pcieipc-layer", "FROM final AS pcieipc-layer")
    seg = seg.replace("${BASE_TAG}", "upstream-final")
    if "FROM ${BASE_TAG}" in seg:
        raise SystemExit("红：仍有未替换的 BASE_TAG FROM")

    lines = up.splitlines()
    if node_arg is None:
        raise SystemExit("红：层定义里找不到 ARG NODE_IMAGE")
    insert_at = 1
    for i, line in enumerate(lines[:10]):
        if line.startswith("ARG BASE_IMAGE="):
            insert_at = i + 1
            break
    lines.insert(insert_at, node_arg)
    lines.insert(insert_at + 1, "")

    header = [
        "# 本文件由 gpustack/gen-allinone-dockerfile-v1.py 生成，勿手改。",
        "# 前半段＝上游 docker/Dockerfile 原文＋gpustack/mirror-patch-v1.patch（与两段构建的基座同内容），",
        "# 后半段＝本仓五层；构建入口：docker build -f gpustack/Dockerfile.allinone --target runtime-pin-layer .",
        "",
    ]
    # 首行必须是上游的 syntax 指令；我们的说明放在它之后。
    first = lines[0]
    rest = lines[1:]
    return "\n".join([first, ""] + header + rest + [
        "",
        "# ===================== 以下为本仓五层（吸收自 gpustack/Dockerfile.v1） =====================",
        seg,
        "",
    ])


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists():
            print("RED missing", OUT)
            return 2
        same = OUT.read_text(encoding="utf-8") == text
        print("CHECK", "SAME" if same else "DIFF", hashlib.sha256(text.encode()).hexdigest()[:16])
        return 0 if same else 2
    OUT.write_text(text, encoding="utf-8")
    print("WROTE", OUT, len(text.encode()), "bytes sha256", hashlib.sha256(text.encode()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
