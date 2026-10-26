"""Install the gated vllm_sm75 package into the vLLM site-packages (image build).

Writes a real dist-info with a ``vllm.general_plugins`` entry point so
``load_general_plugins()`` calls ``vllm_sm75.install()`` in every engine process
(API server and each TP worker) without editing any vLLM file.

Usage: python3 install_vllm_sm75.py <payload_dir_containing_vllm_sm75>
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ENTRY = "[vllm.general_plugins]\nsm75_opt = vllm_sm75:install\n"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    payload = Path(sys.argv[1])
    src = payload / "vllm_sm75"
    if not src.is_dir():
        raise SystemExit("missing payload: %s" % src)
    site = Path(importlib.util.find_spec("vllm").origin).parent.parent

    dst = site / "vllm_sm75"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    version = json.loads((dst / "manifest.json").read_text())["version"]
    di = site / ("vllm_sm75-%s.dist-info" % version)
    if di.exists():
        shutil.rmtree(di)
    di.mkdir()
    (di / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm-sm75\nVersion: %s\n" % version)
    (di / "top_level.txt").write_text("vllm_sm75\n")
    (di / "entry_points.txt").write_text(ENTRY)
    (di / "RECORD").write_text("")
    print("installed vllm_sm75 %s -> %s" % (version, dst))
    print("entry point: %s" % ENTRY.strip().replace("\n", " | "))


if __name__ == "__main__":
    main()
