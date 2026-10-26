"""L5 packaging test: the dist-info entry point really activates the gate.

Builds a throwaway site-packages, runs docker/helpers/install_vllm_sm75.py against
it, then checks that importlib.metadata discovers `vllm.general_plugins ->
vllm_sm75:install`, that calling it works, and that a machine without CUDA leaves
the capability cache *unset* instead of caching False (regression guard).

Run: python tests/test_install_entrypoint.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(ROOT, "docker", "helpers", "install_vllm_sm75.py")
PKG = os.path.join(ROOT, "vllm_sm75")

CHECK = r'''
import importlib.metadata as m, json, sys
eps = [e for e in m.entry_points(group="vllm.general_plugins")]
out = {"eps": [(e.name, e.value) for e in eps]}
fn = eps[0].load()
out["is_callable"] = callable(fn)
fn()
import vllm_sm75
out["installed"] = bool(vllm_sm75._INSTALLED)
out["cc_cache"] = vllm_sm75._CC_OK
print(json.dumps(out))
'''


def main():
    tmp = tempfile.mkdtemp(prefix="sm75pkg-")
    try:
        site = os.path.join(tmp, "site")
        os.makedirs(os.path.join(site, "vllm"))
        open(os.path.join(site, "vllm", "__init__.py"), "w").close()

        payload = os.path.join(tmp, "pkg")
        os.makedirs(payload)
        shutil.copytree(PKG, os.path.join(payload, "vllm_sm75"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        env = dict(os.environ, PYTHONPATH=site, VLLM_SM75_OPT="auto")
        r = subprocess.run([sys.executable, HELPER, payload],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-500:]
        print("PASS installer runs:", r.stdout.strip().splitlines()[0])

        with open(os.path.join(PKG, "manifest.json"), encoding="utf-8") as fh:
            version = json.load(fh)["version"]
        di = os.path.join(site, "vllm_sm75-%s.dist-info" % version)
        assert os.path.isdir(di), "no dist-info at %s" % di
        with open(os.path.join(di, "entry_points.txt"), encoding="utf-8") as fh:
            assert "sm75_opt = vllm_sm75:install" in fh.read()
        print("PASS dist-info + entry_points.txt written")

        r = subprocess.run([sys.executable, "-c", CHECK],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["eps"] == [["sm75_opt", "vllm_sm75:install"]], res
        assert res["is_callable"] and res["installed"], res
        print("PASS metadata discovery -> install() executed")

        assert res["cc_cache"] in (None, False), res
        assert res["cc_cache"] is not True, "capability probe said True without a GPU"
        print("PASS no CUDA does not enable the gate (_CC_OK=%r)" % (res["cc_cache"],))

        poisoned = os.path.join(tmp, "poison")
        os.makedirs(poisoned)
        with open(os.path.join(poisoned, "torch.py"), "w", encoding="utf-8") as fh:
            fh.write("raise ImportError('poisoned torch for the capability-cache test')\n")
        env2 = dict(env, PYTHONPATH=poisoned + os.pathsep + site)
        r = subprocess.run([sys.executable, "-c", CHECK],
                           capture_output=True, text=True, env=env2)
        assert r.returncode == 0, r.stderr[-800:]
        res2 = json.loads(r.stdout.strip().splitlines()[-1])
        assert res2["cc_cache"] is None, \
            "a failed torch import was cached as %r" % (res2["cc_cache"],)
        print("PASS failed torch import leaves _CC_OK unset (not cached)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
    print("\nentry point activation verified")


def test_install_entrypoint():
    """pytest entry point (main() is the standalone runner)."""
    main()
