"""Item registry — manifest.json is the single source of truth.

Rules:
* every item defaults to **off**;
* an item may be listed in ``auto_enabled`` only after it passed L2 (numerics),
  L3 (end-to-end fingerprint/PPL) and L4 (same-ruler A/B, >= 1%);
* an item marked ``not implemented`` or ``already_fused_upstream`` is never imported;
* an item's module is ``vllm_sm75.<item name>``.

Status values and their meaning are documented in
``GPUStack使用SM75\\规范-SM75优化交付形态与CI门禁-v1.md``.
"""
import json
import os

_MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")

with open(_MANIFEST_PATH, encoding="utf-8") as _fh:
    MANIFEST = json.load(_fh)

ITEMS = MANIFEST["items"]
AUTO_ENABLED = set(MANIFEST.get("auto_enabled") or [])

_SKIP = ("not implemented", "already_fused_upstream")


def installable(enabled_fn):
    """Item names that ``install_all`` would import, in manifest order."""
    return [name for name, meta in ITEMS.items()
            if meta.get("status") not in _SKIP and enabled_fn(name)]


def install_all(enabled_fn):
    for name in installable(enabled_fn):
        mod = __import__("vllm_sm75." + name, fromlist=["install"])
        mod.install()
