"""Item registry.

Rules:
* every item defaults to **off**;
* an item may be listed in ``AUTO_ENABLED`` only after it passed L2 (numerics), L3
  (end-to-end fingerprint/PPL) and L4 (same-ruler A/B, >= 3%);
* an item marked ``not implemented`` is never imported.

See ``GPUStack使用SM75\\规范-SM75优化交付形态与CI门禁-v1.md`` for the gate definitions.
"""

AUTO_ENABLED = set()

ITEMS = {
    # name: {"module": str, "default": "off", "status": str, "l4_gain_pct": float|None}
    "hc_fuse": {"module": "hc_fuse", "default": "off", "status": "not implemented", "l4_gain_pct": None},
    "gdn_fuse": {"module": "gdn_fuse", "default": "off", "status": "not implemented", "l4_gain_pct": None},
    "qsa_merge": {"module": "qsa_merge", "default": "off", "status": "not implemented", "l4_gain_pct": None},
}


def install_all(enabled_fn):
    for name, meta in ITEMS.items():
        if meta["status"] == "not implemented":
            continue
        if not enabled_fn(name):
            continue
        mod = __import__("vllm_sm75." + meta["module"], fromlist=["install"])
        mod.install()
