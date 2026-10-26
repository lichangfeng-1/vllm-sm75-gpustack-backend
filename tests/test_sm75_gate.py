"""L1 gate tests for vllm_sm75 — runs on CPU, no GPU needed.

Run:  python -m pytest tests/test_sm75_gate.py -q
 or:  python tests/test_sm75_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vllm_sm75 as m  # noqa: E402


def _reset():
    m._INSTALLED = False
    m._CC_OK = None
    os.environ.pop("VLLM_SM75_OPT", None)
    for k in list(os.environ):
        if k.startswith("VLLM_SM75_") and k != "VLLM_SM75_OPT":
            os.environ.pop(k)


def _force_cc(ok):
    m._CC_OK = ok


def test_opt_zero_disables_everything():
    _reset()
    _force_cc(True)
    os.environ["VLLM_SM75_OPT"] = "0"
    os.environ["VLLM_SM75_HC_FUSE"] = "1"
    assert m.opt_mode() == "0"
    assert m.enabled("hc_fuse") is False


def test_bad_capability_disables_everything():
    _reset()
    _force_cc(False)
    os.environ["VLLM_SM75_OPT"] = "1"
    os.environ["VLLM_SM75_HC_FUSE"] = "1"
    assert m.enabled("hc_fuse") is False


def test_auto_off_when_not_in_auto_enabled():
    _reset()
    _force_cc(True)
    m.opt_mode()  # default auto
    assert m.enabled("hc_fuse") is False  # AUTO_ENABLED is empty by design


def test_explicit_item_flag_wins_over_auto():
    _reset()
    _force_cc(True)
    os.environ["VLLM_SM75_HC_FUSE"] = "1"
    assert m.enabled("hc_fuse") is True
    os.environ["VLLM_SM75_HC_FUSE"] = "0"
    assert m.enabled("hc_fuse") is False


def test_install_with_opt_zero_loads_no_submodule():
    _reset()
    _force_cc(True)
    os.environ["VLLM_SM75_OPT"] = "0"
    before = {k for k in sys.modules if k.startswith("vllm_sm75.")}
    m.install()
    after = {k for k in sys.modules if k.startswith("vllm_sm75.")}
    assert after == before, "submodules loaded while OPT=0: %s" % (after - before)


def test_install_idempotent():
    _reset()
    _force_cc(True)
    m.install()
    m.install()
    assert m._INSTALLED is True


def test_registry_never_installs_unimplemented():
    _reset()
    _force_cc(True)
    os.environ["VLLM_SM75_OPT"] = "1"
    from vllm_sm75 import registry
    called = []
    registry.install_all(lambda name: called.append(name) or True)
    assert called == [], "unimplemented items were installed: %s" % called


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print("\n%d passed, %d failed" % (len(fns) - failed, failed))
    sys.exit(1 if failed else 0)
