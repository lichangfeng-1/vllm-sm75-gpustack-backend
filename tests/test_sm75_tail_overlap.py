"""CPU-only tests for the layer-tail overlap helper (no torch / no GPU)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_sm75 import prefill_tail_overlap as t  # noqa: E402


def test_bounds_cover_all_tokens_and_align():
    chunks = t._bounds(8192, (0.16667, 0.5, 0.83333))
    assert chunks[0][0] == 0 and chunks[-1][1] == 8192
    assert [b for _, b in chunks[:-1]] == [a for a, _ in chunks[1:]]
    for _, end in chunks[:-1]:
        assert end % 16 == 0


def test_bounds_dedup_and_tiny():
    assert t._bounds(100, (0.5, 0.5)) == [(0, 48), (48, 100)]
    assert t._bounds(8, (0.5,)) == [(0, 8)]
    assert t._bounds(0, (0.5,)) == []


def test_splits_fallback():
    old = os.environ.get("VLLM_SM75_TAIL_SPLIT")
    try:
        os.environ["VLLM_SM75_TAIL_SPLIT"] = "nope,2.0,-1"
        assert t._read_splits() == (0.5,)
        os.environ["VLLM_SM75_TAIL_SPLIT"] = "0.25,0.75"
        assert t._read_splits() == (0.25, 0.75)
    finally:
        if old is None:
            os.environ.pop("VLLM_SM75_TAIL_SPLIT", None)
        else:
            os.environ["VLLM_SM75_TAIL_SPLIT"] = old


def test_o_proj_wrapper_returns_pre_proj_only_when_armed():
    class Fake:
        def __init__(self):
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return ("orig", x)

    fake = Fake()
    t._wrap_o_proj(fake)
    assert fake.forward("a") == ("orig", "a")
    fake._sm75_tail_pre = True
    assert fake.forward("b") == ("b", None)
    fake._sm75_tail_pre = False
    assert fake.forward("c") == ("orig", "c")
    assert fake.calls == 2
