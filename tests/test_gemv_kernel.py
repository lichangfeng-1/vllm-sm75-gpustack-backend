"""L2 numeric test for the SM75 M=1 GEMV kernel.

Needs torch + CUDA (CC 7.5). Skips cleanly everywhere else, so the CPU-only gate
tests stay runnable on a workstation.

Run inside the image:
  python3 tests/test_gemv_kernel.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _need_gpu():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_capability() != (7, 5):
        return None
    return torch


def test_kernel_matches_f_linear():
    torch = _need_gpu()
    if torch is None:
        print("SKIP test_kernel_matches_f_linear (no CC 7.5 CUDA)")
        return
    from vllm_sm75.gemv_m1 import PLANS, gemv_m1

    torch.manual_seed(0)
    for (n, k) in list(PLANS) or [(2560, 2560), (2048, 2560), (12, 2560), (336, 10240)]:
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        x = torch.randn(1, k, device="cuda", dtype=torch.float16)
        ref = torch.nn.functional.linear(x, w)
        out = gemv_m1(x, w)          # falls back to F.linear when (n,k) not in PLANS
        err = (out - ref).abs().max().item()
        scale = ref.abs().max().item() or 1.0
        rel = err / scale
        print("N=%d K=%d max_abs_err=%.5f rel=%.2e %s"
              % (n, k, err, rel, "in PLANS" if (n, k) in PLANS else "fallback"))
        assert rel < 5e-3, "N=%d K=%d rel err %.3e too large" % (n, k, rel)


def test_fallback_on_non_fp16_and_batched():
    torch = _need_gpu()
    if torch is None:
        print("SKIP test_fallback_on_non_fp16_and_batched")
        return
    from vllm_sm75.gemv_m1 import PLANS, gemv_m1
    if not PLANS:
        print("SKIP test_fallback_on_non_fp16_and_batched (empty PLANS)")
        return
    (n, k) = next(iter(PLANS))
    w32 = torch.randn(n, k, device="cuda", dtype=torch.float32)
    x32 = torch.randn(1, k, device="cuda", dtype=torch.float32)
    assert torch.allclose(gemv_m1(x32, w32), torch.nn.functional.linear(x32, w32))
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    x2 = torch.randn(4, k, device="cuda", dtype=torch.float16)
    assert torch.allclose(gemv_m1(x2, w), torch.nn.functional.linear(x2, w), atol=2e-2)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as exc:
                failed += 1
                print("FAIL", name, exc)
    sys.exit(1 if failed else 0)
