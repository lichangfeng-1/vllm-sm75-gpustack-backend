"""M=1 fp16 GEMV dispatch for Turing (SM75) — opt-in item ``gemv_m1``.

Shape of the idea is borrowed from vLLM's own Qwen4Exp low-latency GEMM
(``vllm/models/qwen4_exp/nvidia/low_latency_gemm.py``): a per-(N, K) plan table
decides whether a hand-written M=1 kernel is used, everything else keeps
``F.linear``. That table is CuTe-DSL based and gated to SM90/SM103, so it cannot
run here; this module is the Triton equivalent for SM75.

``PLANS`` starts **empty on purpose**: a shape is added only when the kernel beat
cuBLAS on this GPU with the same ruler (see ``读数-gemv微基准-*.json``). An empty
table means this item is a no-op even when ``VLLM_SM75_GEMV_M1=1``.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# (N, K) -> (BN rows per program, BK columns per step, num_warps)
# Filled 2026-10-26 from 读数-flashnext-gemv微基准-20261026.json: only shapes that
# beat cuBLAS by >= 5% in the same-input bench, at T10 full clocks with a 64 MB L2
# flush between iterations. (512, 2560) won by 4.4% and was left out on purpose.
PLANS = {
    (2048, 2560): (1, 1024, 4),   # GDN fused QKVZ, 36x/token, cuBLAS 43.4 -> 38.9 us
    (2560, 768): (1, 2048, 4),    # GDN/QSA out_proj, 48x/token, cuBLAS 20.6 -> 18.4 us
    (160, 2560): (1, 4096, 4),    # shared-expert gate+up, 48x/token, cuBLAS 9.5 -> 8.2 us
}


@triton.jit
def _gemv_kernel(X, W, Y, N, K, stride_wn,
                 BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                    (offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        x = tl.load(X + offs_k, offs_k < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(Y + offs_n, acc.to(Y.dtype.element_ty), offs_n < N)


def _runtime_ok(x, weight):
    return (x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.dtype == torch.float16 and weight.dtype == torch.float16
            and x.is_contiguous() and weight.is_contiguous()
            and x.dim() == 2 and x.shape[0] == 1 and x.shape[1] == weight.shape[1]
            and torch.cuda.get_device_capability(x.device) == (7, 5))


@torch.library.custom_op("sm75::gemv_m1", mutates_args=())
def gemv_m1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    cfg = PLANS.get((n, k))
    if cfg is None or not _runtime_ok(x, weight):
        return F.linear(x, weight)
    bn, bk, warps = cfg
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    _gemv_kernel[(triton.cdiv(n, bn),)](
        x, weight, out, n, k, weight.stride(0), bn, bk, num_warps=warps)
    return out


@gemv_m1.register_fake
def _gemv_m1_fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def install():
    """Swap ``apply`` on the unquantized linear/embedding methods.

    Class-level so it also covers methods created before the plugin ran; the
    allowlist keeps every other shape on ``F.linear``.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if getattr(UnquantizedLinearMethod, "_sm75_gemv_patched", False):
        return

    orig_linear = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None, **kw):
        if bias is None and (weight := getattr(layer, "weight", None)) is not None \
                and weight.dim() == 2 and x.dim() == 2 \
                and (weight.shape[0], weight.shape[1]) in PLANS:
            return gemv_m1(x, weight)
        return orig_linear(self, layer, x, bias, **kw)

    orig_embed = UnquantizedEmbeddingMethod.apply

    def apply_embed(self, layer, *args, **kw):
        x = args[0] if args else kw.get("x")
        weight = getattr(layer, "weight", None)
        if (isinstance(x, torch.Tensor) and weight is not None
                and weight.dim() == 2 and x.dim() == 2
                and (weight.shape[0], weight.shape[1]) in PLANS):
            return gemv_m1(x, weight)
        return orig_embed(self, layer, *args, **kw)

    UnquantizedLinearMethod.apply = apply
    UnquantizedEmbeddingMethod.apply = apply_embed
    UnquantizedLinearMethod._sm75_gemv_patched = True
