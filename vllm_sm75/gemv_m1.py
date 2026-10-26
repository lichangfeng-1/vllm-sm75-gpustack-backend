"""M=1 fp16 GEMV dispatch for Turing (SM75) — opt-in item ``gemv_m1``.

Shape of the idea is borrowed from vLLM's own Qwen4Exp low-latency GEMM
(``vllm/models/qwen4_exp/nvidia/low_latency_gemm.py``): a per-(N, K) plan table
decides whether a hand-written M=1 kernel is used, everything else keeps
``F.linear``. That table is CuTe-DSL based and gated to SM90/SM103, so it cannot
run here; this module is the Triton equivalent for SM75.

``PLANS`` starts **empty on purpose**: a shape is added only when the kernel beat
cuBLAS on this GPU with the same ruler (see ``读数-gemv微基准-*.json``). An empty
table means this item is a no-op even when ``VLLM_SM75_GEMV_M1=1``.

Defences: install() compiles and self-checks every plan before vLLM starts capturing
CUDA graphs (capture runs with zero warmups, so a JIT compile inside it is unsafe), and
falls back to plain ``F.linear`` if the check fails; ``VLLM_BATCH_INVARIANT`` keeps the
upstream deterministic kernel.
"""
import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

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


_BATCH_INVARIANT = False


def _prewarm():
    """Compile and self-check every plan before vLLM captures CUDA graphs.

    vLLM captures decode graphs with ``cudagraph_num_of_warmups=0``, so the first
    M=1 call of a process can land inside stream capture; a Triton JIT compile at
    that point is not capture-safe. A real launch here also proves the kernel
    agrees with cuBLAS before it is allowed anywhere near the engine.
    """
    if not torch.cuda.is_available():
        return False
    for n, k in PLANS:
        x = torch.randn(1, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        got = gemv_m1(x, w)
        ref = F.linear(x, w)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()
    return True


def install():
    """Swap ``apply`` on the unquantized linear/embedding methods.

    Class-level so it also covers methods created before the plugin ran; the
    allowlist keeps every other shape on ``F.linear``.
    """
    global _BATCH_INVARIANT
    from vllm import envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if getattr(UnquantizedLinearMethod, "_sm75_gemv_patched", False):
        return

    try:
        if not _prewarm():
            logger.warning("vllm_sm75: gemv_m1 skipped (no usable CUDA device)")
            return
    except Exception:  # noqa: BLE001 - disable the item, keep the engine alive
        logger.exception("vllm_sm75: gemv_m1 self-check failed; staying on F.linear")
        return

    _BATCH_INVARIANT = bool(envs.VLLM_BATCH_INVARIANT)
    orig_linear = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None, **kw):
        if not _BATCH_INVARIANT and bias is None \
                and (weight := getattr(layer, "weight", None)) is not None \
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
