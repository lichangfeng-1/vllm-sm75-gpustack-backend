"""P1: overlap the row-parallel all-reduce with the next chunk's GEMM (opt-in).

Why: 32K prefill spends 34.8% of its time in all-reduce (3.42 s) while the GPU draws
only ~100 W of a 150 W cap — it is waiting, not computing. The token dimension is
independent, so splitting the row-parallel output into chunks lets NCCL run on its
own stream while the next chunk's GEMM runs on the compute stream.

How: wrap RowParallelLinear.forward. When it would call a single synchronous
all-reduce, instead split the (already TP-local) output along tokens, launch an
async all-reduce per chunk, and only then wait. The producer GEMM of chunk i+1 is
issued between the launch and the wait, so it overlaps with chunk i's collective.

Guards:
  * only the plain path (reduce_results, no bias, not skip_bias_add); return_bias is
    handled (it is True by default, which silently excluded GDN out_proj in the first cut);
  * only when tokens >= VLLM_SM75_AR_OVERLAP_MIN_TOKENS (decode runs 1-16 tokens);
  * skipped while a CUDA graph is being captured;
  * the reduction is element-wise, so chunking cannot change the result.

Env: VLLM_SM75_AR_OVERLAP (auto|0|1), VLLM_SM75_AR_OVERLAP_SPLIT (default 2),
     VLLM_SM75_AR_OVERLAP_MIN_TOKENS (default 1024).
Status: rejected 2026-10-26. Same-image A/B (split=2, min_tokens=1024) gave
2306.6/10081.5 ms with the flag on vs 2266.7/10021.7 ms with it off at ~7.9K/~31.6K
prefill (5-run medians) — no gain. Kept default-off for reference; only 48 of the
~98 AR launches per prefill step are reachable from this insertion point (MoE's 48
and the embedding/PLE reductions are elsewhere).
"""
import os

_INSTALLED = False


def _capturing():
    try:
        import torch

        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.linear import RowParallelLinear

    if getattr(RowParallelLinear, "_sm75_ar_overlap_patched", False):
        return

    split = max(2, int(os.environ.get("VLLM_SM75_AR_OVERLAP_SPLIT", "2")))
    min_tokens = int(os.environ.get("VLLM_SM75_AR_OVERLAP_MIN_TOKENS", "1024"))
    orig = RowParallelLinear.forward

    def forward(self, input_):
        if (
            not self.reduce_results
            or self.tp_size <= 1
            or self.skip_bias_add
            or self.bias is not None
            or not self.input_is_parallel
        ):
            return orig(self, input_)

        tokens = input_.shape[0]
        if tokens < min_tokens or _capturing():
            return orig(self, input_)

        # Chunk the INPUT, not the output: the point is to issue the collective for
        # chunk i and then let the GEMM of chunk i+1 run on the compute stream.
        step = (tokens + split - 1) // split
        outs = []
        works = []
        for chunk in input_.split(step, dim=0):
            o = self.quant_method.apply(self, chunk, None)
            if not o.is_contiguous():
                o = o.contiguous()
            work = dist.all_reduce(o, group=get_tp_group().device_group,
                                   async_op=True)
            outs.append(o)
            works.append(work)
        for work in works:
            work.wait()
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        if not self.return_bias:
            return out
        return out, (self.bias if self.skip_bias_add else None)

    RowParallelLinear.forward = forward
    RowParallelLinear._sm75_ar_overlap_patched = True
