"""Probe only: simulate an 8-bit row-parallel transfer to measure its precision cost.

NOT an optimization. It leaves the fp16 NCCL all-reduce untouched and quantizes the
row-parallel output to fp8-e4m3 (per token) before / after it, so an L3 teacher-forced
logprob run can answer one question: how much accuracy would halving the AR bytes cost?

Approximation: the real scheme would quantize each rank's operand once and sum 8 of
them; this probe quantizes the already-reduced result once, which has the same
per-element RMS error (one quantization) but a slightly different error correlation.
Use it as a go/no-go gate, not as a bit-exact simulation.

Env: VLLM_SM75_AR_Q8_PROBE=1 (registry gate; default off). Prefill only (tokens>=1024).
"""
import torch


def _q8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float16 or t.dim() != 2:
        return t
    amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = amax / 448.0
    q = (t.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(t.dtype)


def _capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def install():
    from vllm.model_executor.layers.linear import RowParallelLinear

    if getattr(RowParallelLinear, "_sm75_q8_probe_patched", False):
        return
    orig = RowParallelLinear.forward

    def forward(self, input_):
        out = orig(self, input_)
        if input_.shape[0] < 1024 or _capturing():
            return out
        if isinstance(out, tuple):
            return (_q8(out[0]),) + out[1:]
        return _q8(out)

    RowParallelLinear.forward = forward
    RowParallelLinear._sm75_q8_probe_patched = True
