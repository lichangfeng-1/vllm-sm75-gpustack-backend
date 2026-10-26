"""Let the FlashInfer PCIe-IPC all-reduce serve PRE-FILL messages too (opt-in).

Why: vLLM sizes the PCIe-IPC workspace as ``max(cudagraph_capture_sizes) * hidden``
(see flashinfer_pcie_ipc_all_reduce.py: ``max_numel = batches[-1] * self.hidden_dim``).
With capture sizes ``[1,2,4,8,16]`` that is 16*2560 elements, so every pre-fill
all-reduce (4096*2560) fails ``workspace.supports()`` and silently falls back to
NCCL. Measured on 2026-10-26: the 32K pre-fill spends 34.8% in
``ncclDevKernel_AllReduce_Sum_f16_RING_LL``.

This module only *adds* pre-fill batch sizes to the list handed to the workspace
setup; it does not touch NCCL, the debug path, or the decode sizes. Gated by
``VLLM_SM75_AR_PREFILL`` (default 0) and sized by
``VLLM_SM75_AR_PREFILL_TOKENS`` (default 4096 = max-num-batched-tokens).

Memory: the workspace keeps two epochs ``world_size * max_numel`` elements apart,
so 4096 tokens x 2560 hidden at TP8 is ~336 MB per rank. Watch for OOM at startup
and fall back to 2048 (168 MB) if needed.

Status: experimental, NOT verified. It must pass the same L4 A/B as every other
item (>= 1% same-ruler, 5 medians) before it may join AUTO_ENABLED.
"""
import os

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        from vllm.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce as mod,
        )
    except Exception:  # noqa: BLE001 - backend not present in this build
        return

    import flashinfer.comm as fcomm

    ws_cls = getattr(fcomm, "PcieIpcAllReduceWorkspace", None)
    if ws_cls is None or getattr(ws_cls, "_sm75_ar_prefill_patched", False):
        return

    tokens = int(os.environ.get("VLLM_SM75_AR_PREFILL_TOKENS", "4096"))
    orig_init = ws_cls.__init__

    # HAZARD (learned the hard way, 2026-10-26): the autotuner benchmarks a
    # COLLECTIVE kernel. Every rank tunes its own config list, so a rank that
    # sees an extra batch size hangs while the others wait inside the collective
    # (observed: 8 workers at ~90% CPU for >13 min, engine never healthy).
    # Therefore this patch changes only the allocation (max_numel) and MUST NOT
    # touch tune_batches. Keep it that way.
    def __init__(self, *args, **kwargs):
        # setup() passes max_numel = max(capture_sizes) * hidden and the same
        # list as tune_batches. Enlarging the *allocation* alone is what we want:
        # tuning a 4096-token config stalls the autotuner for >10 minutes, while
        # launch_config() can pick a config for an untuned size on its own.
        try:
            mn = kwargs.get("max_numel")
            tb = kwargs.get("tune_batches")
            if mn and tb:
                hidden = int(mn) // int(tb[-1])
                want = tokens * hidden
                if want > int(mn):
                    kwargs["max_numel"] = want
        except Exception:  # noqa: BLE001 - never break startup on this
            pass
        return orig_init(self, *args, **kwargs)

    ws_cls.__init__ = __init__
    ws_cls._sm75_ar_prefill_patched = True
