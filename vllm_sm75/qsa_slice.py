"""QSA selected-page prefill: allow larger row slices (env-gated, default off).

Upstream slices the prefill rows into fixed 1024-row calls. Each call pays the
Python/launch overhead of runner.run() (prepare check, two Triton launches,
FlashInfer wrapper call) while the GPU is idle. The FlashInfer plan needs no extra
workspace (workspace_size returns 0); the only cost of a bigger slice is the
selected-pages buffer (2.1 MB at 1024 rows -> 8.4 MB at 4096).

Set VLLM_SM75_QSA_SLICE_ROWS to 2048/4096/8192 to use bigger slices. Attention
math is unchanged: the same runner computes the same rows, only batched larger.
"""
import logging
import math
import os

import torch

_SLICE = 0


def install():
    global _SLICE
    from vllm.models.qwen4_exp.nvidia.ops import qsa_selected_prefill_fp8 as m

    _SLICE = int(os.environ.get("VLLM_SM75_QSA_SLICE_ROWS", "0"))
    if _SLICE <= 1024 or getattr(m, "_sm75_slice_patched", False):
        return

    def run(q, k, v, indices, table, seq, gate, out, *, k_scale, v_scale):
        if not isinstance(k_scale, (float, int)) or not isinstance(v_scale, (float, int)):
            raise TypeError("FP8 selected-page Prefill requires validated host scalar scales")
        if not (math.isfinite(k_scale) and math.isfinite(v_scale) and k_scale > 0 and v_scale > 0):
            raise ValueError("FP8 cache scales must be finite and positive")
        device = q.device
        stream = torch.cuda.current_stream(device)
        key = (device.index, stream.cuda_stream)
        runner = m._RUNNERS.get(key)
        if runner is None:
            runner = m.SelectedPagesRunner(device)
            m._RUNNERS[key] = runner
            logging.getLogger(__name__).info(
                "SM75 QSA FP8 selected-page Prefill active (slice_rows=%d)", _SLICE)
        rows = q.shape[0]
        for first in range(0, rows, _SLICE):
            stop = min(first + _SLICE, rows)
            runner.run(q[first:stop].contiguous(), k, v, indices[first:stop], table,
                       seq - rows + stop, gate[first:stop], out[first:stop],
                       k_scale=float(k_scale), v_scale=float(v_scale))
        m._CALLS += 1
        return out

    m.run = run
    m._sm75_slice_patched = True
