"""Measurement only: log slow QSA selected-page calls (env-gated, no behavior change).

Wraps the per-layer module-level run plus SelectedPagesRunner._prepare/run with
perf_counter timers and logs calls slower than VLLM_SM75_QSA_TIMING_MS
(default 2 ms). Used to attribute the GPU-idle gaps seen before _selected_subpages.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)


def install():
    from vllm.models.qwen4_exp.nvidia.ops import qsa_selected_prefill_fp8 as m

    if getattr(m, "_sm75_timing_patched", False):
        return
    threshold = float(os.environ.get("VLLM_SM75_QSA_TIMING_MS", "2")) / 1000.0
    cls = m.SelectedPagesRunner
    orig_prep = cls._prepare
    orig_runner_run = cls.run
    orig_layer_run = m.run

    def _prepare(self, rows, seq):
        t = time.perf_counter()
        try:
            return orig_prep(self, rows, seq)
        finally:
            dt = time.perf_counter() - t
            if dt >= threshold:
                logger.info("SM75 QSA timing: _prepare rows=%d seq=%d %.2f ms", rows, seq, dt * 1e3)

    def runner_run(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_runner_run(self, *a, **kw)
        finally:
            dt = time.perf_counter() - t
            if dt >= threshold:
                logger.info("SM75 QSA timing: runner.run %.2f ms", dt * 1e3)

    def layer_run(q, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_layer_run(q, *a, **kw)
        finally:
            dt = time.perf_counter() - t
            if dt >= threshold:
                logger.info("SM75 QSA timing: layer.run rows=%d %.2f ms", q.shape[0], dt * 1e3)

    cls._prepare = _prepare
    cls.run = runner_run
    m.run = layer_run
    m._sm75_timing_patched = True
