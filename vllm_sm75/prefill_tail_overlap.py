"""Prefill layer-tail overlap (opt-in, SM75).

The attention core (QKV, QSA/GDN, KV writes, GDN state) stays whole-batch. Only
the per-token tail after it is chunked:

    o_proj (row-parallel) -> all-reduce -> mlp_hc.combine_and_mix -> MLP

Chunk i's all-reduce runs on a side stream while chunk i+1's GEMM/MLP runs on
the compute stream; only the last chunk's MLP all-reduce stays exposed. This is
the vllm-turing "tailoverlap" idea adapted to Qwen4Exp's HC/MoE layers; unlike
full dual-batch overlap it does not split attention, KV, GDN state or PLE
n-gram context, so a single request and DP=1 are fine.

The on/off gate lives in registry.py; this module only reads the tuning knobs
below:
  VLLM_SM75_TAIL_MIN_TOKENS   default 2048; smaller batches use the plain path
  VLLM_SM75_TAIL_SPLIT        default 0.16667,0.5,0.83333 (4 chunks; 0.5 = 2)
  VLLM_SM75_TAIL_SIDE_GROUP   same (TP group) or new (dist.new_group)
  VLLM_SM75_TAIL_DEBUG        log chunk bounds once per layer

Status: implemented_pending_gate. Default off; with the item off the model is
byte-identical to baseline (no monkeypatch is installed).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_STREAMS: dict = {}


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(lo, value)


def _read_splits() -> tuple:
    raw = os.environ.get("VLLM_SM75_TAIL_SPLIT", "0.16667,0.5,0.83333")
    vals = []
    for part in raw.split(","):
        try:
            value = float(part)
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            vals.append(value)
    return tuple(sorted(set(vals))) or (0.5,)


def _compiling() -> bool:
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - old torch / CPU-only process
        return False


def _bounds(tokens: int, splits: tuple) -> list:
    """Contiguous, 16-aligned chunk bounds covering [0, tokens)."""
    if tokens <= 0:
        return []
    pts = [0]
    for frac in splits:
        point = int(frac * tokens) // 16 * 16
        if pts[-1] < point < tokens:
            pts.append(point)
    pts.append(tokens)
    return list(zip(pts, pts[1:]))


def _side_stream(device):
    import torch

    index = getattr(device, "index", None)
    key = 0 if index is None else index
    stream = _STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device, priority=-1)
        _STREAMS[key] = stream
    return stream


def _proj(linear, x):
    """The row-parallel GEMM without its all-reduce (this rank's partial sum)."""
    bias = None if (linear.tp_rank > 0 or getattr(linear, "skip_bias_add", False)) else linear.bias
    return linear.quant_method.apply(linear, x, bias)


class _Tail:
    def __init__(self, layer) -> None:
        attn = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        self.o_proj = getattr(attn, "out_proj", None) or getattr(attn, "o_proj", None)
        self.mlp_hc = layer.mlp_hyper_connection
        self.mlp = layer.mlp
        self.min_tokens = _env_int("VLLM_SM75_TAIL_MIN_TOKENS", 2048, 0)
        self.max_tokens = _env_int("VLLM_SM75_TAIL_MAX_TOKENS", 8192, 0)
        self.splits = _read_splits()
        # The side AR must not share the TP communicator with the MLP's internal
        # collectives: two streams issuing on one communicator can mismatch.
        # "same" stays only as a diagnostic.
        self.side_group_mode = os.environ.get("VLLM_SM75_TAIL_SIDE_GROUP", "new").strip().lower()
        self.debug = _env_int("VLLM_SM75_TAIL_DEBUG", 0, 0) > 0
        self._side_group = None
        self._side_group_tried = False

    def active(self, tokens: int) -> bool:
        if tokens < self.min_tokens or tokens > self.max_tokens or _compiling():
            return False
        try:
            import torch

            return not torch.cuda.is_current_stream_capturing()
        except Exception:  # noqa: BLE001 - CPU-only utility process
            return False

    def _group(self):
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        if self.side_group_mode != "new":
            return tp.device_group
        if self._side_group_tried:
            return self._side_group
        self._side_group_tried = True
        try:
            import torch.distributed as dist

            if tp.world_size != dist.get_world_size():
                logger.warning("vllm_sm75 tail: a new side group needs world==tp; using the plain path")
                return None
            self._side_group = dist.new_group(ranks=tp.ranks)
        except Exception:  # noqa: BLE001 - optional; fall back to the plain path
            logger.exception("vllm_sm75 tail: new side group failed; using the plain path")
            self._side_group = None
        return self._side_group

    def plain(self, attn, residual, injection):
        from vllm.distributed import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(_proj(self.o_proj, attn))
        hidden, block_input, inj = self.mlp_hc.combine_and_mix(residual, out, injection)
        mlp_out = self.mlp(block_input)
        return hidden, mlp_out, inj

    def chunked(self, attn, residual, injection):
        import torch
        import torch.distributed as dist

        tokens = attn.shape[0]
        chunks = _bounds(tokens, self.splits)
        if len(chunks) <= 1:
            return self.plain(attn, residual, injection)
        if self.debug:
            logger.info_once("vllm_sm75 tail: chunks %s (min %d)", chunks, self.min_tokens)

        cur = torch.cuda.current_stream(attn.device)
        side = _side_stream(attn.device)
        grp = self._group()
        if grp is None:
            return self.plain(attn, residual, injection)

        parts = []
        works = []
        dones = []
        for a, b in chunks:
            part = _proj(self.o_proj, attn[a:b])
            parts.append(part)
            ready = torch.cuda.Event()
            ready.record(cur)
            side.wait_event(ready)
            with torch.cuda.stream(side):
                works.append(dist.all_reduce(part, group=grp, async_op=True))
            part.record_stream(side)
            done = torch.cuda.Event()
            done.record(side)
            dones.append(done)

        # Preallocate the full outputs on the first chunk and copy each chunk in
        # place; a final torch.cat would hold both the chunk tensors and a second
        # full-size output. The token cap keeps this tail off the 262K row.
        hidden_out = None
        mlp_out = None
        inj_out = None
        for i, ((a, b), part, done) in enumerate(zip(chunks, parts, dones)):
            cur.wait_event(done)
            inj = None if injection is None else injection[a:b]
            hidden, block_input, inj_chunk = self.mlp_hc.combine_and_mix(
                residual[a:b], part, inj
            )
            mlp_chunk = self.mlp(block_input)
            if hidden_out is None:
                hidden_out = torch.empty(
                    (tokens, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device
                )
                mlp_out = torch.empty(
                    (tokens, mlp_chunk.shape[1]), dtype=mlp_chunk.dtype, device=mlp_chunk.device
                )
                if inj_chunk is not None:
                    inj_out = torch.empty(
                        (tokens, inj_chunk.shape[1]), dtype=inj_chunk.dtype, device=inj_chunk.device
                    )
            hidden_out[a:b].copy_(hidden)
            mlp_out[a:b].copy_(mlp_chunk)
            if inj_out is not None and inj_chunk is not None:
                inj_out[a:b].copy_(inj_chunk)
            parts[i] = None  # the AR is done and the combine consumed this chunk

        cur.wait_stream(side)
        for work in works:
            work.wait()
        return hidden_out, mlp_out, inj_out

    def run(self, attn, residual, injection):
        if not self.active(attn.shape[0]):
            return self.plain(attn, residual, injection)
        return self.chunked(attn, residual, injection)


def _wrap_o_proj(o_proj) -> None:
    """Return the pre-o_proj activation while a tail forward owns the layer."""
    if getattr(o_proj, "_sm75_tail_wrapped", False):
        return
    orig = o_proj.forward

    def forward(x):
        if getattr(o_proj, "_sm75_tail_pre", False):
            return x, None
        return orig(x)

    o_proj.forward = forward
    o_proj._sm75_tail_orig_forward = orig
    o_proj._sm75_tail_pre = False
    o_proj._sm75_tail_wrapped = True


def _register(layer) -> None:
    from vllm.model_executor.layers.linear import RowParallelLinear

    attn = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
    if attn is None:
        return
    o_proj = getattr(attn, "out_proj", None) or getattr(attn, "o_proj", None)
    if not isinstance(o_proj, RowParallelLinear):
        return
    if (
        o_proj.tp_size <= 1
        or not o_proj.reduce_results
        or o_proj.bias is not None
        or getattr(o_proj, "skip_bias_add", False)
        or not getattr(o_proj, "input_is_parallel", False)
    ):
        logger.info_once("vllm_sm75 tail: skip an o_proj that is not the simple row-parallel case")
        return
    if not hasattr(layer, "mlp_hyper_connection") or not hasattr(layer, "mlp"):
        return
    layer._sm75_tail = _Tail(layer)
    _wrap_o_proj(o_proj)
    logger.info_once("vllm_sm75 tail: registered on %s", type(layer).__name__)


def _patch_decoder_layer(cls) -> None:
    if getattr(cls, "_sm75_tail_patched", False):
        return
    cls._sm75_tail_patched = True
    orig_init = cls.__init__
    orig_forward = cls.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _register(self)
        except Exception:  # noqa: BLE001 - optional item, keep the engine alive
            logger.exception("vllm_sm75 tail: registration failed")

    def forward(
        self,
        hidden_states,
        prev_block_output,
        prev_injection,
        positions,
        *,
        input_ids,
        query_start_loc,
        ngram_context,
        **kwargs,
    ):
        tail = getattr(self, "_sm75_tail", None)
        if tail is None or kwargs or not tail.active(hidden_states.shape[0]):
            return orig_forward(
                self,
                hidden_states,
                prev_block_output,
                prev_injection,
                positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                **kwargs,
            )

        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(hidden_states, prev_block_output, prev_injection)
                prev_block_output = prev_injection = None
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        o_proj = tail.o_proj
        o_proj._sm75_tail_pre = True
        try:
            if self.layer_type == "linear_attention":
                attn_pre = self.linear_attn(hidden_states=block_input)
            elif self.layer_type == "full_attention":
                attn_pre = self.self_attn(hidden_states=block_input, positions=positions)
            else:
                raise ValueError("Invalid layer_type")
        finally:
            o_proj._sm75_tail_pre = False

        return tail.run(attn_pre, hidden_states, injection)

    cls.__init__ = __init__
    cls.forward = forward


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer
    except Exception:  # noqa: BLE001 - this item only targets Qwen4Exp
        logger.exception("vllm_sm75 tail: Qwen4Exp decoder layer not available")
        return
    _patch_decoder_layer(Qwen4ExpDecoderLayer)
    _INSTALLED = True
