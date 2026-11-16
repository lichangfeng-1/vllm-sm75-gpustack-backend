# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant 4-bit KV-cache attention backend.

UltraQuant is a 4-bit KV-cache format (FP4 E2M1 codes + UE8M0 power-of-two
per-group-32 scales) with Hadamard-rotated keys. It reuses the TurboQuant
attention scaffolding (forward orchestration, metadata, cache-layout addressing,
CUDA-graph pre-warm) by subclassing, and overrides only the format-specific
seams:

  * store         -> ultraquant_store (rotate K, pack FP4 + UE8M0 scales)
  * decode        -> FlyDSL D=256 scaled-MFMA kernel, Triton unified fallback
  * continuation  -> small chunk: Triton unified; large chunk: dequant + FA

``UltraQuantAttentionBackend`` is a standalone backend with its own identity
(``ULTRAQUANT``) and dtype (``ultraquant_4bit``); the TurboQuant backend/impl
carry no UltraQuant knowledge.
"""

import contextlib
import math
from dataclasses import replace
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.turboquant_attn import (
    _CONTINUATION_DECODE_THRESHOLD,
    _fa2_supported,
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
    _build_hadamard,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# CK FlashAttention on ROCm has no compiled kernel above head_dim 256 and no
# sinks argument; those cases route through the Triton unified kernel instead.
_CK_MAX_HEAD_DIM = 256


class UltraQuantAttentionBackend(TurboQuantAttentionBackend):
    """Standalone backend for the ``ultraquant_4bit`` KV-cache dtype.

    Subclasses :class:`TurboQuantAttentionBackend` for the shared metadata and
    cache-layout plumbing, and overrides identity, spec, impl, block size, and
    capabilities so no UltraQuant knowledge lives in the TurboQuant backend.
    """

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["ultraquant_4bit"]
    # sm75 无 bf16; 重写后的 Triton decode QK 用 fp16 (K dequant→fp16 + Q fp16 +
    # tl.dot), 故放行 fp16。上游 FlyDSL D=256 decode 只认 bf16, 但 FlyDSL 仅 gfx950
    # (sm75 自动回退 Triton, on_gfx950() False), 不冲突。Selection 读 backend class。
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """UltraQuant packs K+V into one 4-bit FP4/UE8M0 slot per head."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_ultraquant:
            return spec
        return replace(spec, state_content_bytes=slot_size(spec.head_size))

    @staticmethod
    def get_name() -> str:
        return "ULTRAQUANT"

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # FlyDSL D=256 decode geometry requires a 64-token KV block; pin it so
        # the KV-cache manager pages stay dense and splittable into kernel blocks.
        return 64

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # UltraQuant 内核 (triton_store/dequant/unified) 按 block_table 逐 token 遍历,
        # 用 kv_cache.shape[1]/stride 自适应 block 粒度, 不依赖固定 kernel block 大小
        # (FA2 时代 [16,32,64,128] 切分约束对我们是人为的)。
        # 混合模型 (full-attn + GDN/mamba) 的 unify_kv_cache_spec_page_size
        # (kv_cache_utils.py:1338) 会把 FP4 组 spec.block_size 撑到 16*ratio
        # (ratio = GDN_state_page // FP4_page, 本模型=97 → 1552), 若仍走 [16,32,64,128]
        # 则 select_common_block_size 取 16, create_kv_cache_views 里 ratio=1552//16>1
        # 触发 dense-page 守卫 (block_stride 按更大的 GDN page 定 != FP4 page) 崩
        # "block stride 794624 != page 422144 ... split into 97 kernel blocks"。
        # 报 MultipleOf(1) → select_common_block_size(1552) 直接返回 1552 (不切分),
        # ratio=1, 守卫 (ratio>1 才触发) 跳过, 绕开矛盾。
        return [MultipleOf(1)]

    @staticmethod
    def get_impl_cls() -> type["UltraQuantAttentionImpl"]:
        return UltraQuantAttentionImpl

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "ultraquant_4bit"

    @classmethod
    def supports_sink(cls) -> bool:
        # Decode and continuation prefill route sinks to the Triton unified
        # kernel, but first-chunk prefill goes through flash_attn_varlen, which
        # takes no sinks argument. Until that path is covered, reject sink
        # models at selection rather than returning silently wrong output.
        return False


class UltraQuantAttentionImpl(TurboQuantAttentionImpl):
    """UltraQuant 4-bit KV-cache attention (FP4 E2M1 + UE8M0, rotated K).

    Overrides the init, on-device setup, store, decode, and prefill/continuation
    seams of :class:`TurboQuantAttentionImpl`; the shared ``forward`` /
    ``do_kv_cache_update`` orchestration and metadata handling are inherited.
    """

    # sm75 无 bf16; Triton decode QK 重写为 fp16 (K dequant→fp16 + Q fp16 +
    # tl.dot), 故放行 fp16。FlyDSL 仅 gfx950, sm75 自动回退 Triton, 不冲突。
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # sm75 门控: get_flash_attn_version() 在 cc<8.0 上会 logger.error +
        # assert 崩 ("FA2 is only supported on devices with compute capability
        # >= 8")。sm75 上 FA 路径全部被 _fa2_supported() 关掉 (prefill first-chunk
        # 走 SDPA 兜底, continuation 大 chunk 同理), 置 None → _flash_attn_varlen
        # 永不被调用。同 turboquant 同款坑。
        self.fa_version = (
            get_flash_attn_version(head_size=head_size) if _fa2_supported() else None
        )

        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self.sliding_window = sliding_window
        self.sinks = kwargs.get("sinks")
        self._max_model_len = vllm_config.model_config.max_model_len

        self._use_ultraquant_flydsl = is_ultraquant_flydsl_available()
        # FlyDSL decode eligibility is static per layer config; compute once.
        self._flydsl_decode_eligible = ultraquant_flydsl_decode_eligible(
            head_size=self.head_size,
            num_kv_groups=self.num_kv_groups,
            has_sinks=self.sinks is not None,
            sliding_window=self.sliding_window,
            flydsl_loaded=self._use_ultraquant_flydsl,
        )

    def _ensure_on_device(self, layer, device):
        """Build the shared Hadamard rotation and pre-warm FlyDSL buffers.

        UltraQuant needs no Lloyd-Max centroids, but the inherited forward reads
        ``layer._tq_centroids``, so an empty tensor is kept on the layer.
        """
        if self._use_ultraquant_flydsl:
            # CUDA-graph capture safety for the FlyDSL decode path on ROCm:
            # pre-allocate index buffers and grow the workspace to its max size
            # before capture so replay never lands on stale addresses.
            _max_len = self._max_model_len
            _already_ok = (
                hasattr(self, "_arange_cache")
                and self._arange_cache.device.type == str(device).split(":")[0]
                and self._arange_cache.shape[0] >= _max_len + 2
            )
            if not _already_ok:
                self._arange_cache = torch.arange(
                    0, _max_len + 2, device=device, dtype=torch.int32
                )
            if not hasattr(self, "_cu_2") or self._cu_2.device != torch.device(device):
                self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)
            if (
                is_workspace_manager_initialized()
                and not current_workspace_manager().is_locked()
            ):
                B_max = self._max_capture_batch_size()
                D = self.head_size
                Hq = self.num_heads
                S = self.max_num_kv_splits
                _pre_warm_bytes = (
                    B_max * Hq * (S * (D + 1) + D) * 4  # fp32 mid_o + fp32 lse
                    + B_max * Hq * D * 2  # query-dtype output (bf16 = 2 B)
                    + 512  # alignment padding
                )
                with contextlib.suppress(AssertionError):
                    current_workspace_manager().get_simultaneous(
                        ((_pre_warm_bytes,), torch.uint8)
                    )

        if not hasattr(layer, "_tq_cached"):
            D = self.head_size
            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            layer._tq_Pi_half = H.to(torch.float16)
            # Inherited TurboQuantAttentionImpl.forward reads _tq_centroids
            # unconditionally; UltraQuant has no centroid table, so an empty
            # tensor satisfies the read.
            layer._tq_centroids = torch.empty(0, device=device, dtype=torch.float32)
            layer._tq_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Rotate K by the Hadamard and pack K+V into FP4/UE8M0 slots."""
        from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

        ultraquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            PiT=layer._tq_PiT,
        )

    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch from the WorkspaceManager (UltraQuant
        # only needs the output buffer; the mid_o/lse slots keep the triple
        # allocation identical to the base decode for buffer reuse).
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        output_buf = None
        if is_workspace_manager_initialized():
            _mid_o, output_buf, _lse = current_workspace_manager().get_simultaneous(
                ((B, Hq, S, D + 1), torch.float32),
                ((B, Hq, D), query.dtype),
                ((B, Hq), torch.float32),
            )

        if self._flydsl_decode_eligible:
            return flydsl_ultraquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                PiT=PiT,
                max_seq_len=attn_metadata.max_seq_len,
                output_buf=output_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
            )

        logger.info_once(
            "UltraQuant FlyDSL is ineligible for head_size=%s, GQA=%s, "
            "sinks=%s, sliding_window=%s; using Triton fallback.",
            self.head_size,
            self.num_kv_groups,
            self.sinks is not None,
            self.sliding_window,
        )
        return ultraquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            scale=self.scale,
            PiT=PiT,
            output=output_buf,
            max_query_len=1,
            max_seq_len=attn_metadata.max_seq_len,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Firefly prefill: 整批 first-chunk 的单个 FI wrapper (build 阶段已 plan,
        # qo_indptr=kv_indptr, causal)。命中即 run 全批, 绕开 FA2 (sm75 崩) /
        # SDPA (O(N²) 长 ctx OOM)。UltraQuant 继承 TurboQuant 的 builder, wrapper
        # 按 model_config.dtype (sm75=fp16) plan, 与这里 raw K/V (fp16) 一致。
        if attn_metadata.flashinfer_first_chunk_wrapper is not None:
            return attn_metadata.flashinfer_first_chunk_wrapper.run(query, key, value)

        # Fast path: flash_attn for first-chunk prefills (all K/V in batch).
        # CK cannot serve head dims above 256, so those fall through to the
        # per-request unified-attention path below.
        if (
            _fa2_supported()
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and D <= _CK_MAX_HEAD_DIM
        ):
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if D > _CK_MAX_HEAD_DIM:
                    # Head dims CK cannot handle (Gemma 4 global layers,
                    # D=512). This kernel is generic over head size.
                    from vllm.v1.attention.ops.triton_unified_attention import (
                        unified_attention,
                    )

                    out = torch.empty_like(q_seq)
                    cu_single = torch.tensor(
                        [0, q_len], dtype=torch.int32, device=query.device
                    )
                    seq_lens_single = torch.tensor(
                        [q_len], dtype=torch.int32, device=query.device
                    )
                    block_table_single = torch.zeros(
                        (1, 1), dtype=torch.int32, device=query.device
                    )
                    unified_attention(
                        q=q_seq,
                        k=k_seq.unsqueeze(0),
                        v=v_seq.unsqueeze(0),
                        out=out,
                        cu_seqlens_q=cu_single,
                        max_seqlen_q=q_len,
                        seqused_k=seq_lens_single,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(-1, -1),
                        block_table=block_table_single,
                        softcap=0.0,
                        q_descale=None,
                        k_descale=None,
                        v_descale=None,
                        sinks=None,
                    )
                else:
                    # Firefly prefill: per-request first-chunk wrapper (build 阶段
                    # plan, causal) 优先, 绕开 FA2 (sm75 崩) / SDPA (O(N²) OOM)。
                    _fi_first = (
                        attn_metadata.flashinfer_first_chunk_wrappers.get(i)
                        if attn_metadata.flashinfer_first_chunk_wrappers
                        else None
                    )
                    if _fi_first is not None:
                        out = _fi_first.run(q_seq, k_seq, v_seq)
                    elif _fa2_supported():
                        self._cu_2[1:2] = q_len
                        cu = self._cu_2
                        out = self._flash_attn_varlen(
                            q=q_seq,
                            k=k_seq,
                            v=v_seq,
                            cu_seqlens_q=cu,
                            cu_seqlens_k=cu,
                            max_seqlen_q=q_len,
                            max_seqlen_k=q_len,
                        )
                    else:
                        q_t = q_seq.transpose(0, 1).contiguous()
                        k_t = k_seq.transpose(0, 1).contiguous()
                        v_t = v_seq.transpose(0, 1).contiguous()
                        out = F.scaled_dot_product_attention(
                            q_t,
                            k_t,
                            v_t,
                            is_causal=True,
                            scale=self.scale,
                            enable_gqa=use_gqa,
                        ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: prefix already stored (Hadamard-rotated
                # FP4) by do_kv_cache_update. Small chunks reuse the Triton
                # unified decode-shaped kernel; large chunks dequant the prefix
                # once and run a dense flash-attention prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = ultraquant_unified_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        query_start_loc=_arange_cache[: q_len + 1],
                        scale=self.scale,
                        PiT=PiT,
                        max_query_len=1,
                        max_seq_len=seq_len,
                        sinks=self.sinks,
                        sliding_window=self.sliding_window,
                    )
                else:
                    # Firefly prefill: per-request continuation wrapper (build
                    # 阶段 plan, kv_indptr=[0,seq_len] 右对齐 causal), 在 dequant+
                    # 旋转后直接 run, 绕开 FA2 (sm75 崩) / SDPA (O(N²) OOM)。
                    _fi_cont = (
                        attn_metadata.flashinfer_continuation_wrappers.get(i)
                        if attn_metadata.flashinfer_continuation_wrappers
                        else None
                    )
                    out = self._ultraquant_continuation_prefill(
                        layer=layer,
                        query=q_seq,
                        key_chunk=k_seq,
                        val_chunk=v_seq,
                        flashinfer_wrapper=_fi_cont,
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i : i + 1],
                        cached_len=cached_len,
                        seq_len=seq_len,
                        PiT=PiT,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _ultraquant_continuation_prefill(
        self,
        *,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D) unrotated
        key_chunk: torch.Tensor,  # (q_len, Hk, D) unrotated
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        flashinfer_wrapper: Any | None = None,
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        PiT: torch.Tensor,  # (D, D) fp32 Hadamard
    ) -> torch.Tensor:
        """Large continuation chunk: dequant the cached prefix, then flash_attn.

        The cache stores K already Hadamard-rotated, so only the current chunk's
        K and the queries need rotating. Hadamard is symmetric orthonormal, so
        rotating both sides leaves the attention scores unchanged.
        """
        from vllm.v1.attention.ops.ultraquant.triton_dequant import (
            ultraquant_full_dequant_kv,
        )

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        qdtype = query.dtype
        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / block_size) * block_size

        # 四个 scratch buffer 全部走 workspace manager (跨 16 层共享一个分配,
        # 请求间释放), 不按 layer 属性常驻。
        #
        # 早期把 k_full/v_full 缓存在 layer 属性上并按 seq_len 惰性增长: 服务跑
        # 一段时间见过长序列后, 16 层各自驻留 seq_len×Hk×D×2B 永不释放, 在 2080Ti
        # 的紧显存预算里累积到 ~2GiB, 最终连 32MiB 的 embedding 分配都 OOM。
        # 改走 workspace 后峰值=单次最长请求的 4 块 scratch (dequant + concat),
        # 不再随历史最长序列累积。
        #
        # (cap 不能再用 block_table.shape[1]*block_size: FP4 混合模型
        #  unify_kv_cache_spec_page_size 把 block_size 撑到 1552, 乘积远超 seq_len。)
        full_shape = (seq_len, Hk, D)
        deq_shape = (1, Hk, alloc_len, D)
        k_buf, v_buf, k_full, v_full = current_workspace_manager().get_simultaneous(
            (deq_shape, qdtype),
            (deq_shape, qdtype),
            (full_shape, qdtype),
            (full_shape, qdtype),
        )
        ultraquant_full_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_buf,
            v_out=v_buf,
            alloc_len=alloc_len,
        )

        k_full[:cached_len] = k_buf[0, :, :cached_len, :].transpose(0, 1)
        v_full[:cached_len] = v_buf[0, :, :cached_len, :].transpose(0, 1)
        k_full[cached_len:] = (key_chunk.to(torch.float32) @ PiT).to(qdtype)
        v_full[cached_len:] = val_chunk
        q_rot = (query.to(torch.float32) @ PiT).to(qdtype)

        # Firefly prefill: FI wrapper (build 阶段 plan, kv_indptr=[0,seq_len] 右对齐
        # causal) 直接 run 旋转空间里的 q_rot/k_full/v_full, 绕开 FA2 (sm75 崩) /
        # SDPA (O(N²) OOM)。K 在 cache 里已预旋转, 故须在旋转后的 q_rot/k_full 上跑
        # (同 FA2/SDPA 的输入), 不能用原始 query (与 turboquant 不同, 那里 K 未旋转)。
        if flashinfer_wrapper is not None:
            return flashinfer_wrapper.run(q_rot, k_full, v_full)

        # Unequal q/k lengths with causal=True gives lower-right alignment:
        # query i (absolute position cached_len + i) sees keys 0..cached_len+i.
        #
        # FA2 on ROCm has no sinks parameter, and CK cannot serve head dims
        # above 256, so those cases route through the Triton unified kernel.
        # Q/K are already in Hadamard-rotated space (q_rot, k_full) so the dot
        # products match the original-space attention scores.
        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )

            out = torch.empty_like(q_rot)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=q_rot,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out.to(qdtype)
        if _fa2_supported():
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            return self._flash_attn_varlen(
                q=q_rot,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        if Hq > Hk:
            k_t = k_t.expand(1, Hq, -1, -1)
            v_t = v_t.expand(1, Hq, -1, -1)
        return (
            F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale
            )
            .squeeze(0)
            .transpose(0, 1)
            .to(qdtype)
        )
