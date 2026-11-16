# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: MarlinLinearKernel + firefly(int4→int8 prefill)混合。

以 v0.30.0 原版为基线, 仅叠加 firefly 分支(env 门控, off=上游行为):
  - process_weights_after_loading: repack 前快照干净 int4 布局 [N, K/8] + scale
  - apply_weights: 大 M(prefill)走 int8 cutlass(IMMA), 小 M(decode)走上游 int4 Marlin

见 vllm/model_executor/layers/quantization/utils/firefly.py。
"""

import torch

from vllm import _custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    marlin_act_int8_process_scales,
    marlin_make_empty,
    marlin_make_workspace_new,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.firefly import (
    dequant_marlin_to_int8,
    dequant_marlin_to_int8_cached,
    firefly_active_int4,
    int8_prefill_linear,
)
from vllm.model_executor.layers.quantization.utils.firefly import _unpack_awq_zp
from vllm.model_executor.layers.quantization.utils.firefly_idecode import (
    idecode_gemv,
    idecode_ready,
    natural_c_n,
    natural_pack_from_gptq,
    warm_idecode,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)


# ---- firefly 混合 op(dynamo/inductor 兼容) ----
# firefly 分支判据(`m > min_m`, m 是动态 shape) 若直接留在编译图里, inductor 会把
# firefly 路径(含 custom op)一起编进 decode(M=1) 的图, 实测 decode 从 27 tok/s 掉到
# 4.8 tok/s(baseline 无 firefly 时该分支被静态消除)。
# 图断点方案走不通: torch._dynamo.disable + fullgraph=False 与 VllmBackend
# ("can only be called once", 只收单图)和 AOT compile("不支持图断点")冲突。
# 故把"门控 + firefly/Marlin 选择 + 执行"整体包成一个 torch.library custom op:
# 图里只剩一个不透明 leaf, inductor 看不到分支, decode 图干净(=baseline 性能)。
# op 内部按真实 M 求值: prefill(m>min_m)现反量化 int8 走 IMMA, decode 走上游 Marlin。
@torch.library.custom_op("firefly::hybrid_linear", mutates_args=())
def _ff_hybrid_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s_marlin: torch.Tensor,
    w_zp_marlin: torch.Tensor,
    workspace: torch.Tensor,
    input_global_scale: torch.Tensor,
    bias: torch.Tensor,
    w_s_clean: torch.Tensor,
    w_zp_clean: torch.Tensor,
    c_n: torch.Tensor,
    size_k: int,
    size_n: int,
    padded_k: int,
    padded_n: int,
    gs: int,
    min_m: int,
    has_zp: bool,
    use_recip: bool,
    w_natural: bool,
    w_zp_exp: torch.Tensor,
    bias_clean: torch.Tensor,
    idec_max_m: int,
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if m > min_m:
        # prefill: transient 反量化 int4→int8 + cutlass_scaled_mm(SM75 即 IMMA)
        # natural(VLLM_FIREFLY_IDEC)时 w_q 是自然布局 uint8、zp 用已展开的 fp16 版,
        # 走 natural op; 否则原 marlin 反 repack 路。
        w_int8 = dequant_marlin_to_int8_cached(
            w_q,
            w_s_clean,
            c_n,
            gs,
            size_k,
            size_n,
            padded_k,
            padded_n,
            w_zp=(w_zp_clean if w_zp_clean.numel() > 0 else None),
            use_recip=use_recip,
            natural=w_natural,
            wzp_fp16=(w_zp_exp if w_zp_exp.numel() > 0 else None),
        )
        return int8_prefill_linear(x, w_int8, c_n)
    # decode: 自然布局 + M<=idec_max_m → 自研 GEMV(marlin 布局已退场, 无回退副本);
    # 否则上游 int4 Marlin(仅 marlin 布局模式会走到)。
    if w_natural:
        if m <= idec_max_m:
            return idecode_gemv(
                x.reshape(m, -1),
                w_q,
                w_s_clean,
                gs,
                size_k,
                size_n,
                wzp=(w_zp_exp if w_zp_exp.numel() > 0 else None),
                bias=(bias_clean if bias_clean.numel() > 0 else None),
            ).reshape(*x.shape[:-1], size_n)
        logger.warning_once(
            "firefly idecode: M=%d > VLLM_FIREFLY_IDEC_MAX_M=%d 且落在 marlin 区间 "
            "(自然布局无 marlin 权重可回退), 请调大 IDEC_MAX_M 或关掉 IDEC",
            m,
            idec_max_m,
        )
    wtype = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    return apply_gptq_marlin_linear(
        input=x,
        weight=w_q,
        weight_scale=w_s_marlin,
        weight_zp=w_zp_marlin,
        workspace=workspace,
        wtype=wtype,
        input_size_per_partition=size_k,
        output_size_per_partition=size_n,
        input_global_scale=(
            input_global_scale if input_global_scale.numel() > 0 else None
        ),
        bias=bias if bias.numel() > 0 else None,
        input_dtype=x.dtype,
    )


@_ff_hybrid_linear.register_fake
def _(
    x,
    _w_q,
    _w_s_marlin,
    _w_zp_marlin,
    _workspace,
    _input_global_scale,
    _bias,
    _w_s_clean,
    _w_zp_clean,
    _c_n,
    _size_k,
    size_n,
    _padded_k,
    _padded_n,
    _gs,
    _min_m,
    _has_zp,
    _use_recip,
    _w_natural,
    _w_zp_exp,
    _bias_clean,
    _idec_max_m,
):
    return torch.empty(x.shape[:-1] + (size_n,), dtype=x.dtype, device=x.device)


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # A group straddling TP ranks cannot be fixed by padding.
        if (
            c.group_size != -1
            and c.group_size < c.full_weight_shape[0]
            and c.partition_weight_shape[0] % c.group_size != 0
        ):
            return False, (
                f"in_features per partition {c.partition_weight_shape[0]} is "
                f"not divisible by group_size = {c.group_size}."
            )

        # Tile misalignment is fixed by zero-padding at weight prep.
        return True, None

    # ---- firefly 门控(int4 权重 → int8 prefill) ----
    def _firefly_enabled(self) -> bool:
        # 仅对 int4 权重 + fp16/bf16 激活启用; int8/fp8 激活(W4A8/FP8)走上游,
        # 不参与 firefly 反量化。
        # gate 卡 scalar type 而非量化框架:
        #   uint4b8 = 对称 int4(zp=8, compressed-tensors/GPTQ-sym)
        #   uint4   = 非对称 int4(per-group qzeros, AWQ)
        # 反量化统一 w_deq=(q-zp)*s, 对称 zp=8 是其特例(不回归)。
        # 只有 hard 模式: 不常驻副本, prefill 现从 marlin 布局反。B1-hard 反推
        # 仅针对 has_perm=false 的 marlin 布局; v0.30 gptq_marlin_repack 恒无 perm,
        # 与我方 AWQ(非 act-order)模型一致。若将来加载 act-order checkpoint,
        # firefly 会误对其反量化(错误结果), 故前提: 仅用于无 act-order 的模型库。
        return (
            firefly_active_int4()
            and self.config.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
            and self.config.act_type in (torch.float16, torch.bfloat16)
            and not getattr(self.config, "has_g_idx", False)
        )

    # ---- firefly idecode 门控(int4 decode → 自研 GEMV, 自然布局) ----
    def _idec_shape_ok(self) -> bool:
        # kernel 结构要求: 每 warp 一个 32*k 段(K%32)、每 block 8 行(N%8); scale 组
        # 宽度只支持 GS∈{64,128,256}(GS=-1 表示 per-channel, 组宽=K, 未覆盖)。
        # scale dtype = 激活 dtype(MPLinearKernel 契约), 故 act_type==fp16 同时保证
        # scale 落在 half2 反量化域(bf16 层不参与 idecode)。
        c = self.config
        size_k, size_n = c.partition_weight_shape
        shape_ok = (
            size_k % 32 == 0
            and size_n % 8 == 0
            and c.group_size in (64, 128, 256)
            and c.act_type == torch.float16
        )
        if not shape_ok:
            return False
        # smem 上界守卫: decode 激活 M 行各一份 swizzle 副本驻留 smem, 每行
        # kRowStrideHalf2(k_half2)=k_half2+k_half2/16+1 个 half2 × **4B**(勿按 2B,
        # 曾算小一半致 sx 越界)。M=IDEC_MAX_M 时若超 sm75 动态 smem opt-in 上限
        # (**64KB**, 实测 cudaDevAttr...Optin=65536; 96KB 是 sm80+ 值), 大 K 层会
        # launch 失败 → 该层回退 marlin(布局此刻未定, 安全)。生产 W4A16 层
        # (K=2560/640) M=8 时 smem 仅 42.5KB, 远低于上限, 不触发。
        k_half2 = size_k // 2
        smem = envs.VLLM_FIREFLY_IDEC_MAX_M * (k_half2 + k_half2 // 16 + 1) * 4
        return smem <= 64 * 1024

    def _idecode_enabled(self) -> bool:
        # idecode 生效条件 = firefly 生效 AND IDEC 开 AND JIT 成功 AND 形状支持 AND
        # 权重未被 fp8 预处理(marlin_int4_fp8_preprocess 会就地改 pack 语义)。
        # JIT 决策与 load 期布局选择、运行期分支同源: 都查 idecode_ready()。
        if not self._firefly_enabled():
            return False
        if not envs.VLLM_FIREFLY_IDEC:
            return False
        if self.config.act_type == torch.float8_e4m3fn:
            return False
        warm_idecode()  # 提前触发 JIT, 使本决策即最终结论
        return idecode_ready() and self._idec_shape_ok()

    def _firefly_m_large(self, x: torch.Tensor) -> bool:
        m = x.numel() // x.shape[-1]
        return m > envs.VLLM_FIREFLY_MIN_M

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        size_k, size_n = c.partition_weight_shape
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)
        # idecode 生效 → 权重改存自然 row-major int4, marlin 布局整体退场(单布局)。
        # 该标志必须在下面所有 repack 分支之前定下, 并记到 layer 供 apply_weights 用。
        natural = self._idecode_enabled()

        # hybrid: repack 前快照(此时 weight/scale 仍为干净布局, 未 permute/repack)。
        # clone 防后续 in-place 修改。scale 统一转置成 N 在前 [N, K/gs]
        # (compressed-tensors 已是 N 在前; AWQ 是 K 在前 [K/gs, N] 需转置)。
        # 只有 hard 模式: 存干净 scale(小, ~0 字节/参数) + 维度; 非对称(AWQ)另存
        # 干净 qzeros(packed, 小, dequant 时解包); 不存 int4 副本, prefill 步现从
        # marlin 布局反回(单权重, 不爆显存)。
        # Allocate marlin workspace, reusing existing storage on reload (v0.30 API;
        # 本树不移植 v0.31 的持久 workspace 管理器)。
        self.workspace = marlin_make_workspace_new(
            device, existing=getattr(self, "workspace", None)
        )

        if self._firefly_enabled():
            ws = getattr(layer, self.w_s_name).data.clone()
            if ws.shape[0] != size_n:  # AWQ K 在前 → N 在前
                ws = ws.t().contiguous()
            layer._firefly_ws = ws  # [N, K/gs]
            layer._firefly_gs = (
                self.config.group_size if self.config.group_size > 0 else -1
            )
            layer._firefly_size_k = size_k
            layer._firefly_size_n = size_n
            layer._firefly_padded_k = padded_k
            layer._firefly_padded_n = padded_n
            # 非对称(AWQ): 快照干净 qzeros(packed [N/8, K/gs]); 对称: None(zp=8)。
            layer._firefly_wzp = (
                getattr(layer, self.w_zp_name).data.clone() if c.zero_points else None
            )
            # 只有 hard: 不存干净 int4 副本, prefill 步现从 marlin 布局反回。

        # Default name since marlin requires empty parameter for zp,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        # idecode: 非对称(AWQ)的 zp 在 load 期一次性展开成干净 [padded_n, padded_k/gs]
        # fp16(decode kernel 与 prefill natural 路共用一份, 都在 fp16 域);
        # 对称则 None(kernel 内 zp=8)。padded 区 scale=0 → 该项乘 0, zp 取什么都不影响。
        if natural:
            scale_cols = padded_k // c.group_size
            if c.zero_points and layer._firefly_wzp is not None:
                zp_clean = _unpack_awq_zp(
                    layer._firefly_wzp, size_n, size_k // c.group_size
                ).to(torch.float16)
                if (padded_n, size_k // c.group_size) == (size_n, scale_cols):
                    zp_pad = zp_clean
                else:
                    zp_pad = torch.zeros(
                        (padded_n, scale_cols), dtype=torch.float16, device=device
                    )
                    zp_pad[:size_n, : size_k // c.group_size] = zp_clean
                layer._firefly_wzpe = zp_pad.contiguous()
            else:
                layer._firefly_wzpe = None
            # prefill 的 c_n 与 natural 反量化不再需要 packed qzeros(fp16 展开版够用)
            layer._firefly_wzp = None

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            if natural:
                # 自然布局: GPTQ packed int32 → uint8 [padded_n, padded_k/2]
                # (低 nibble = k 偶)。marlin 布局不再产出, 省 repack 也省显存。
                x.data = natural_pack_from_gptq(
                    x.data.contiguous(), size_k, size_n, padded_k, padded_n
                )
                return x
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            if natural:
                # scale 不做 marlin permute, 只 pad + N-first contiguous —— 于是
                # w_s 与 _firefly_ws(prefill/decode 共用的干净 scale)同物, 无歧义。
                x.data = marlin_pad_scales(
                    x.data.contiguous(),
                    size_n,
                    size_k,
                    padded_n,
                    padded_k,
                    c.group_size,
                ).contiguous()
                layer._firefly_ws = x.data
            else:
                x.data = marlin_permute_scales(
                    marlin_pad_scales(
                        x.data.contiguous(),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_k,
                    size_n=padded_n,
                    group_size=c.group_size,
                    is_a_8bit=is_a_8bit,
                )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if natural:
            # 自然布局下 marlin 的 w_zp 变换无用, 置 empty(省 padded_n*K/gs*4 字节)。
            setattr(layer, self.w_zp_name, marlin_make_empty(device))
        elif c.zero_points:
            grouped_k = size_k // c.group_size if c.group_size != -1 else 1
            padded_grouped_k = padded_k // c.group_size if c.group_size != -1 else 1
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    marlin_pad_scales(
                        unpack_cols(
                            x.t(),
                            c.weight_type.size_bits,
                            grouped_k,
                            size_n,
                        ),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_grouped_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            # idecode 的 GEMV 需要"干净" bias(marlin 的 permute 会把 N 顺序打乱);
            # marlin 路仍要 permute, 故先存干净副本再原 permute(不破坏现有语义)。
            if natural:
                layer._firefly_bias = layer.bias.data.clone()
            layer.bias.data = marlin_permute_bias(
                marlin_pad_dim(layer.bias, size_n, padded_n)
            )

        # firefly c_n 缓存: load 时算一次 per-channel c_n(纯权重导出, M/chunk 无关),
        # 运行时单遍反量化复用(免两遍 amax, 实测省反量化 ~46%)。放在 repack
        # (transform_w_q) 之后: 此时 w_q 已是 marlin 布局, 与 apply_weights 读取一致。
        # 复用现有全量反量化取 c_n(w_int8 临时算出即弃)。
        if natural:
            # 自然布局的 c_n: 直接从自然权重算(不反 repack), zp 用 load 期展开的
            # fp16 版 → 与运行期 natural 反量化(_ff_dequant_natural_cached_op)同域。
            layer._firefly_c_n = natural_c_n(
                getattr(layer, self.w_q_name).data,
                layer._firefly_ws,
                c.group_size,
                padded_k,
                padded_n,
                size_n,
                wzp=layer._firefly_wzpe,
            )
            # 记标志供 apply_weights(op 参数); marlin 布局不存在, 运行期只能走 idecode。
            layer._ff_natural_layout = True
        elif self._firefly_enabled():
            # 只有 hard: c_n 从 marlin 布局现算(无干净 int4 副本)
            _, layer._firefly_c_n = dequant_marlin_to_int8(
                getattr(layer, self.w_q_name).data,
                layer._firefly_ws,
                layer._firefly_gs,
                layer._firefly_size_k,
                layer._firefly_size_n,
                layer._firefly_padded_k,
                layer._firefly_padded_n,
                w_zp=layer._firefly_wzp,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        # v0.30 基类 _get_weight_params 返回 3 元组(w_q, w_s, w_zp), act-order 相关
        # 的 g_idx 权重已全线移除。
        w_q, w_s, w_zp = self._get_weight_params(layer)

        # hybrid: 大 M(prefill)把 int4 现反量化成 int8 走 cutlass(IMMA), 小 M(decode)
        # 走上游 int4 Marlin。整个"门控+选择+执行"包在 _ff_hybrid_linear(op) 里,
        # 对 inductor 不透明(图里一个 leaf), 避免 m>min_m 数据依赖分支拖慢 decode。
        # input_global_scale/bias/w_zp_clean 用空张量表示 None(custom op 不收 None)。
        if self._firefly_enabled():
            empty = torch.empty(0, device=x.device)
            igs = getattr(layer, "input_global_scale", None)
            igs_t = igs.data if igs is not None else empty
            bias_t = bias.data if bias is not None else empty
            wzp_c = getattr(layer, "_firefly_wzp", None)
            wzp_c_t = wzp_c if wzp_c is not None else empty
            # idecode(自然布局)专用参数: 展开的 fp16 zp / 干净 bias / M 上限。
            natural = bool(getattr(layer, "_ff_natural_layout", False))
            wzpe = getattr(layer, "_firefly_wzpe", None)
            wzpe_t = wzpe if wzpe is not None else empty
            bias_clean = getattr(layer, "_firefly_bias", None)
            bias_clean_t = bias_clean if bias_clean is not None else empty
            return _ff_hybrid_linear(
                x,
                w_q,
                w_s,
                w_zp,
                self.workspace,
                igs_t,
                bias_t,
                layer._firefly_ws,
                wzp_c_t,
                layer._firefly_c_n,
                c.partition_weight_shape[0],
                c.partition_weight_shape[1],
                layer._firefly_padded_k,
                layer._firefly_padded_n,
                layer._firefly_gs,
                envs.VLLM_FIREFLY_MIN_M,
                c.zero_points,
                envs.VLLM_FIREFLY_DEQUANT_MODEL == "fast",
                natural,
                wzpe_t,
                bias_clean_t,
                envs.VLLM_FIREFLY_IDEC_MAX_M,
            )

        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
