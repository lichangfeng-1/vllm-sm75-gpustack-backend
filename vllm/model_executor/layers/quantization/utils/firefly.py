# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly(int4→int8 prefill)混合方案辅助函数(SM75)。

思路: decode 保持 int4 Marlin(fp16 激活, 正确处理负 group scale);
prefill 大 M 时把 int4 权重反量化成 int8(per-group -> per-channel 折叠),
配 per-token 动态 int8 激活走 cutlass_scaled_mm(SM75 上即 IMMA tensor core)。

数学(与 P0 bench 验证一致, tmp/sm75-int8-verify/docker/p0_dequant_bench.py):
    w_deq[n,k] = (q[n,k] - 8) * s_g[n,k//gs]   # s_g 可负(AutoRound ~50% 负)
    c_n        = amax_k |w_deq[n,k]| / 127      # per-channel 正 scale
    w_int8[n,k]= clamp(round(w_deq[n,k] / c_n), -127, 127)
负 s_g 由 int8 码承载, c_n 取绝对值, 符号正确。

int8_prefill_linear(激活量化 + GEMM)供 int4 firefly prefill 使用;
fp8 firefly 已移除(sm75 实测不赚, fp8 恒走 marlin, 见 firefly_active_fp8)。
"""

import logging
import os

import torch

from vllm import _custom_ops as ops
import vllm.envs as envs

logger = logging.getLogger(__name__)


def firefly_active_int4() -> bool:
    """int4(W4A16) 是否走 firefly prefill: VLLM_FIREFLY_DIRECT 开(auto/1 等价)。"""
    return bool(envs.VLLM_FIREFLY_DIRECT)


def firefly_active_fp8() -> bool:
    """fp8(W8A8) prefill 是否走 firefly: 恒 False。

    fp8 firefly(fused / 非 fused 两子模式)已移除(sm75 实测 firefly-fp8 不比 marlin
    快, 加速走 VLLM_FIREFLY_AR), fp8 恒走 marlin。
    """
    return False

# B1-hard 的 reverse-repack+反量化 CUDA kernel(独立 torch extension, 首次用时编译)。
# 编译失败(无 nvcc/CUDA)或无 GPU 时回退 PyTorch 版(正确但慢 ~50ms/层)。
_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly.cu(torch.utils.cpp_extension.load), 失败返回 None。

    只在首次调用时编译(缓存到 torch extensions 目录), 不拖慢 vllm import。
    """
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly kernel source (firefly.cu) not found at %s; "
                "use PyTorch path",
                cu_path,
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75(T10)。本 overlay 面向 SM75, 强制单 arch, 首次 prefill
        # JIT 编译 ~2s(而非基础镜像多 arch 默认的 ~50s)。
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly CUDA kernel loaded (B1-hard fast path)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境回退 PyTorch
        logger.warning(
            "firefly CUDA ext load failed, fallback to PyTorch: %s", e
        )
        _cuda_mod = None
    return _cuda_mod


def _unpack_awq_zp(
    qzeros_packed: torch.Tensor, size_n: int, scale_cols: int
) -> torch.Tensor:
    """AWQ 干净 qzeros [N/8, K/gs] int32(nibble j = N offset j) -> 干净 zp [N, K/gs] float。

    镜像 auto_awq._convert_awq_to_standard_format 的 qzeros 打包:
    new_qz[n//8, g] = Σ_j zp[n//8*8+j, g] << 4j, 故 nibble j = N offset j。
    """
    shifts = torch.arange(0, 32, 4, device=qzeros_packed.device)
    zp = (qzeros_packed.long().unsqueeze(-1) >> shifts) & 0xF  # [N/8, K/gs, 8]
    zp = zp.permute(0, 2, 1).reshape(size_n, scale_cols)  # [N, K/gs]
    return zp.float()


def int8_prefill_linear(
    x: torch.Tensor,
    w_int8: torch.Tensor,  # [N, K]
    c_n: torch.Tensor,  # [N]
) -> torch.Tensor:
    """x [*, K] -> per-token 动态 int8 -> cutlass_scaled_mm -> [*, N](x.dtype)。

    cutlass 要求 b 列主序(w_int8.t(), stride(0)==1, 勿 contiguous)。
    """
    orig = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    x_q, x_s, _ = ops.scaled_int8_quant(x2d.contiguous())
    y = ops.cutlass_scaled_mm(
        x_q, w_int8.t(), scale_a=x_s, scale_b=c_n, out_dtype=x.dtype
    )
    return y.reshape(*orig[:-1], -1)


# ---- B1-hard: 从 marlin 布局现反回干净 int4(不存干净 int4 副本) ----
# 镜像 csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu 的
# repack_tile, 仅 int4/fp16 场景(num_bits=4, has_perm=false, is_a_8bit=false):
#   输出 tile 位置 p, 字节内 nibble i 对应干净 (dk, dn):
#     a=p//4, b=p%4; tc_row=(a%4)*2, cur_n=b*16+a//4
#     dk=tc_row+8*(i%2)+(i//4);  dn=cur_n+8*((i//2)%2)
#   反解 (dk, dn) -> (i, p):
#     dk_lo=dk%8, dk_hi=dk//8, tc_row=dk_lo&~1, dk_off_lo=dk_lo&1
#     dn_off8=(dn%16>=8); cur_n=dn-8*dn_off8
#     i=4*dk_off_lo+2*dn_off8+dk_hi;  p=((cur_n%16)*4+tc_row//2)*4+(cur_n//16)
#   tile 顺序: tile_idx=k_tile*(N//64)+n_tile, 每 tile 128 个 int32。
# round-trip 已在 tmp/sm75-int8-verify/docker/p2_reverse_repack_test.py 逐 bit 验证。

def _reverse_repack_indices(size_k: int, size_n: int, device) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """对每个干净 (n, k) 预计算它在 marlin 扁平张量里的位置(src_flat)和位移(shift)。"""
    K, N = size_k, size_n
    n_tiles = N // 64
    n = torch.arange(N, device=device)[:, None]  # [N, 1]
    k = torch.arange(K, device=device)[None, :]  # [1, K]
    nt = n // 64
    dn = n % 64
    kt = k // 16
    dk = k % 16
    dk_lo = dk % 8
    dk_hi = dk // 8
    tc_row = dk_lo & ~1  # 向下取偶
    dk_off_lo = dk_lo & 1
    dn_off8 = (dn % 16 >= 8).long()
    cur_n = dn - dn_off8 * 8
    i = 4 * dk_off_lo + 2 * dn_off8 + dk_hi
    th_id = (cur_n % 16) * 4 + tc_row // 2
    p = th_id * 4 + (cur_n // 16)
    tile_idx = kt * n_tiles + nt
    src_flat = (tile_idx * 128 + p).long()  # [N, K]
    shift = (4 * i).long()  # [N, K]
    return src_flat, shift


def marlin_to_int4_q(
    marlin_w: torch.Tensor,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
    natural: bool = False,
) -> torch.Tensor:
    """marlin int4 [padded_k/16, padded_n*2] int32 -> 干净 int4 值 q [size_n, size_k]。

    反 gptq_marlin_repack(int4/fp16, 非 act-order)。padded_* 缺省取 size_*(无 padding)。
    有 tile padding 时先用 padded 维度反回再裁到 size。
    natural=True: 权重已是自然 row-major uint8 [padded_n, padded_k/2](VLLM_FIREFLY_IDEC
    换了 repack 目标, marlin 布局退场), 不做反 repack, 直接展开高低 nibble(低 4bit = k
    偶、高 = k 奇, 与 firefly_idecode 的 pack 约定一致)。
    """
    pk = padded_k if padded_k is not None else size_k
    pn = padded_n if padded_n is not None else size_n
    if natural:
        q_pad = unpack_natural_q(marlin_w, pn, pk)
        return q_pad[:size_n, :size_k]
    M_flat = marlin_w.reshape(-1).long()
    src_flat, shift = _reverse_repack_indices(pk, pn, marlin_w.device)
    q_pad = ((M_flat[src_flat] >> shift) & 0xF).to(torch.uint8)  # [padded_n, padded_k]
    return q_pad[:size_n, :size_k]


def unpack_natural_q(
    natural_w: torch.Tensor, padded_n: int, padded_k: int
) -> torch.Tensor:
    """自然布局 [padded_n, padded_k/2] uint8 -> 每 nibble 一字节 [padded_n, padded_k]。

    优先 firefly_idecode 的 CUDA kernel(无索引 gather); JIT 不可用时 PyTorch 回退。
    """
    from vllm.model_executor.layers.quantization.utils import firefly_idecode

    return firefly_idecode.unpack_natural(natural_w, padded_n, padded_k)


def dequant_marlin_to_int8(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,  # 干净 scale [N, K/group_size] (N 在前)
    group_size: int,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
    w_zp: torch.Tensor | None = None,  # None=对称(zp=8); AWQ qzeros [N/8, K/gs]
    natural: bool = False,  # 权重是自然布局(VLLM_FIREFLY_IDEC), 不做反 repack
    # natural=True 时 w_zp 的语义变为**已展开的 [padded_n, padded_k/gs] fp16 zp**
    # (load 期 _firefly_wzpe), 与运行期 natural 反量化 / idecode kernel 同 fp16 域。
) -> tuple[torch.Tensor, torch.Tensor]:
    """B1-hard: marlin int4 + 干净 scale (+zp) -> (w_int8 [N, K], c_n [N])。

    反量化 w_deq=(q-zp)*s; 对称 zp=8(uint4b8), 非对称(AWQ) zp=per-group qzeros。
    优先 CUDA kernel(firefly_dequant, 寄存器现算 marlin 位置, ~ms 级);
    编译失败/无 GPU 时回退 PyTorch 版(正确但 ~50ms/层)。不缓存 int8。
    natural=True: marlin_w 实为自然布局 uint8 [pn, pk/2], 走 unpack + PyTorch 折叠
    (本函数只在 load 期取 c_n 用一次, 无性能要求)。
    """
    N, K = size_n, size_k
    if natural:
        # 自然布局: 折叠反量化走分组算(fold_groupwise_dequant), 免 repeat_interleave
        # 的两个 [N,K] fp32 大临时(T10 显存余量≈0); zp 走 load 期展开的 fp16 版
        # (w_zp 参数在 natural 下即该张量, None = 对称 zp=8)。
        from vllm.model_executor.layers.quantization.utils.firefly_idecode import (
            fold_groupwise_dequant,
            unpack_natural,
        )

        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        q = unpack_natural(marlin_w, pn, pk)[:N]
        w_deq = fold_groupwise_dequant(q, weight_scale[:N], group_size, pk, w_zp)
        c_n = w_deq.abs().amax(dim=1) / 127.0
        w_int8 = torch.clamp(torch.round(w_deq / c_n.unsqueeze(1)), -127, 127).to(
            torch.int8
        )
        return w_int8, c_n
    mod = _load_cuda_mod()
    if mod is not None and marlin_w.is_cuda:
        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        # zp: 对称→空张量(kernel 取 8.0f); 非对称→干净 [N, K/gs] float
        zp_t = (
            torch.empty(0, dtype=torch.float32, device=marlin_w.device)
            if w_zp is None
            else _unpack_awq_zp(w_zp, N, weight_scale.shape[1])
        )
        out_int8 = torch.empty((N, K), dtype=torch.int8, device=marlin_w.device)
        c_n = torch.empty((N,), dtype=torch.float32, device=marlin_w.device)
        mod.firefly_dequant(
            marlin_w,
            weight_scale,
            zp_t,
            out_int8,
            c_n,
            N,
            K,
            pn,
            pk,
            group_size,
        )
        return out_int8, c_n

    # PyTorch 回退路径
    q = marlin_to_int4_q(marlin_w, size_k, size_n, padded_k, padded_n)  # [N, K]
    s = weight_scale.float().repeat_interleave(group_size, dim=1)  # [N, K]
    zp = (
        8.0
        if w_zp is None
        else _unpack_awq_zp(w_zp, N, weight_scale.shape[1]).repeat_interleave(
            group_size, dim=1
        )
    )
    w_deq = (q.float() - zp) * s  # [N, K]
    c_n = w_deq.abs().amax(dim=1) / 127.0  # [N] 正
    w_int8 = torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
    return w_int8, c_n


# ---- torch.compile(fullgraph) 兼容 ----
# 裸 pybind 反量化 kernel dynamo 追不了: VLLM_COMPILE(fullgraph=True, 即去掉
# --enforce-eager 后) 追到 prefill 分支会抛 "Unsupported: Attempted to call
# function marked as skipped"(gb0007)。注册成 torch.library op 后 dynamo 当
# 不透明 op 处理, 兼容 fullgraph 编译 + CUDA graph capture。CUDA mod 仍懒加载:
# op body 首次执行才 _load_cuda_mod()(JIT 编译), 不拖 vllm import。
@torch.library.custom_op("firefly::dequant_marlin_cached", mutates_args=())
def _ff_dequant_marlin_cached_op(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,
    zp_t: torch.Tensor,
    c_n: torch.Tensor,
    n: int,
    k: int,
    pn: int,
    pk: int,
    gs: int,
    use_recip: bool,
) -> torch.Tensor:
    mod = _load_cuda_mod()
    out_int8 = torch.empty((n, k), dtype=torch.int8, device=marlin_w.device)
    kernel = (
        mod.firefly_dequant_cached_recip if use_recip else mod.firefly_dequant_cached
    )
    kernel(marlin_w, weight_scale, zp_t, out_int8, c_n, n, k, pn, pk, gs)
    return out_int8


@_ff_dequant_marlin_cached_op.register_fake
def _(_marlin_w, _weight_scale, _zp_t, _c_n, n, k, _pn, _pk, _gs, _use_recip):
    return torch.empty((n, k), dtype=torch.int8, device=_marlin_w.device)


# natural 版单遍反量化(VLLM_FIREFLY_IDEC 开、权重是自然 row-major 布局时用):
# 同样注册成 torch.library op(兼容 VLLM_COMPILE fullgraph + cudagraph)。
# zp 传 **fp16**(与 idecode kernel 同约定: 非对称的展开 zp 在 load 期就转 fp16,
# prefill/decode 共用一份; marlin 版则用 fp32, 二者仅 zp 域有极小差异)。
@torch.library.custom_op("firefly::dequant_natural_cached", mutates_args=())
def _ff_dequant_natural_cached_op(
    natural_w: torch.Tensor,
    weight_scale: torch.Tensor,
    zp_t: torch.Tensor,
    c_n: torch.Tensor,
    n: int,
    k: int,
    gs: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.firefly_idecode import (
        fold_groupwise_dequant,
        unpack_natural,
    )

    q = unpack_natural(natural_w, n, k)
    # 无 zp 时调用方传 torch.empty(0, dtype=fp16)(custom op 不收 None); 用 numel 判空
    # (与 firefly.cu 侧 zp 的 numel() 判据一致)。
    w_deq = fold_groupwise_dequant(
        q, weight_scale, gs, k, (zp_t if zp_t.numel() > 0 else None)
    )
    r = torch.where(c_n > 0, 1.0 / c_n, 0.0)
    return torch.clamp(torch.round(w_deq * r.unsqueeze(1)), -127, 127).to(torch.int8)


@_ff_dequant_natural_cached_op.register_fake
def _(_natural_w, _weight_scale, _zp_t, _c_n, n, k, _gs):
    return torch.empty((n, k), dtype=torch.int8, device=_natural_w.device)


def dequant_marlin_to_int8_cached(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,  # 干净 scale [N, K/group_size] (N 在前)
    c_n: torch.Tensor,  # [N] 预计算 per-channel scale(load 时算一次)
    group_size: int,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
    w_zp: torch.Tensor | None = None,  # None=对称(zp=8); AWQ qzeros [N/8, K/gs]
    use_recip: bool = False,
    natural: bool = False,  # 权重是自然布局(VLLM_FIREFLY_IDEC), 走 natural op
    wzp_fp16: torch.Tensor | None = None,  # natural 用: 已展开 [pn,pk/gs] fp16 zp
) -> torch.Tensor:
    """B1-hard 单遍反量化: c_n 已预计算, 只跑量化 pass(免两遍 amax)。

    use_recip=False: 除法, 与 dequant_marlin_to_int8 的 pass2 逐 bit 一致。
    use_recip=True : 乘倒数(fast), 更快但 off-by-one ≤0.06%。
    优先 CUDA kernel(firefly_dequant_cached[_recip]); 编译失败/无 GPU 回退 PyTorch 版。
    返回 w_int8 [N, K] int8。
    natural=True: marlin_w 实为自然布局 uint8 [pn, pk/2], 走
    _ff_dequant_natural_cached_op(unpack + PyTorch 折叠), zp 用已展开 fp16 wzp_fp16
    (与 idecode kernel 同 fp16 域, 两条路径数值一致)。
    """
    N, K = size_n, size_k
    if natural:
        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        zp_t = (
            torch.empty(0, dtype=torch.float16, device=marlin_w.device)
            if wzp_fp16 is None
            else wzp_fp16
        )
        return _ff_dequant_natural_cached_op(
            marlin_w, weight_scale, zp_t, c_n, N, K, group_size
        )
    if _load_cuda_mod() is not None and marlin_w.is_cuda:
        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        zp_t = (
            torch.empty(0, dtype=torch.float32, device=marlin_w.device)
            if w_zp is None
            else _unpack_awq_zp(w_zp, N, weight_scale.shape[1])
        )
        # 走 torch.library op(见上定义), 兼容 VLLM_COMPILE fullgraph。
        return _ff_dequant_marlin_cached_op(
            marlin_w, weight_scale, zp_t, c_n, N, K, pn, pk, group_size, use_recip
        )

    # PyTorch 回退路径
    q = marlin_to_int4_q(marlin_w, size_k, size_n, padded_k, padded_n)  # [N, K]
    s = weight_scale.float().repeat_interleave(group_size, dim=1)  # [N, K]
    zp = (
        8.0
        if w_zp is None
        else _unpack_awq_zp(w_zp, N, weight_scale.shape[1]).repeat_interleave(
            group_size, dim=1
        )
    )
    w_deq = (q.float() - zp) * s  # [N, K]
    if use_recip:
        r = torch.where(c_n > 0, 1.0 / c_n, 0.0)
        return torch.clamp(torch.round(w_deq * r.unsqueeze(1)), -127, 127).to(torch.int8)
    return torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
