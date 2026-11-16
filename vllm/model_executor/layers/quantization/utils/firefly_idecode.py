# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly idecode: SM75 W4A16 decode(小 M)自研 int4 GEMV。

与 firefly.py(prefill 反量化 int8 → cutlass IMMA)互补: 这里管 **decode 小 M**,
用自研 GEMV 取代 Marlin int4 GEMV(g292 实测 9.5~12us vs marlin 27~30us)。

前提: 权重必须存成**自然 row-major int4 布局** uint8 [N, K/2](低 4bit = k 偶,
高 4bit = k 奇), 由 mixed_precision/marlin.py 的 process_weights_after_loading 在
VLLM_FIREFLY_IDEC 生效时改换 repack 目标(marlin 布局整体退场, T10 显存余量≈0,
不做双副本)。prefill 侧的反量化相应走 natural 分支(firefly.py)。

可用状态是**模块级单例**: JIT 编译失败(nvcc 缺失/编译错)→ idecode_ready() 永久
False, process_weights 决策与 apply 分支同源于此, 全链路退回现状(纯 marlin)。
独立 torch extension(namespace firefly_idecode), 懒加载, 只编 sm_75。
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly_idecode.cu, 失败返回 None(与 firefly.py 同款模式)。"""
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly_idecode.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly_idecode kernel source (firefly_idecode.cu) not found at %s; "
                "int4 decode GEMV 不可用, 回退 Marlin",
                cu_path,
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75(本 overlay 面向 SM75): 强制单 arch, 避免多 arch 默认 ~50s。
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_idecode_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly_idecode CUDA kernel loaded (int4 decode GEMV)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境一律回退 marlin
        logger.warning("firefly_idecode CUDA ext load failed, decode 回退 Marlin: %s", e)
        _cuda_mod = None
    return _cuda_mod


def warm_idecode() -> None:
    """提前触发 JIT 编译(weight 加载前调用), 使决策与执行同源。"""
    _load_cuda_mod()


def idecode_ready() -> bool:
    """GEMV 是否可用(JIT 成功)。process_weights 与 apply 分支必须同查此函数。"""
    return _load_cuda_mod() is not None


def natural_pack_from_gptq(
    qweight: torch.Tensor,  # GPTQ 输入布局 [size_k/8, size_n] int32
    size_k: int,
    size_n: int,
    padded_k: int,
    padded_n: int,
) -> torch.Tensor:
    """GPTQ packed int32 → 自然布局 uint8 [padded_n, padded_k/2](load 时一次)。

    GPTQ 输入: word = k//8, nibble = k%8(低 4bit 起)。自然布局: 每 byte 两个
    nibble, **低 4bit = k 偶、高 4bit = k 奇**(与 kernel / unpack_natural 约定一致)。
    padding 区 weight nibble 填 8: 对称语义(uint4b8)下反量化恒 0, 避免 NaN/污染。
    """
    assert qweight.dtype == torch.int32 and qweight.dim() == 2
    assert qweight.shape == (size_k // 8, size_n)
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    # [size_k/8, size_n, 8] → [size_k, size_n](nibble 顺序即 k 递增)
    q = (
        (qweight.to(torch.int64).unsqueeze(-1) >> shifts.to(torch.int64)) & 0xF
    ).reshape(size_k, size_n)
    q = q.t().contiguous()  # [size_n, size_k]
    if (padded_n, padded_k) != (size_n, size_k):
        q = torch.nn.functional.pad(q, (0, padded_k - size_k, 0, padded_n - size_n), value=8)
    q = q.to(torch.uint8)
    lo, hi = q[:, 0::2], q[:, 1::2]
    return (lo | (hi << 4)).contiguous()  # [padded_n, padded_k/2]


def unpack_natural(w: torch.Tensor, n_rows: int, k: int) -> torch.Tensor:
    """自然布局 [N,K/2] uint8 → 每 nibble 一字节 [N,K] uint8(CUDA kernel)。"""
    mod = _load_cuda_mod()
    if mod is None or not w.is_cuda:
        # PyTorch 回退(load 期一次性, 慢但正确)。
        lo = (w & 0xF).to(torch.uint8)
        hi = (w >> 4).to(torch.uint8)
        return torch.stack((lo, hi), dim=1).reshape(n_rows, k).contiguous()
    return mod.unpack_natural(w, n_rows, k)


def natural_c_n(
    w_natural: torch.Tensor,  # [padded_n, padded_k/2] uint8 自然布局
    scale: torch.Tensor,  # [padded_n, padded_k/gs] 干净 N-first scale(fp16)
    gs: int,
    padded_k: int,
    padded_n: int,
    size_n: int,
    wzp: torch.Tensor | None = None,  # 非对称已展开 [padded_n, padded_k/gs] fp16
) -> torch.Tensor:
    """自然布局权重的 per-channel c_n(prefill int8 GEMM 的 b scale, load 时算一次)。

    与 firefly.dequant_marlin_to_int8 的 c_n 同定义: amax_k |(q-zp)*s| / 127。
    用 padded 维度算(与 marlin 版按 padded 反解一致), 截回 size_n; padding 行/列
    的 scale=0(或 weight=8)使该项为 0, 不污染。
    峰值显存注意: repeat_interleave 会产出 [padded_n, padded_k] fp32(27B TP8 单层
    ~63MB), 只在 load 期瞬时占用。
    """
    q = unpack_natural(w_natural, padded_n, padded_k)
    w_deq = fold_groupwise_dequant(q, scale, gs, padded_k, wzp)
    c = w_deq.abs().amax(dim=1) / 127.0
    return c[:size_n].contiguous()


def fold_groupwise_dequant(
    q: torch.Tensor,  # [padded_n, padded_k] uint8(每 nibble 一字节)
    scale: torch.Tensor,  # [padded_n, padded_k/gs] fp16/float
    gs: int,
    padded_k: int,
    wzp: torch.Tensor | None = None,  # 非对称 [padded_n, padded_k/gs](fp16); None=8
) -> torch.Tensor:
    """per-group 折叠反量化 (q-zp)*s → [padded_n, padded_k] fp32(分组算, 控峰值)。

    逐组 gs 列切片相乘, 避免 repeat_interleave 一次性产出两个 [N,K] fp32 大临时
    (prefill 每步都要反量化, T10 显存余量≈0)。
    """
    n_rows = q.shape[0]
    out = torch.empty((n_rows, padded_k), dtype=torch.float32, device=q.device)
    groups = padded_k // gs
    q32 = q  # uint8, 逐组切片后 .float()
    for g in range(groups):
        sl = slice(g * gs, (g + 1) * gs)
        s = scale[:, g].float().unsqueeze(1)
        z = 8.0 if wzp is None else wzp[:, g].float().unsqueeze(1)
        out[:, sl] = (q32[:, sl].float() - z) * s
    return out


def natural_to_gptq_packed(
    w_natural: torch.Tensor,
    size_k: int,
    zp: torch.Tensor | None = None,
) -> torch.Tensor:
    """自然布局 uint8 [N, K/2] → GPTQ packed int32 [K/8, N](nibble 域)。

    打包(元素 j 在 word j//8 的 nibble j%8, 行主序 N)按 GPTQ 约定。可选先按 per-group
    zp 平移: packed_nib = clip(nibble - zp + 8, 0, 15), 配合 marlin 的 uint4 **非对称**
    路(wtype=uint4 + b_zeros = zp-8)可精确复现 idecode 的 (q-zp) 语义:
      marlin: (packed_nib - (zp-8) - 8) = clip(nib-zp) = nib - zp(值域内不裁剪)。
    zp=None 即对称语义(nibble 直接就是 uint4b8 码, marlin 传 wtype=uint4b8 无 zp)。
    仅测试/交叉比用, 不在运行路径上。
    """
    n, half_k = w_natural.shape
    K = half_k * 2
    lo = (w_natural & 0xF).to(torch.int16)
    hi = (w_natural >> 4).to(torch.int16)
    q = torch.stack((lo, hi), dim=1).reshape(n, K)  # [N,K] int16, 0..15
    if zp is not None:
        g = (torch.arange(K, device=w_natural.device) // (K // zp.shape[1]))[None, :]
        shift = zp.to(torch.int16)[:, g]  # [N,K] = zp[n, k//gs]
        q = (q - shift + 8).clamp(0, 15)
    nib = q.to(torch.int64) & 0xF
    shifts = torch.arange(0, 32, 4, device=w_natural.device, dtype=torch.int64)
    words = (nib.reshape(n, size_k // 8, 8) << shifts).sum(dim=2).to(torch.int32)
    return words.t().contiguous()  # [K/8, N]


def idecode_gemv(
    x2d: torch.Tensor,  # [M, size_k] 激活(fp16, 行主序)
    w_natural: torch.Tensor,  # [padded_n, padded_k/2] uint8
    scale: torch.Tensor,  # [padded_n, padded_k/gs] fp16 N-first
    gs: int,
    size_k: int,
    size_n: int,
    wzp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """decode GEMV: out[m,n] = sum_k (q-zp)*s*x + bias → [M, size_n](x.dtype)。

    调用方保证: x dtype fp16、contiguous、x.size(-1) == size_k(未 pad, 本函数按
    padded_k 补零)、M<=8。scale/zp 必须是 **fp16**(kernel 在 half2 域做反量化)。
    """
    mod = _load_cuda_mod()
    if mod is None:
        raise RuntimeError("firefly_idecode kernel 不可用; 调用方应回退 Marlin")
    m = x2d.shape[0]
    padded_k = w_natural.shape[1] * 2
    if padded_k != size_k:
        # K 有 padding(marlin tile 对齐): 激活补零到 padded_k, 与 marlin 同语义。
        x2d = torch.nn.functional.pad(x2d, (0, padded_k - size_k))
    empty = torch.empty(0, device=x2d.device)
    out = mod.ff_idecode_gemv(
        x2d.contiguous(),
        w_natural,
        scale,
        wzp if wzp is not None else empty,
        bias if bias is not None else empty,
        int(gs),
        int(m),
        int(size_n),
    )
    return out
