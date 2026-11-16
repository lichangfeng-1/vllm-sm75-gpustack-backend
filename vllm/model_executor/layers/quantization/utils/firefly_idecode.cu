// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// firefly idecode (SM75 W4A16 decode): 自研 M<=8 int4 GEMV 取代 Marlin decode。
//
// 权重是**自然 row-major int4 布局** uint8 [N, K/2](低 4bit = k 偶, 高 4bit = k 奇,
// 与 PoC tmp/decode_gemv_poc 的 pack 约定一致), 不是 marlin tile 布局。故 load 时
// repack 目标改为自然布局, marlin 布局整体退场(T10 显存余量≈0, 不做双副本)。
//
// 数值语义 w_deq = (q - zp) * s:
//   对称(uint4b8, GPTQ-sym/compressed-tensors): zp = 8 常数(不传 zp 指针)。
//   非对称(uint4, AWQ): 传 load 期已解包的干净 zp [N, K/gs] fp16, kernel 内只做一次
//   乘减(fp16), 与 prefill 的 fp32 zp 路径存在极小差异(由 g292 对 torch 参考实测兜底)。
//
// kernel 结构移植自 PoC 的 gemv_m1_v2(9.5~12us vs marlin 27~30us, 数值 PASS):
//   每 block 8 warp = 8 个 N 行, 激活整块进 smem 并 swizzle(每 16 个 half2 补 1,
//   消 16-way bank 冲突——PoC 实测根因), 每 warp 一行按 uint4(16B)流式读权重,
//   每 lane 一个 scale 组内做 half2 反量化 + fp32 累加, warp shuffle 归约。
//   M>1: 同一 warp 内层对 M 份激活各累加一份 fp32 寄存器(模板参数 M, 1..8 全实例化),
//   权重只读一遍 —— 这是 idecode 相对 marlin 的主要结构优势(decode batch 越小越赚)。
//
// 约束(不满足则该层回退 marlin): 激活 fp16、K%32==0、N%8==0、GS ∈ {64,128,256}。
// padding: 自然布局必须按 padded_n/padded_k 存(c_n/scale 已按 padded 布局、prefill
// 的 cutlass 输出是 padded_n), 否则要每步切片。padding 区 weight nibble 填 8(对称语义
// 下反量化恒 0)→ 对称路径 padded 行/列贡献严格为 0; 非对称 padding scale=0 同样为 0。
// 输出按 size_n 截回, 与 marlin 的 unpadded 语义一致。
//
// 算子(独立 torch extension, namespace firefly_idecode, 不污染 vllm._C):
//   ff_idecode_gemv(x[M,K]fp16, w[N_p,K_p/2]uint8, s[N_p,K_p/gs]fp16,
//                   wzp(可空 [N_p,K_p/gs]fp16), bias(可空 [N]fp16), gs, M, size_n)
//     -> out [M, size_n] fp16
//   unpack_natural(w[N_p,K_p/2]uint8, N_p, K_p) -> q [N_p,K_p] uint8(自然布局展开,
//     无索引 gather 的精确版, 供 load 期算 c_n 与 host 侧测试参考)
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API, 不拖 cusparse/cublas

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

// M<=8 时每 block 的 warp 数(= 一次覆盖的 N 行数)。
constexpr int kRows = 8;

// uint32 寄存器视图转 half2(half2 在寄存器就是一个 b32)。
__device__ __forceinline__ half2 u32_as_h2(uint32_t v) {
  half2 r;
  asm("mov.b32 %0, %1;" : "=r"(*reinterpret_cast<uint32_t*>(&r)) : "r"(v));
  return r;
}

// 一个 uint32 = 8 个 int4(little-endian: byte0 低 nibble = k0, 高 = k1, ...)。
// magic-bitcast: bitcast<half>(0x6400|v) = 1024+v, 末尾统一减 1024+zp 偏移。
// h[j] = half2(k(2j), k(2j+1)), 位式逐字节精确拼接(LOP3+PRMT 友好)。
// zp_off = 1024+8(对称 uint4b8) 或 1024+zp(非对称, 该 scale 组的 fp16 zp 已转 float)。
__device__ __forceinline__ void dequant8(uint32_t word, float zp_off,
                                         half2 h[4]) {
  const uint32_t magic = 0x64006400u;
  h[0] = u32_as_h2(((word & 0xF0u) << 12) | (word & 0x0Fu) | magic);
  h[1] = u32_as_h2(((word & 0xF000u) << 4) | ((word & 0x0F00u) >> 8) | magic);
  h[2] = u32_as_h2(((word & 0xF00000u) >> 4) | ((word & 0xF0000u) >> 16) | magic);
  h[3] =
      u32_as_h2(((word & 0xF0000000u) >> 12) | ((word & 0xF000000u) >> 24) | magic);
  const half2 off = __floats2half2_rn(zp_off, zp_off);
#pragma unroll
  for (int j = 0; j < 4; j++) h[j] = __hsub2(h[j], off);
}

// scale 组偏移: uint4 下标 idx(覆盖 k = [32*idx, 32*idx+32)) 落在哪个 scale 组。
// GS>=32 且 GS%32==0: 一个 uint4 恒在同一组 → 组号 = 32*idx/GS(移位)。
template <int GS>
__device__ __forceinline__ int scale_group(int idx) {
  return (idx * 32) / GS;
}

// 激活 smem 的 swizzle: 每 16 个 half2 补 1 个空槽。半字(4B)访问 × 连续下标 × 32 lane
// 会撞在同一地址区间 → 16-way bank 冲突(PoC 实测的根因), 补槽打散。
//
// 一切换 helper 都标 __host__ __device__: 只写 constexpr(=host-only)时 device 端调用在
// torch extension 这种"不解析未定义符号的 .so"里会静默留下 unresolved call → 运行期跳飞
// = illegal memory access(本轮 g292 实测踩到)。
//
// 【swizzle 越界 gotcha(务必保留)】历史教训: 上一轮"smem 读 = illegal access"的
// 真正根因是**分配尺寸算小了**(按无 padding 的 K/2 个 half2 分配, 而 swz 写入要
// K/2 + K/32 上界), 曾一度误判为 CUDA13+sm75 的 extern __shared__ 声明问题或
// "smem 读本身有毒"。kSmemBytes(host) 与 kernel 内 stride **必须同源**于
// kRowStrideHalf2, 改一侧另一侧不跟 = 又踩 CUDA **异步 sticky** 越界(现场会归罪到
// 后面某个 torch op)。K=2560 时最大写入下标 = 639+639/16 = 679, 故每行须 ≥ 680。
//
// 逻辑 half2 下标 p → 存储下标 p + p/16(整行单调 bijection, 与 PoC v2 逐位一致:
// 装载 sa2[i + i/16], warp 读 16idx 起 16 个连续 half2 时 p/16 恒等于 idx → 读址
// 17idx + offset)。17 与 bank 数 32 近似互质, 32 lane × 2B 读分两个 phase 各落 16
// 个不同 bank, 无冲突(冲突根因=无 padding 时 32 lane 落 8 bank 4-way)。
// 映射是整行的, 任何"分段取模"变体(如 p%512 后加段基)必须显式加上 544*(p/512)
// 才等价 —— 漏了段基跨段别名。分配上界 = swz(K/2-1) = K/2-1 + K/32-1,
// 按每行 K/2 + K/32 + 1 个 half2 分配恒不越界(K%32==0 时 K/32 取整精确)。
__host__ __device__ __forceinline__ constexpr int swz_abs(int p) {
  return p + p / 16;
}
// 每行激活驻留的 half2 数(swz_abs 上界 + 1)。
__host__ __device__ __forceinline__ constexpr int kRowStrideHalf2(int k_half2) {
  return k_half2 + k_half2 / 16 + 1;
}

// kernel 形态: 两个同构的 smem swizzle kernel, 只在激活装载与行偏移上不同。
//   M==1 专化版: 激活按全局线性下标一次 swizzle 装载, 无逐行换算(decode 主力)。
//   M>=2 通用版: 每行一份 swizzle 副本 + 预取流水(投机解码 batch)。
// smem 按"每行 kRowStrideHalf2 个 half2"分配(M 行 = M 份)。
//
// 【环境 gotcha(务必保留)】g292 同款环境(CUDA 13.0 nvcc + -arch=sm_75)独立探针实测
// (min_probe_nan.cu, 同一 smem 装载逻辑只改声明类型): `extern __shared__ char sm[]`
// 再 reinterpret 成 half2* 的写入会被**静默丢弃**(装载后读回全 0, 不报 illegal) →
// 输出 NaN/0; `extern __shared__ half sm[]` 版装载正确(bad=0)。与尺寸/launch_bounds
// 无关。故本文件统一用 **half[]** 声明动态 smem(与 PoC v2 的 `extern __shared__ half
// sa_pad[]` 一致 —— 那是真正跑通 9.5us 的形态)。曾误判"char 正常/half 挂 illegal",
// 真相是 char 装载静默失效被当成"能跑"。
__host__ __device__ __forceinline__ constexpr int kSmemBytes(int m, int k_pad) {
  const int k_half2 = k_pad / 2;
  // half2 = 4B(2 个 half × 2B), 勿写 2 —— 曾按 2 算只分到一半 smem, sx 写入越界
  // = illegal memory access(CUDA_LAUNCH_BLOCKING 定位, 本 case K=2560 swz 上界
  // 1358 需 1359 half2 = 5436B, *2 只给 2722B)。PoC 的 BK*2+BK/4 = 5440B 口径一致。
  return m * kRowStrideHalf2(k_half2) * 4;
}

// decode 的主力形态(M=1, 单流 decode): 与通用版同一套 smem 装载(swizzle padding),
// 差别只有一处 —— 激活只有一行, 按全局线性下标直接装载, 省掉逐行行内下标换算。
template <int GS>
__global__ __launch_bounds__(kRows * 32) void ff_idecode_gemv_m1_kernel(
    const half* __restrict__ x, const uint4* __restrict__ w,
    const half* __restrict__ s, const half* __restrict__ wzp,
    const half* __restrict__ bias, half* __restrict__ out, int n_rows, int k_pad,
    int out_cols) {
  // 动态 smem 声明用 half[](本环境 CUDA 13.0 + sm_75 独立探针实测: 声明成 char[] 再
  // reinterpret 成 half2*, 写入会被**静默丢弃**(装载后读回全 0, 不报 illegal) → 输出
  // NaN/0; 换 half[] 即正常 —— 见 kSmemBytes 附近 gotcha。PoC v2 用的正是 half[]。)
  extern __shared__ half sm_x[];
  const int tid = threadIdx.x;
  const int k_half2 = k_pad / 2;
  half2* const sx = reinterpret_cast<half2*>(sm_x);
  const half2* const xv = reinterpret_cast<const half2*>(x);
  // 装载即落 swizzle 槽: p 是逻辑 half2 下标, 存储下标 = swz_abs(p) = p + p/16。
  for (int i = tid; i < k_half2; i += blockDim.x) sx[swz_abs(i)] = xv[i];
  __syncthreads();

  const int warp_id = (blockIdx.x * blockDim.x + tid) >> 5;
  const int lane = tid & 31;
  // 不能 early return: 下面归约用 __shfl_down_sync(全 warp 同步原语), 整 warp 提前
  // 退出会让同 warp 存活 lane 的 shuffle 变成非法访问(g292 实测 illegal access)。
  // 故界外 warp 只是不进循环, 照常走到归约后由写回处的 warp_id>=out_cols 拦住。
  const bool live = warp_id < n_rows;
  // 每行 uint4 数 = K/2 字节 / 16B = K/32 = k_half2/16(一个 idx = 1 个 uint4 = 32 个 k)。
  // 注意: 曾误写 k_half2/32(=K/64) → 每行只点前半 K, 输出差一倍(PoC 是 UV=K/32)。
  const int uv = k_half2 / 16;
  // 界外 warp 的基址一律夹到 0 行: shuffle 之后仍会被 out_cols 判据拦住, 但访存
  // 必须落在合法地址(夹到行 0 是合法行; 越界行号会打出非法地址)。
  const int wid_safe = live ? warp_id : 0;
  const uint4* const wrow = w + (size_t)wid_safe * uv;
  const int scale_cols = k_pad / GS;
  const half* const srow = s + (size_t)wid_safe * scale_cols;
  const half* const zrow = wzp ? wzp + (size_t)wid_safe * scale_cols : nullptr;

  float acc = 0.f;
  for (int base = 0; live && base < uv; base += 32) {
    const int idx = base + lane;
    if (idx >= uv) break;
    const uint4 v = __ldg(wrow + idx);
    const int g = scale_group<GS>(idx);
    const float sc = __half2float(__ldg(srow + g));
    const float z = zrow ? __half2float(__ldg(zrow + g)) : 8.0f;
    float part = 0.f;
#pragma unroll
    for (int wi = 0; wi < 4; wi++) {
      half2 h[4];
      dequant8((&v.x)[wi], 1024.0f + z, h);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        // k = 32*idx + 8*wi + 2q / +1 → half2 逻辑下标 p = 16*idx + 4*wi + q,
        // 存储下标 = swz_abs(p)(装载时已落槽, 见 kernel 头部)。
        const int p = idx * 16 + wi * 4 + q;
        // h 是本 wi 迭代 word 的 4 个 half2, 索引 h[q](q∈[0,4))。曾误写 h[wi*4+q]
        // → wi>=1 越界读未初始化寄存器 = 全输出 NaN(独立探针 min_probe_nan 揪出)。
        const float2 f = __half22float2(__hmul2(h[q], sx[swz_abs(p)]));
        part += f.x + f.y;
      }
    }
    acc += sc * part;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, acc, off);
    if (lane + off < 32) acc += other;
  }
  if (lane != 0 || warp_id >= out_cols) return;
  const float b = bias ? __half2float(__ldg(bias + warp_id)) : 0.f;
  out[warp_id] = __float2half_rn(acc + b);
}

// 通用版(M>=2, 投机解码 batch): smem staging + 每行一份 padding 副本 + 预取流水。
template <int M, int GS>
__global__ __launch_bounds__(kRows * 32) void ff_idecode_gemv_kernel(
    const half* __restrict__ x,   // [M, K_pad]
    const uint4* __restrict__ w,  // [N_p, K_pad/2] 自然布局
    const half* __restrict__ s,   // [N_p, K_pad/GS]
    const half* __restrict__ wzp,  // 可空 [N_p, K_pad/GS](null → 对称 zp=8)
    const half* __restrict__ bias,  // 可空 [size_n]
    half* __restrict__ out,         // [M, size_n](padded_n 区不写)
    int n_rows,    // padded_n(自然布局的行数)
    int k_pad,     // padded_k(= 权重行的 k 数)
    int out_cols,  // size_n(写出的列数, <= n_rows)
    int m) {
  extern __shared__ half sm_x[];  // 同上: half[] 声明(本环境 char[] 写入静默丢失)+ half2* 视图
  // 每行 uint4 数 = K/2 字节 / 16B = K/32 = k_half2/16(UV=16)。
  // 曾误取 32(=K/64) → 每行只点前半 K(与 m1 kernel 同款坑, 见彼处注释)。
  constexpr int UV = 16;
  const int k_half2 = k_pad / 2;     // 每行激活的 half2 数
  half2* const sx = reinterpret_cast<half2*>(sm_x);

  // 每行 stride = swz_abs 映射上界(与 host 侧 kSmemBytes 的 kRowStrideHalf2 一致)。
  const int stride = kRowStrideHalf2(k_half2);

  // 块协作装载 M 行激活(每行 K/2 个 half2, 落 swz_abs 槽)。
  const int tid = threadIdx.x;
  for (int i = tid; i < m * k_half2; i += blockDim.x) {
    const int row = i / k_half2;
    const int j = i - row * k_half2;
    sx[(size_t)row * stride + swz_abs(j)] = reinterpret_cast<const half2*>(x)[i];
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int row = blockIdx.x * kRows + warp;
  // 同 m1 kernel: 界外 warp 不 early return(__shfl_down_sync 需全 warp 齐访)。
  const bool live = row < n_rows;
  const int row_safe = live ? row : 0;

  const uint4* const wrow = w + (size_t)row_safe * (k_half2 / UV);
  const int scale_cols = k_pad / GS;
  const half* const srow = s + (size_t)row_safe * scale_cols;

  const half* const zrow = wzp ? wzp + (size_t)row_safe * scale_cols : nullptr;
  float acc[M];
#pragma unroll
  for (int mm = 0; mm < M; mm++) acc[mm] = 0.f;

  // 一次迭代 = 每 lane 一个 uint4 → 整 warp 恰好覆盖一个 32*k 段(GS 的整数因子)。
  const int uv = k_half2 / UV;  // 每行的 uint4 数
  // 预取流水: 权重流式读是延迟受限, 先下一轮再算本轮; scale/zp 一次取整轮 32 个 uint4
  // 用到的值(GS>=32 → 本轮恒同一组, 故只需各一个标量, 提到反量化之前)。
  uint4 v = (live && lane < uv) ? wrow[lane] : make_uint4(0u, 0u, 0u, 0u);
  for (int base = 0; live && base < uv; base += 32) {
    const int idx = base + lane;
    const bool active = idx < uv;
    const int next = idx + 32;
    const uint4 v_next = (next < uv) ? wrow[next] : v;
    // scale/zp 读必须界内: uv 非 32 倍数时(K=640→uv=20)末轮 lane idx≥uv,
    // 不 clamp 会越界读 srow(实测 Xid 13 Out Of Range, g292 17:03 复现)。
    const int gc = idx < uv ? idx : (uv - 1);
    const float sc = __half2float(__ldg(srow + scale_group<GS>(gc)));
    const float z = zrow ? __half2float(__ldg(zrow + scale_group<GS>(gc))) : 8.0f;
    if (active) {
      const float zp_off = 1024.0f + z;
      // h[wi][q]: 4 个 word 各解码一次(权重只读一遍), M 个激活复用 —— M>=2 结构优势。
      // 曾误用单 h[4] 反复覆盖(dequant 循环后只剩最后 word) + h[wi*4+q] 越界读未初始化
      // 寄存器 → 全输出 NaN(独立探针 min_probe_nan 揪出)。
      half2 h[4][4];
#pragma unroll
      for (int wi = 0; wi < 4; wi++) dequant8((&v.x)[wi], zp_off, h[wi]);
#pragma unroll
      for (int mm = 0; mm < M; mm++) {
        // 每行激活独立, 行偏移 mm*stride; 行内下标经 swz_abs() 落进 padding 后的 bank 槽。
        const half2* ak = sx + (size_t)mm * stride;
        float part = 0.f;
#pragma unroll
        for (int wi = 0; wi < 4; wi++) {
#pragma unroll
          for (int q = 0; q < 4; q++) {
            // k = 32*idx + 8*wi + 2q / +1 → half2 逻辑下标 p = 16*idx + 4*wi + q,
            // 存储下标 = swz_abs(p)(装载时已落槽, 每 16 个 half2 补 1 消 bank 冲突)。
            const int p = idx * 16 + wi * 4 + q;
            const float2 f =
                __half22float2(__hmul2(h[wi][q], ak[swz_abs(p)]));
            part += f.x + f.y;
          }
        }
        acc[mm] += sc * part;
      }
    }
    v = v_next;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int mm = 0; mm < M; mm++) {
      const float other = __shfl_down_sync(0xffffffffu, acc[mm], off);
      if (lane + off < 32) acc[mm] += other;
    }
  }
  // 只 lane 0 持完整和(上面 shfl_down 时高位 lane 不累加越界项)。
  if (lane != 0 || row >= out_cols) return;
  const float b = bias ? __half2float(__ldg(bias + row)) : 0.f;
#pragma unroll
  for (int mm = 0; mm < M; mm++)
    if (mm < m) out[(size_t)mm * out_cols + row] = __float2half_rn(acc[mm] + b);
}

// 自然布局 → 每 nibble 一字节(q [N_p, K_p] uint8)。无索引 gather 的精确版,
// 供 load 期算 c_n(替代 marlin 版两遍反量化)与 host 侧参考实现。
__global__ void unpack_natural_kernel(const uint8_t* __restrict__ w,
                                      uint8_t* __restrict__ q, int64_t total) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;  // total = N_p * K_p / 2 (byte 数)
  const uint8_t byte = w[i];
  const int64_t k2 = i * 2;
  q[k2] = byte & 0xFu;
  q[k2 + 1] = byte >> 4;
}

// smem 超 48KB 默认上限时提升单 kernel 的动态 smem 限额。Turing(sm75) opt-in 上限
// 是 **64KB**(cudaDevAttrMaxSharedMemoryPerBlockOptin=65536, 实测 M=3/K=10240=63.75KB
// 过、M=4=85KB 失败; 96KB 是 sm80+ 的值, 别用)。超 64KB 的 (M,K) 组合由 load 期
// _idec_shape_ok 守卫拦掉回退 marlin, 不靠这里。每实例化一次生效(static 防热路径
// 重复 driver 调用)。
template <typename Kernel>
static void set_smem(Kernel kernel, int bytes) {
  // 每次 launch 无条件设动态 smem 限额(小 kernel 的 driver 调用开销可忽略)。
  // 不再缓存 raised 值: sm75 上"同一 kernel 先按小值 set、再按更大值二次 set"会失败
  // (复现见 repro_seq: K=2560 M=2 后 K=10240 M=2 崩, 单独跑后者却好), 缓存会让第二次
  // set 被跳过、限额仍停在旧值 → invalid argument。无条件 set 按当前 bytes 恒一致。
  // 48KB 内为默认限额(set 冗余但无害); 超 48KB 需 opt-in(上限 64KB, 超者由 load 期
  // _idec_shape_ok 守卫拦掉回退 marlin)。
  cudaFuncSetAttribute((const void*)kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

// 按模板 M 分发(M 是运行期值, 编译期 2..8 实例化; M==1 走专化版; GS 同 64/128/256)。
template <int GS>
static void launch_by_m(const half* x, const uint4* w, const half* s,
                        const half* wzp, const half* bias, half* out, int n_rows,
                        int k_pad, int out_cols, int m, int smem,
                        const cudaStream_t stream) {
  const dim3 grid((unsigned)((n_rows + kRows - 1) / kRows));
  const int threads = kRows * 32;
  if (m == 1) {
    set_smem(&ff_idecode_gemv_m1_kernel<GS>, smem);
    ff_idecode_gemv_m1_kernel<GS><<<grid, threads, smem, stream>>>(
        x, w, s, wzp, bias, out, n_rows, k_pad, out_cols);
    return;
  }
#define LAUNCH_GEMV(MV)                                                   \
  set_smem(&ff_idecode_gemv_kernel<MV, GS>, smem);                        \
  ff_idecode_gemv_kernel<MV, GS><<<grid, threads, smem, stream>>>(        \
      x, w, s, wzp, bias, out, n_rows, k_pad, out_cols, m)
  switch (m) {
    case 2:
      LAUNCH_GEMV(2);
      break;
    case 3:
      LAUNCH_GEMV(3);
      break;
    case 4:
      LAUNCH_GEMV(4);
      break;
    case 5:
      LAUNCH_GEMV(5);
      break;
    case 6:
      LAUNCH_GEMV(6);
      break;
    case 7:
      LAUNCH_GEMV(7);
      break;
    default:
      LAUNCH_GEMV(8);
      break;
  }
#undef LAUNCH_GEMV
}

}  // namespace

torch::Tensor ff_idecode_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s,
                              torch::Tensor wzp, torch::Tensor bias, int64_t gs,
                              int64_t m, int64_t size_n) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda(),
              "ff_idecode_gemv: x/w/s 需在 CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                  w.scalar_type() == torch::kUInt8 &&
                  s.scalar_type() == torch::kFloat16,
              "ff_idecode_gemv: 需 x/s fp16, w uint8");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous(),
              "ff_idecode_gemv: x/w/s 需 contiguous");
  const int64_t K = x.size(-1);
  const int64_t n_rows = w.size(0);
  TORCH_CHECK(w.size(1) * 2 == K, "ff_idecode_gemv: 权重 K 与激活 K 不符");
  TORCH_CHECK(s.size(0) == n_rows && s.size(1) * gs == K,
              "ff_idecode_gemv: scale 形状应为 [N, K/gs]");
  TORCH_CHECK(m >= 1 && m <= 8, "ff_idecode_gemv: M∈[1,8], got ", m);
  TORCH_CHECK(x.numel() == m * K, "ff_idecode_gemv: x 元素数应为 M*K");
  TORCH_CHECK(K % 32 == 0 && n_rows % 8 == 0 && size_n <= n_rows,
              "ff_idecode_gemv: 需 K%32==0, N%8==0, size_n<=padded_n");
  TORCH_CHECK(gs == 64 || gs == 128 || gs == 256,
              "ff_idecode_gemv: GS ∈ {64,128,256}, got ", gs);

  auto out = torch::empty({m, size_n}, x.options());
  // smem 尺寸含末尾 padding 槽(见 kSmemBytes 注释); M==1 专化版同用。
  const int smem = kSmemBytes((int)m, (int)K);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  const uint4* wp = reinterpret_cast<const uint4*>(w.data_ptr());
  const half* sp = reinterpret_cast<const half*>(s.data_ptr());
  const half* zp =
      (wzp.defined() && wzp.numel() > 0)
          ? reinterpret_cast<const half*>(wzp.data_ptr())
          : nullptr;
  const half* bp =
      (bias.defined() && bias.numel() > 0)
          ? reinterpret_cast<const half*>(bias.data_ptr())
          : nullptr;
  half* op = reinterpret_cast<half*>(out.data_ptr());
  const int n_rows_i = (int)n_rows, K_i = (int)K, out_cols = (int)size_n,
          m_i = (int)m;

  if (zp != nullptr) {
    TORCH_CHECK(wzp.scalar_type() == torch::kFloat16 && wzp.is_contiguous() &&
                    wzp.size(0) == n_rows && wzp.size(1) == s.size(1),
                "ff_idecode_gemv: wzp 需 contiguous fp16 [N, K/gs]");
  }
  if (bp != nullptr) {
    TORCH_CHECK(bias.scalar_type() == torch::kFloat16 && bias.is_contiguous() &&
                    bias.numel() >= size_n,
                "ff_idecode_gemv: bias 需 contiguous fp16 [size_n]");
  }

#define LAUNCH_BY_GS(GSV)                                                \
  launch_by_m<GSV>(xp, wp, sp, zp, bp, op, n_rows_i, K_i, out_cols, m_i, \
                   smem, stream)
  if (gs == 64) {
    LAUNCH_BY_GS(64);
  } else if (gs == 128) {
    LAUNCH_BY_GS(128);
  } else {
    LAUNCH_BY_GS(256);
  }
#undef LAUNCH_BY_GS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor unpack_natural(torch::Tensor w, int64_t n_rows, int64_t k) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 &&
                  w.is_contiguous(),
              "unpack_natural: 需 contiguous CUDA uint8");
  TORCH_CHECK(w.size(0) == n_rows && w.size(1) * 2 == k,
              "unpack_natural: w 形状应为 [N, K/2]");
  auto q = torch::empty({n_rows, k}, w.options());
  const int64_t total = n_rows * k / 2;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  constexpr int threads = 256;
  const int blocks = (int)((total + threads - 1) / threads);
  unpack_natural_kernel<<<blocks, threads, 0, stream>>>(
      w.data_ptr<uint8_t>(), q.data_ptr<uint8_t>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("ff_idecode_gemv", &ff_idecode_gemv,
        "firefly idecode: M<=8 int4 GEMV (自然布局, 对称 zp=8 / 非对称 fp16 zp)");
  m.def("unpack_natural", &unpack_natural,
        "firefly idecode: 自然布局 [N,K/2] → 每 nibble 一字节 [N,K]");
}
