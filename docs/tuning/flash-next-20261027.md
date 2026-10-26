# Flash-Next 调优基线（2026-10-27，8×T10 / 262K）

> 适用：`Qwen3.8-Flash-Next-W4A16-AutoRound`，TP8+EP8，镜像 `v0.1.7-beta2-v13-opt9`。
> 同尺：直测脚本（随机前缀绕缓存、ignore_eos、流式 TTFT），5 次中位。面板尺另见 run-book。

## 1 已验证、建议默认的组合

| 项 | 取值 | 相对 4096+flashqla 基线 |
|---|---|---:|
| `--max-num-batched-tokens` | **8192**（16384 会 CUDA OOM） | 8K −1.1%，32K −4.1% |
| `--gdn-prefill-backend` | **triton**（配合门控包 `gdn_tc`） | 叠加后 8K −10.9%，32K −13.7% |
| `vllm_sm75` 门控包 | `VLLM_SM75_OPT=auto`（gemv_m1 + gdn_tc） | 解码 +1.4% |
| `--kv-cache-dtype` | fp8_e4m3 | — |

实测（opt9，`:40001`，262K，批 8192，triton）：8K TTFT **2019.1 ms**、32K **8643.5 ms**；
解码 8K 上下文 / 256 输出 **59.6–61.6 tok/s**。

## 2 分饼（32K 预填 trace，8.19 s GPU 忙 / 9.37 s 墙钟）

| 类别 | 占比 | 判读 |
|---|---:|---|
| NCCL AllReduce（490 次，中位 6.3 ms） | 36% | 73.5 MB/次 ÷ 6.3 ms = 11.7 GB/s，已达 PCIe 13.17 的 89% |
| 注意力（BatchPrefill + SinglePrefill） | 16% | — |
| cuBLAS GEMM | 17% | 接近算力上限 |
| HC（combine_norm + gate_mix） | 12% | 隔离实测已到带宽 87–89%，**无空间** |
| MoE（Marlin + fused） | 6% | — |
| GPU 空闲 | 12% | 开场 325 ms 一次性；另有 QSA 选页 plan ~10×17 ms |

## 2b 解码分饼（8K 上下文，32 token，trace；每步约 14.8 ms）

| 内核 | ms/步 | 说明 |
|---|---:|---|
| `_row_gemv`（HC 混流 GEMV，188 次/步） | 3.84 | 单次约 20 µs |
| `_gemv_kernel`（门控包 gemv_m1，140 次/步） | 2.42 | 单次约 17 µs |
| Marlin MoE（93 次/步） | 1.52 | 小批延迟型 |
| PCIe-IPC all-reduce（93 次/步） | 1.27 | 5 KB 小消息，约 13.6 µs/次 |
| cuBLAS gemvx（93 次/步） | 0.89 | 未进白名单的形状 |
| topk gating / QSA attn / HC combine / GDN / MoE align+sum / act_mul | 2.4 | 各 0.2–0.6 |
| 其它 | ~2.4 | — |

结论：解码是"~800 个小内核的延迟链"，GEMV 已接近带宽/延迟下限；再往上要么减少权重字节（dense 量化），要么改写集合通信格式。

## 3 已判负 / 不可行（有数据，勿重复）

- **`--max-num-batched-tokens=16384`**：首个 8K 请求即 CUDA OOM（Triton 分配失败），引擎退出。
- **AR 重叠（`ar_overlap`）**：同镜像 A/B，开 2306.6/10081.5 ms vs 关 2266.7/10021.7 ms，无收益。
- **PCIe-IPC 预填（`ar_prefill`）**：同字节比 NCCL 慢 15–22%。
- **HC 内核重写**：`combine_norm` 0.37 ms（下限 0.33）、`gate_mix` 1.02 ms（下限 0.89），已在带宽地板上。
- **NCCL fp8 集合通信**：torch 对 fp8 只在 sm90+ 走真 fp8 路径，SM75 不支持；要 8-bit 集合通信只能自写内核。
- **NCCL 缓冲/通道调参**（隔离微基准，42 MB all-reduce）：基线 6.426 ms / 11.42 GB/s；`NCCL_BUFFSIZE=16M` 6.473、`MIN_NCHANNELS=8` 6.636、`MAX_NCHANNELS=16` 6.420 —— 全部在噪声内，PCIe ring 已到顶。
- **MTP（k=3）**：接受率 53.6%、64K 上下文下 61–74 tok/s，但 262K + MTP 需再腾约 1.2 GiB/全卡，16 GB 卡放不下；已封存。

## 4 约束与后续方向

- 262K KV（2.25 GiB 池）下每卡只剩 **8.5 MiB** 空闲：任何更大的 workspace / 批大小 / 图捕获都会 OOM。
- 剩余大杠杆只有两条，都需额外显存或自写集合通信：
  1. **8-bit 集合通信**（量化 → reduce-scatter → all-gather），预填上限约 9–15%；需先做量化精度 + 半字节传输的有界验证。
  2. **dense 权重再量化**（当前 dense 为 fp16，仅路由专家 W4A16）：可同时降解码字节与腾显存，但需新权重与精度对拍。
