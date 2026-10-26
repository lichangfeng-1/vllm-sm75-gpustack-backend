# 下一步两个大方向的设计与 go/no-go 判据（2026-10-27）

前置事实（均有读数）：
- 预填 32K：AR 36%（11.4 GB/s，PCIe 93%）、注意力 88%、HC 87–89%、GEMM 接近 cuBLAS 上限；
- 解码每步 14.8 ms：GEMV 48%（`_row_gemv` 93–95% HBM、gemv_m1 已转正）、Marlin/PCIe-IPC 各约 1.3–1.5 ms；
- 无精度损失是硬约束 → 量化、fp8 集合通信、MTP 全部出局。

## 方向 A：SBO（单批计算/通信重叠，借 SGLang #9660 思路）

机制：把一个 prefill batch 切成两半 A/B，两条 CUDA 流；A 做第 L 层的 GEMM/注意力时，B 在同一层或相邻层的
all-reduce 在 NCCL 流上跑，反之亦然。理论上可把 36% 的 AR 藏进计算，预填上限约 **−10%**。

**已做 go/no-go（2026-10-27，8 卡隔离容器）**：
1. 42 MB all-reduce vs 单个约 1 ms GEMM：并发 7.30 ms（和 7.51）→ 只藏住 20%；
2. 但真实比例是"6.4 ms 的 AR 对约 30 ms 的层计算窗"：**32×1 ms GEMM 循环 + 一次 AR 并发 31.93 ms
   （GEMM 单独 30.18、AR 6.38、和 36.57）→ 藏住 73%**。
3. 结论：**SBO 在本硬件成立**；改动面＝model_runner 把 batch 切两半、两条 CUDA 流、QSA/GDN 的流安全与 KV 写入顺序。
   上限估计：AR 占预填 36%，藏 73% ⇒ 预填最多约 **−25%**（打折后 10–15% 量级）。

## 方向 B：解码的 out_proj + all-reduce + residual 融合（借 vLLM Kimi K3 / Flint 思路）

机制：把行并行 GEMV 的 epilogue 与后续残差相加合并，AR 结果直接写回目标缓冲；目标是把每步 ~800 个内核
压 20–30%。Kimi K3 在 NVLink 上用"直接写 peer 共享缓冲"实现；我们只有 PCIe，需要用 PCIe-IPC 的 push 语义。
可行性未知点：PCIe 写 peer 的延迟（单次 5 KB 消息约 13.6 µs）是否低于独立 AR 内核 + 残差内核的开销。
微基准：2 卡/8 卡上比较 `gemv+AR+add` 三内核 vs `gemv(epilogue 内 push)+add` 两内核。

## 顺序

先做 A 的并发微基准（1 小时内可出结论）；A 不成立再做 B。两者都需独立镜像层与 L2/L3 门禁。
