# 外部最新技术对照（vLLM / SGLang / DeepSeek，2026-10）与我们的取舍

约束：8×T10（Turing CC7.5）、TP8+EP8、单流、**不能降低精度**、262K 上下文。
目的：把 2026-09/10 各家公开的推理优化逐条对照，只保留对我们成立的方向。

## 1 逐条对照

| 技术 | 出处（公开） | 机制 | 我们的判断 |
|---|---|---|---|
| 解码 megakernel / 逐层融合（23→13 kernels，130→69 µs/层） | NVIDIA×vLLM Kimi K3（2026-10-16） | CuTe DSL 把一层内的 QKV/注意力/输出投影/AR/残差融合 | **可借鉴但需自写内核**；我们解码每步 ~800 内核、14.8 ms，融合 out_proj+AR+residual 与 norm+RoPE+KV 写入是明确方向 |
| 参考精度一致的逐算子替换（Flint 40B） | vLLM（2026-10-21） | 每个新内核与参考实现同一舍入点，GSM8K/GPQA 对齐 | **方法论可用**：与我们 L2/L3 门禁一致；单融合 −8% |
| Prefill 用解码 CUDA graph replay（−30~40% 预填） | DeepSeek-V4.1-Flash / vLLM（2026-10-07） | 复用已捕获图跑预填，削内核启动开销 | **暂不可行**：262K 下每卡仅剩 8.5 MiB，扩图池会 OOM |
| SBO 单批计算/通信重叠 | SGLang #9660、LMSYS DeepSeek-V4-Pro（2026-08） | 一个 batch 切两个 microbatch，计算与集合通信流水 | **最值得做的预填方向**：AR 占 36%，理论上限 ~10%；改动大、需专门立项 |
| TBO 两批重叠 | DeepSeek（2026） | decode 两 batch 互补 | 单流无第二 batch，不适用 |
| DeepSelect top-k 内核 | DeepSeek（2026-09） | 针对 DSA top-512 的高性能 top-k | 我们 persistent_topk 解码 0.086 ms/次、32K 预填约 86 ms；上限小，低优先 |
| MegaAttention（RoPE+稀疏注意力+反 RoPE+FP8 cast 单内核） | DeepSeek/FlashMLA（2026-09/10） | 注意力前后处理融合 | 我们 QSA 已是多段内核，融合最多省几十 ms，低优先 |
| DP Attention（注意力 DP、MoE TP） | SGLang/DeepSeek | 避免 MLA 的 KV 复制 | 我们 QSA/GQA 的 KV 已按 TP 分片，不适用 |
| 零开销/异步调度 | SGLang、vLLM v1 | CPU 调度与 GPU 重叠 | 已具备（SM75Scheduler 继承 AsyncScheduler，实测确认） |
| 前缀复用（RadixAttention） | SGLang | 前缀缓存 | vLLM 已有 prefix caching；测速用随机前缀绕开 |
| NVFP4/FP8/INT4 量化、MTP、DFlash | 各家 | 降精度或投机 | **按"不能降低精度"作废**；MTP 另按用户要求暂停 |

## 2 本轮同时得到的负结果（有数据）

- **QSA 切片放大**（1024→4096 行，pages 缓冲 2.1→8.4 MB，workspace 为 0）：8K 2019.1→2000.7 ms（−0.9%）、
  32K 8643.5→8646.1 ms（+0.0%）⇒ 不是预填空隙的来源，判负。
- **QSA 选页计时探针**：`runner.run/_prepare` 单次全部 <2 ms（阈值 2 ms 无一条日志）⇒ 32K 热态
  的 48×18 ms 空隙不在选页路径内，来源仍未定位（下一步只能在调用者/索引器侧做更细埋点）。
- 8K 带栈 trace 里 186 ms 的大空隙＝首请求懒加载（栈里是 importlib/pathlib），不是计算。

## 3 建议的下一步（按收益/代价）

1. **解码内核融合**（不降精度、可增量做）：先把 out_proj+all-reduce+residual 合成一个内核/一次发射，
   在解码上做 A/B；目标把每步 800 内核压 20–30%。
2. **SBO 预填流水**：batch 切两半、两条流，让一半的 AR 与另一半的 GEMM/注意力重叠；先出 2 页设计再动码。
3. 预填空隙：若还要追，在 `try_grouped_attention` 与索引器调用点加埋点（当前证据已排除选页内核）。
