# 8-bit 集合通信可行性备忘（Flash-Next 预填 AR）

日期：2026-10-27　状态：**未立项，先留判断依据**　范围：8×T10、TP8、PCIe Gen3、262K。

## 1 为什么盯上这条

32K 预填分饼里，NCCL AllReduce 占 **36%**（490 次、中位 6.3 ms/次、73.5 MB 有效字节/次）。
隔离微基准确认：42 MB all-reduce 6.426 ms / **11.42 GB/s**，`NCCL_BUFFSIZE` 16M、`MIN_NCHANNELS=8`、
`MAX_NCHANNELS=16` 全部在噪声内（6.42–6.64 ms）⇒ **PCIe ring 已到顶，参数调不动**。
字节减半是唯一还能打的方向。

## 2 现成 NCCL 为什么不行

- `nccl.h` 有 `ncclFloat8e4m3 = 10`，但 PyTorch 的集合通信对 fp8 有 sm90 门：
  `torch/distributed/distributed_c10d.py:3142` —— “FP8 is supported by NCCL on sm90+,
  use workaround for older GPUs”，SM75 上会 `view(torch.uint8)` 直接搬运（不做求和）。
- 结论：**拿 stock NCCL 在 T10 上做 fp8 sum 不可用**，不是配置问题。

## 3 两条实现路线

| | A 重编 NCCL | B 自写集合通信（走仓内 PCIe-IPC） |
|---|---|---|
| 改动 | NCCL 源码加 sm75 软件 fp8 转换/规约 | 新增 CUDA 内核：量化 → reduce-scatter → all-gather |
| 与 torch 关系 | 必须与 torch 自带 NCCL ABI/版本对齐，替换 `.so` | 不动 NCCL，调用方改成 IPC 通道 |
| 工作量 | 大（构建链 + 兼容回归） | 中（约 500–800 行 + 单测） |
| 风险 | 高（全局替换通信库） | 中（只影响开启该项的行） |
| 判断 | **不推荐** | 有界可行，先做验证 |

## 4 精度账（决定要不要做）

- fp8 e4m3：3 位尾数，单元素相对误差最坏 ~12.5%、RMS ~3.6%；per-token/per-channel 缩放不改变尾数精度。
- reduce-scatter/all-gather 各量化一次 ⇒ 误差约 2 次量化，不相乘（不像 ring 逐步量化会累积 7 跳）。
- 对 residual-stream 激活，3.6% RMS 是 fp16 的 ~70 倍。**大概率过不了我们现有的 L3 门**（教师强制 logprob 噪声带 0.002–0.005 nats）。
- 若只对 **MoE 的 48 次 reduce** 用 8-bit（其余 48 次 dense 保持 fp16），影响面减半，但收益也从 ~13% 降到 ~6–7%。
- 更保守的替代：**12-bit 打包**（fp16 截掉 4 位尾数）⇒ 字节 −25%、误差 ~0.4%（仍是 fp16 的 8 倍），预填上限约 9%；实现同样需要自写内核。

## 5 先做什么（若立项）

1. **精度门**（1 小时内）：在引擎里加一个 env 闸，把 row-parallel 输出做“量化→反量化”（不改通信），跑 L3 教师强制 logprob，看 Δ 是否落在噪声带；MoE-only 与全部两种口径各测一次。
2. **传输门**（1 小时内）：隔离微基准，现有 IPC 内核搬 **半个张量**（21 MB）的耗时 + 2 个量化/反量化内核耗时；若 `t_ipc(21MB) + t_quant < 6.3 ms × 0.8` 才继续。
3. 两道门都过，再写 reduce-scatter/all-gather 内核；任何一门不过即结案。

## 6 当前建议

**暂不立项。** 理由：① 精度门大概率不过（3.6% RMS 对 residual 流太大）；② 12-bit 路线收益略低但实现同样重；③ 262K 下每卡只剩 8.5 MiB，新内核/缓冲的显存也要从 KV 里挤。
若用户接受“MoE-only 8-bit + L3 放宽”，可作为下一个窗口的第一项实验。
