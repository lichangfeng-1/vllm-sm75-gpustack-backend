# Turing 0.32 兼容件层与容量档（2026-11-16）

> 口径：8×Tesla T10 16 GiB、SMT Off、面板同尺（8192 / 16384 / 32768，output 1024，并发 1，temp 1，topP 0.95）。
> 生产镜像 `…-final-v1` 未改动；本文的兼容层是**可选薄层**，全部开关默认关。**GPU 实机 A/B 尚未跑**（见 §3），默认关可以叠，逐件开启前先过门禁。

## 1 兼容件层是什么

把 vllm-Turing 0.32 包里的兼容/加速件按 0.30 API 适配后并入一层薄镜像：新文件为主，配 env 默认关（fail-closed）。不开任何开关时，行为应与父镜像 `…-final-v1` 一致。

| 项 | 值 |
|---|---|
| 镜像 | `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1-compat1`（manifest list `sha256:9cb13d85…`，**已构建、未注册、未切行**） |
| 父镜像（未动） | `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1`（`7267ff1c3db0`） |
| 配方 | [`gpustack/Dockerfile.compat-v1-20261116`](../../gpustack/Dockerfile.compat-v1-20261116) |
| 构建 | `docker build -f gpustack/Dockerfile.compat-v1-20261116 -t <tag> .` |
| 镜像级对拍 | `dist-packages/vllm` 共 5140 件（排除 `__pycache__`）：新增 18、修改 18、删除 0；其余 5104 件与父镜像逐字节相同 |
| 构建期验收 | [`docker/helpers/verify_compat_switches.py`](../../docker/helpers/verify_compat_switches.py)：开关默认关、EXL3 默认未注册、UQ dtype fail-closed、11 个新模块可导入，构建日志 `COMPAT-VERIFY: ALL PASS` |
| L1 静态测试 | `tests/test_sm75_compat_switches.py` 8/8、`tests/test_build_contract.py` 14/14、`tests/test_sm75_gate.py` 12/12（CPU） |

**夹带处置**：交付仓比 final-v1 领先 3 件 loader 改动（rank 分片加载 ＋ EP 过滤放行 `.qweight/.qzeros`）。本层从父镜像取原样快照回填，**不夹带**；是否单独吸收另议（见 §4）。

## 2 开关表（全部默认关）

| 件 | 开关 | 默认 | 开 = 什么 | 关 = final-v1？ |
|---|---|---|---|---|
| firefly fp8 AR | `VLLM_FIREFLY_AR` | **关**（未设/`auto` 均关；`1`/`on`/`fp8` 才开） | fp8 SHM/P2P allreduce（≥1 MiB 消息） | 是（行 26 显式 `=0`，行为不变；**唯一语义收紧**：未设该键时不再随 `VLLM_FIREFLY` 自动开） |
| AR 分块流水线 | `VLLM_FIREFLY_AR_PIPE` | 关 | 2-GPU 全双工 SHM 分块 overlap | 是 |
| NVFP4 firefly 快填 | `VLLM_FIREFLY_NVFP4` | 关 | 大 M 现反 int8 走 IMMA；decode 仍 Marlin | 是（不建快照、零额外显存） |
| NVFP4 IDEC decode | `VLLM_FIREFLY_IDEC_NVFP4` | 关 | NVFP4 decode 也走 int8（需上项先开） | 是 |
| NVFP4 Skillet decode | `VLLM_FIREFLY_SKILLET` / `_SPK`（默认 8） | 关 | m16n8k8 fp16 TC decode ＋ 独享快照 | 是 |
| int4 IDEC GEMV | `VLLM_FIREFLY_IDEC` / `_MAX_M`（默认 8） | 关 | W4A16 decode M≤8 自研 GEMV；**改 load 期权重布局** | 是（默认不改布局） |
| 0.32 生态总门别名 | `VLLM_FIREFLY_DIRECT` | 跟随 `VLLM_FIREFLY` | 0.32 包代码用此名 | 与现状等价 |
| EXL3 在线解码 | `VLLM_SM75_EXL3` | 关 | 注册 EXL3 量化法（仅 EXL3 模型需要） | 是（关时解析同 final-v1 报错） |
| TurboQuant SM75 prefill | `VLLM_FIREFLY_TQ` | 关（0.32 包原默认开，已按令翻转） | TQ KV 在 SM75 走 FlashInfer ragged | 是（TQ 不在生产路径） |
| UltraQuant 4bit KV | `VLLM_SM75_ULTRAQUANT` | 关 | 允许 `--kv-cache-dtype ultraquant_4bit` | 是（关时该 dtype 在 arg_utils 显式报错＝fail-closed） |
| DFlash2 0.32 改进 | `VLLM_SM75_DFLASH2_OPT` | 关 | 分组卷积 Triton 核 ＋ tp≥2 context-FC 切分 ＋ draft 量化传递 | 是（生产未配投机） |

NVFP4 三项是嵌套门：`VLLM_FIREFLY_NVFP4` 为总门，`IDEC_NVFP4` 依赖总门的快照，Skillet 自带独立快照门。全部关时不建任何快照、不改任何权重布局。

## 3 GPU 实机门禁（未完成，默认关不得开）

| # | 件 | 判据 |
|---|---|---|
| G1 | 全关等价（总门） | 同参数分别跑 final-v1 与 compat1（不设任何兼容开关）：面板同尺 decode 中位差 ≤1%、prefill/TTFT ≤3%、固定题（temp=0）逐字一致、显存差 ≤1% |
| G2 | NVFP4 firefly / Skillet / IDEC | 27B NVFP4 实验行逐项开；日志出现对应快照行；同题文本一致或 L2 相对误差 ≤1e-3；记录 decode 提速 |
| G3 | int4 IDEC | Flash-Next W4A16 实验行开 `VLLM_FIREFLY_IDEC=1`；日志 `firefly_idecode CUDA kernel loaded`；文本一致；记录提速 |
| G4 | TurboQuant SM75 | 起 `--kv-cache-dtype turboquant_k8v4`（或 4bit_nc）＋`VLLM_FIREFLY_TQ=1`：无 FA2 assert、预填不掉速/OOM |
| G5 | UltraQuant 4bit KV | 开 `VLLM_SM75_ULTRAQUANT=1` 后该 dtype 可起服；关时同参数必须报错（负控） |
| G6 | EXL3 | 需 EXL3 权重（当前无）；只能停在「注册与导入已验证」 |
| G7 | firefly AR 默认收紧 | 行 26 现役日志确认无 `FIREFLY_AR` 初始化 |

在此之前，compat1 只能停在「已构建」，不得注册到生产行；行 26／最终验收口径不变。

## 4 容量档：block 400 ＋ mamba fp16（纯配置，无代码）

**机制**（0.30 `vllm/platforms/interface.py` 实读）：开 prefix caching 时是 align 模式，`mamba_block_size` 被强制等于 `block_size`（独立 `--mamba-block-size` 无效）；mamba 页会被 padding 到与 attention 页相等：

```
padding% = 100 × (block × P − M_mamba) / M_mamba
```

**block 880 ＋ mamba fp16**：padding **120.55%**、`kv_cache_size_tokens=338,994`，与基线 339,110 基本相同 ⇒ 无增益，**勿用 880**。

**block 400 ＋ mamba fp16**（实验行实测，其余同生产参数）：

| 项 | 读数 |
|---|---|
| mamba padding | 0.25%（日志 `Padding mamba page size by 0.25%`；`block_size=400, mamba_block_size=400, mamba_ssm_cache_dtype=float16`） |
| KV tokens | **344,137**（+1.48% vs 339,110） |
| num_gpu_blocks / 最大并发 | 873 / 1.3128（+1.5%） |
| decode（8K/16K/32K） | 63.47 / 63.25 / 63.31（对照 block784/fp32：63.13 / 63.06 / 62.75，噪声内） |
| TTFT（8K/16K/32K） | 2674.4 / 4232.6 / 9367.9 ms（对照 2005 / 4266.2 / 9303 ms；8K 单次偏高，16K/32K 持平，**8K 需复测**） |

用法（模型行后端参数）：

```
--block-size 400
--mamba-ssm-cache-dtype float16
```

边界：固定题 5/5 正常输出；与 block880 版文本 1/5 逐字同（fp16 SSM 数值差异），未做质量判决。

## 5 仅登记、不吸收的项

| 项 | 结论 | 理由 |
|---|---|---|
| `c5b685f` EP 过滤补 `.qweight/.qzeros` | 标记「加载优化」，低优先，未烘入镜像 | 纯过滤单 rank 读量 164.62→115.02 GiB（−30%），但纯过滤耗时未单独实测，decode 无影响 |
| `ce4f4e7` PLE 按 rank 分片加载 | 不吸收 | 2026-10-26 已判死（读量降到 0.19× 只快 22–33%，且减的是每 rank 拷贝）；分片目录已删 |
| Config B（`enable_mamba_fine_grained_prefix_cache` ＋ `--prefix-match-unit 80`） | 仅文档 | 无 MTP 时为 no-op；MTP 下共享前缀才有行为收益，且额外收益未获铁证 |
| MTP（投机解码） | 暂不吸收 | MTP2 面板同尺 decode 85.33 / 80.35 / 79.91（均值 81.86，≈×1.27–1.35 基线），接受率 45.1%、平均接受长度 1.90；但 262K 撞显存墙：KV 2.25 GiB 时 250K 预填 OOM 158 MiB，KV 1.5 GiB 时容量门拒（≈195,200 tokens）。**本机 MTP 实用上下文 ≤ ~195K**，与 262K 互斥 |

MTP 的完整材料（配置、崩溃记录、复现要点）留在内部材料档，不进公开仓；复现路线：先测草稿每卡占用（分片 vs 复制），再评估 Swift 权重省显存或 CPU KV offload。