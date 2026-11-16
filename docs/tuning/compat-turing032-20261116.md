# Turing 0.32 兼容件层与容量档（2026-11-16）

> 口径：8×Tesla T10 16 GiB、SMT Off、面板同尺（8192 / 16384 / 32768，output 1024，并发 1，temp 1，topP 0.95）。
> 生产镜像 `…-final-v1` 未改动；本文的兼容层是**可选薄层**，功能开关默认关（`VLLM_FIREFLY_DIRECT` 是 0.32 生态别名，跟随现有 `VLLM_FIREFLY`，与 final-v1 等价）。**GPU 实机 A/B 尚未跑**（见 §3），默认关可以叠，逐件开启前先过门禁。

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

**夹带处置**：交付仓比 final-v1 领先 3 件 loader 改动（rank 分片加载 ＋ EP 过滤放行 `.qweight/.qzeros`）。本层从父镜像取原样快照回填，**不夹带**；是否单独吸收见 §5。

**两件 loader 改动必须分辨，不可混为一谈**：`ce4f4e7`（PLE 按 rank 分片加载）**2026-10-26 已判死**——127.61 s、读量降到 0.19× 却只快 22–33%，多卡 page cache 去重 ⇒ 减的是每 rank 拷贝，分片目录已删；`c5b685f`（EP 过滤补 `.qweight/.qzeros`）是**另一件**，无损、单 rank 读量 −30.1%，**仓内源码已含此修复**（`vllm/model_executor/model_loader/ep_weight_filter.py:84-85`）但**未烘入现役镜像**，优先级已上调（见 §5）。

镜像构建自审查前的兼容件树；本轮审查后的配方改动（loader 回填逻辑、EXL3 失败即抛、校验加固）只影响重建行为，镜像本身未重建。需要与当前树逐一对应时，按配方重建即可。

## 1a 0.32 整栈对照（2026-11-16 面板正式验收）

**结论：整栈不迁，只吸收兼容件（默认关）。** 把 vllm-Turing 0.32 的全配非 MTP 清单（参照实现 compose 复刻，MTP/KV offload 除外）跑面板正式验收路径：

| Prompt | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 8192 | 3378.70 | **59.71** |
| 16384 | 3436.31 | **59.62** |
| 32768 | 3003.35 | **59.25** |

汇总：平均输出 **59.47**、P50 59.51、P90 59.64、P95 59.66、平均预填充 3248.05。

对本栈的差（**本轮算术复核**）：对 2026-10-27 Intel 档 63.30 为 **−6.05%**；对 2026-11-16 Swift 行 48 的 66.68 为 **−10.81%**。8K TTFT 2.42 s 对本栈 ~2.0 s、32K 10.9 s 对 ~9.37 s 亦更差。
口径并列不择一：审计报告原文记「净负 **4–10%**」、上机状态档面板口径记「**6–10%**」、本轮实算 **−6.05%~−10.81%**——三者对象与分母不同，引用时须带全。

⇒ **0.32 的价值在兼容件（NVFP4/Skillet/IDEC、EXL3、TurboQuant、UltraQuant、DFlash2 改进），不在整栈迁移**；这些件已按 0.30 API 适配并入本层、全部 env 默认关。

## 2 开关表（全部默认关）

| 件 | 开关 | 默认 | 开 = 什么 | 关 = final-v1？ |
|---|---|---|---|---|
| firefly fp8 AR | `VLLM_FIREFLY_AR` | **关**（未设/`auto` 均关；`1`/`on`/`fp8` 才开） | fp8 SHM/P2P allreduce（≥1 MiB 消息） | 是（行 26 显式 `=0`，行为不变；**唯一语义收紧**：未设或显式 `auto` 都不再视为开，只有 `1`/`on`/`fp8` 才开；仓内 8 个 27B 预设与 `docker/run.sh` 默认值已同步改为显式 `0`） |
| AR 分块流水线 | `VLLM_FIREFLY_AR_PIPE` | 关 | 2-GPU 全双工 SHM 分块 overlap | 是 |
| NVFP4 firefly 快填 | `VLLM_FIREFLY_NVFP4` | 关 | 大 M 现反 int8 走 IMMA；decode 仍 Marlin | 是（不建快照、零额外显存） |
| NVFP4 IDEC decode | `VLLM_FIREFLY_IDEC_NVFP4` | 关 | NVFP4 decode 也走 int8（需上项先开） | 是 |
| NVFP4 Skillet decode | `VLLM_FIREFLY_SKILLET` / `_SPK`（默认 8） | 关 | m16n8k8 fp16 TC decode ＋ 独享快照 | 是 |
| int4 IDEC GEMV | `VLLM_FIREFLY_IDEC` / `_MAX_M`（默认 8） | 关 | W4A16 decode M≤8 自研 GEMV；**改 load 期权重布局** | 是（默认不改布局） |
| 0.32 生态总门别名 | `VLLM_FIREFLY_DIRECT` | 跟随 `VLLM_FIREFLY` | 0.32 包代码用此名 | 与现状等价 |
| EXL3 在线解码 | `VLLM_SM75_EXL3` | 关 | 注册 EXL3 量化法（仅 EXL3 模型需要） | 是（关时解析同 final-v1 报错） |
| TurboQuant SM75 prefill | `VLLM_FIREFLY_TQ` | 关（0.32 包原默认开，已改为默认关） | TQ KV 在 SM75 走 FlashInfer ragged | 是（TQ 不在生产路径） |
| UltraQuant 4bit KV | `VLLM_SM75_ULTRAQUANT` | 关 | 允许 `--kv-cache-dtype ultraquant_4bit` | 是（关时该 dtype 在 arg_utils 显式报错＝fail-closed） |
| DFlash2 0.32 改进 | `VLLM_SM75_DFLASH2_OPT` | 关 | 分组卷积 Triton 核 ＋ tp≥2 context-FC 切分 ＋ draft 量化传递 | 是（生产未配投机） |

NVFP4 三项是嵌套门：`VLLM_FIREFLY_NVFP4` 为总门，`IDEC_NVFP4` 依赖总门的快照，Skillet 自带独立快照门。全部关时不建任何快照、不改任何权重布局。

`VLLM_FIREFLY_AR` 的生效口径：`1`/`on`/`fp8` 算开，**未设与显式 `auto` 都按关**。仓内 27B 预设（`ultra/source/profiles/qwen38-27b-*.json`，8 件）与 `docker/run.sh` 的默认值已同步改为显式 `0`，避免旧 `auto` 语义静默变化。

## 3 GPU 实机门禁（未完成，默认关不得开）

| # | 件 | 判据 |
|---|---|---|
| G1 | 全关等价（总门） | 同参数分别跑 final-v1 与 compat1（不设任何兼容开关）：面板同尺 decode 中位差 ≤1%、prefill/TTFT ≤3%、固定题（temp=0）逐字一致、显存差 ≤1%；另比对启动日志的 block/mamba 页对齐与 `Padding mamba page size` 行一致 |
| G2 | NVFP4 firefly / Skillet / IDEC | 27B NVFP4 实验行逐项开；日志出现对应快照行；同题文本一致或 L2 相对误差 ≤1e-3；记录 decode 提速 |
| G3 | int4 IDEC | Flash-Next W4A16 实验行开 `VLLM_FIREFLY_IDEC=1`；日志 `firefly_idecode CUDA kernel loaded`；文本一致；记录提速 |
| G4 | TurboQuant SM75 | 起 `--kv-cache-dtype turboquant_k8v4`（或 4bit_nc）＋`VLLM_FIREFLY_TQ=1`：无 FA2 assert、预填不掉速/OOM |
| G5 | UltraQuant 4bit KV | 开 `VLLM_SM75_ULTRAQUANT=1` 后该 dtype 可起服；关时同参数必须报错（负控） |
| G6 | EXL3 | 需 EXL3 权重（当前无）；只能停在「注册与导入已验证」 |
| G7 | firefly AR 默认收紧 | 行 26 现役日志确认无 `FIREFLY_AR` 初始化 |

在此之前，compat1 只能停在「已构建」，不得注册到生产行；行 26／最终验收口径不变。

## 4 容量档：block 400 ＋ mamba fp16（纯配置，无代码）

**机制**（`vllm/platforms/interface.py` 的混合模型块对齐）：混合模型先保证 attention 页 ≥ mamba 页；`mamba_cache_mode=align` 时再把 `mamba_block_size` 强制等于 `block_size`（独立 `--mamba-block-size` 会被覆盖）。随后 mamba 页会被 padding 到与 attention 页相等：

```
padding% = 100 × (block × P − M_mamba) / M_mamba
```

口径说明：下面的 0.25% / 344,137 是在 **final-v1（0.30 基座）** 上、行 26 参数＋`--block-size 400 --mamba-ssm-cache-dtype float16` 实测；启动日志应出现 `Padding mamba page size by 0.25%` 与 `block_size=400, mamba_block_size=400`。compat1 覆盖了 `cache.py` 与 `platforms/interface.py`（UQ/TQ 支持），全关时这两件的 mamba 页行为是否与 final-v1 完全一致由 G1 门验证；**G1 通过前，容量档继续在 final-v1 上使用**。

**在 Swift 行 48 上不值得优先排窗口（2026-11-16 现读）**：行 48 声明 `--block-size=16` 被分配器**覆写成 784**（`interface.py:933`），mamba padding 已 **0.13%**（`:957`）——近零浪费。上面那 **+1.48%** 是在**有浪费的基线**（padding 可观）上拿到的，Swift 现态没有那块浪费。且**容量 ≈ KV 预算 ÷ 每 token 字节，与 block 基本无关**；这一族**只动容量不动速度**（Intel 上实测 decode 63.25–63.47，与基线同档无变化）。机制与显存口径见 [Swift 现役调优页](swift15-prod-20261116.md) §4／§5.C2。

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
| `c5b685f` EP 过滤补 `.qweight/.qzeros` | **优先级已上调（2026-11-16）**：不是可选优化，是**修一个已在生产里空转的开关**；仓内源码已修、**未烘入镜像**，解冻须用户明示 | **空转四步取证**：①flag 在行 48 live argv；②日志 `default_loader.py:406` 自报 `loading 64/512 experts`（执行者确实跑了）；③镜像内判据 `ep_weight_filter.py:79` 只认 `.weight`/`.weight_packed`；④实际张量名是 `.qweight`/`.qzeros`/`.scales`（expert0 共 432 个＝48层×3proj×3）⇒ **全部放行、一个没滤**，每 rank 仍读全 512 专家，**那句日志是假绿**。修复后（Intel 上实测）无损（64-token 指纹一致）、单 rank 读量 **164.62→115.02 GiB（−30.1%）**；`.scales` 保持不滤（有些后端要全专家 scale）。**Swift 上未实测**；按读量比例折算加载 209.46→~146.4 s（省 ~63 s）属**推算**，纯过滤耗时从未单测。**只动启动 I/O，不动 decode，也不改显存** |
| `ce4f4e7` PLE 按 rank 分片加载 | 不吸收 | 2026-10-26 已判死（读量降到 0.19× 只快 22–33%，且减的是每 rank 拷贝）；分片目录已删 |
| Config B（`enable_mamba_fine_grained_prefix_cache` ＋ `--prefix-match-unit 80`） | 仅文档 | 无 MTP 时为 no-op；MTP 下共享前缀才有行为收益，且额外收益未获铁证 |
| MTP（投机解码） | **用户裁定不做**（2026-11-16） | MTP2 面板同尺 decode 85.33 / 80.35 / 79.91（均值 81.86，≈×1.27–1.35 基线），接受率 45.1%、平均接受长度 1.90；但 262K 撞显存墙：KV 2.25 GiB 时 250K 预填 OOM 158 MiB，KV 1.5 GiB 时容量门拒（≈195,200 tokens）。**本机 MTP 实用上下文 ≤ ~195K**，与 262K 互斥。实测对象为行 26（Intel AutoRound）参数+block400/ConfigB（实验行 53），比值对 Intel 无 MTP 基线 ~63.3（63.30 九档均值）；Swift 权重+MTP2 未实测，同读数对 Swift 66.68 折算约 ×1.19–1.28（均值 ≈1.23），非 Swift 实测收益 |

MTP 的完整材料（配置、崩溃记录、复现要点）未随包发布；复现路线：先测草稿每卡占用（分片 vs 复制），再评估 Swift 权重省显存或 CPU KV offload。

**资产状态（2026-11-16 退役后）**：实验行 49–53 已删注册（备份留档）、0.32 引擎容器与 `vllm-turing:032mp-20261116` 镜像已删、草稿 `mtp-int4-g32`（3.9G）与 0.32 侧模型目录（121G）已删，共释放 **121G**；**保留不清理**（用户裁定「存起来以后有用」）：0.32 base 镜像 `vllm/vllm-openai:v0.32.0`（33.1G，两个 tag）与 0.32 源码树/编译缓存/脚本，留作后续吸收与复现底座。生产镜像 final-v1（`7267ff1c`）与本层 compat1（`9cb13d85`）均在；行 48 Swift 在线、行 26 待命，均未动。