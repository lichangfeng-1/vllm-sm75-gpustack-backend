# vLLM-SM75 × GPUStack 自定义后端

把 SM75 版 vLLM 接入 GPUStack，让 T4 / T10 这类 Turing 卡在 GPUStack 里跑大模型。同一枚镜像也可单独起 Ultra / 面板；两种形态共用同一条构建链与 SM75 适配。

## 你得到什么

- **GPUStack 自定义后端**：在 GPUStack 模型行里跑 `vllm serve`，支持 8×T10、262K、A 路线（图 4 / 视频 0）。
- **独立 Ultra / 面板**：同一镜像用 `runtime-entrypoint.py` 起控制面板、Ultra 工作台与 BenchLocal 校验器。
- **两层 SM75 适配**：pcieipc（八卡 PCIe 直连 allreduce）与 incple（AutoRound / INC 量化白名单）；上游 QSA / FlashQLA / 调度器由环境变量启用。
- **门控优化包 `vllm_sm75`**：只新增、不覆盖上游文件；`VLLM_SM75_OPT=auto` 只开已验证项，`0` 全关等于原基线。

## 快速开始

### 0 前置

- 8×Tesla T10 16 GiB／卡，主机内存 ≥ 251 GiB，数据盘 ≥ 250 GB。
- Ubuntu 24.04、NVIDIA 驱动 580.178.04（CUDA 13.0）、Docker 29.8、NVIDIA Container Toolkit 1.20、GPUStack v2.3.1。
- 详细版本与说明见 [§运行环境](#运行环境)。

### 1 拉两个基础镜像

```bash
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.3.1
```

### 2 构建后端

**0→1（教程口径：上游 vLLM → SM75 基座 → GPUStack 后端）**

```bash
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh --two-stage
```

起点是上游 vLLM 0.30.0；第一段产出中间基座 `vllm-sm75:v0.1.7-beta2`（只是构建缓存），第二段产出公开一键镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v7`。成功标志：`BUILD_BACKEND_OK ... target=runtime-pin-layer mode=two-stage`。

**升级到最新生产层（可选；不是 0→1 上游）**

0→1 的输出是 `vllm-sm75-gpustack:v0.1.7-beta2-v7`；v7/opt15 都是中间或交付基座，**不是上游**。按你的起点二选一：

- 一条命令从上游到最新（0→1 + 升级）：`bash gpustack/build-latest-from-zero-v1.sh`（已有 v7 时 `SKIP_V7=1`），产出 `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1` 与 `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1-sqlprompt-audit`。
- 已有 opt15（内部优化基座），在仓库根目录：

```bash
docker build -f gpustack/Dockerfile.final-v1-20261029 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1 .
docker build -f gpustack/Dockerfile.sql-prompt-audit-v20261029 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1-sqlprompt-audit .
```

- `final-v1`：现役生产镜像（父 opt15；console maxTokens、测速 v3.5 配额/上传、SQL finalanswer）。**现役生产模型是行 48 Swift-1.5，用的就是这枚镜像（`7267ff1c3db0`）——换的是模型权重与行号，不是镜像。**
- `sqlprompt-audit`：最新 SQL 提示词审计薄层（父 final-v1；财年/财季格式约束 + 工具结果截断）。
- `compat1`（可选实验层）：Turing 0.32 兼容件薄层（父 final-v1；功能开关默认关；已构建、未注册），配方 `gpustack/Dockerfile.compat-v1-20261116`，说明见 [tuning](docs/tuning/compat-turing032-20261116.md)。

### 3 起 GPUStack，注册后端版本

server / worker 的完整命令见 [run-book §3/§5](gpustack/run-book-v1.md)。注册后端时填：

**0→1 基线（教程口径）**
- 版本名：`0.1.7-sm75.1-fast-custom`
- 镜像：`vllm-sm75-gpustack:v0.1.7-beta2-v7`

**最新升级层（可选）**
- 版本名：`0.1.7-sm75.1-sqlprompt-audit-custom`
- 镜像：0→1 路径 `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1-sqlprompt-audit`；已有 opt15 用 `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1-sqlprompt-audit`
- 现役生产：`0.1.7-sm75.1-final-v1-custom` + `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1`；模型行 **48 Swift-1.5**（served name `EXP-swift15-qwen3.8-flash-next-w4a16-autoround`，端口 40000，参数 25 项）在线，行 26（Intel AutoRound）replicas=0 待命
- 入口覆盖：`python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`

一键写入入口与面板 env：`gpustack/set-panel-autostart-v1.sh --apply`（默认 dry-run；先看再执行）。

### 4 部署模型

在 GPUStack 模型行的“后端参数”和“环境变量”里整块粘贴：

- 参数：`gpustack/args-paste-videooff-v1.txt`（24 行，A 路线；纯文本用 `args-paste-v1.txt`）。**这是 0→1 基线块**（PIECEWISE／批 4096／`flashqla_sm75`／`--generation-config=vllm`）；生产档另按下方与调优文档改四处，且现役行 48 带 `--override-generation-config`（0→1 块不含，缺它会丢 top_p/top_k）。
- 环境变量：`gpustack/env-paste-v1.txt`（25 行）。
- 要面板、工作台与校验器：再加 `gpustack/env-paste-panel-v1.txt`（8 行，端口统一 1615）。

自检：

```bash
bash gpustack/lint-paste-v1.sh --args gpustack/args-paste-videooff-v1.txt
# 末行须为 LINT_PASTE_OK env=25 args=24
```

要复现 **2026-10-27 Intel opt15 历史档（63.30 tok/s）**，再把 `--compilation-config` 改成 `FULL_AND_PIECEWISE`，加 `--max-num-batched-tokens=8192`、`--gdn-prefill-backend=triton`、`VLLM_FIREFLY_AR=0`、`VLLM_SM75_OPT=auto`；完整对比见 [历史调优文档](docs/tuning/flash-next-20261027.md)。**现役生产档是 Swift-1.5 行 48（decode 均 66.68）**，同一组引擎参数、不同权重，另需 `--override-generation-config`；现读运行时值与判定见 [Swift 现役调优页](docs/tuning/swift15-prod-20261116.md)。

## 运行环境

| 件 | 本次实测通过的版本 |
|---|---|
| 操作系统 | Ubuntu 24.04.5 LTS（内核 6.8.0-146-generic） |
| NVIDIA 驱动 | 580.178.04（CUDA 13.0） |
| Docker | 29.8.0（storage overlayfs，cgroup v2） |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.3.1（2026-11-14 实测，commit `458030e`；回退链 v2.3.0 → v2.2.3，回退须恢复升级前 pg_dump 备份） |
| 硬件 | 8×Tesla T10 16 GiB／卡，主机内存 ≥ 251 GiB |
| BIOS | SMT=Off；物理机直装 Ubuntu（P2P 13.1–13.2 GB/s） |

低于 / 高于上表都不保证；逐项说明见 [run-book §0](gpustack/run-book-v1.md)。

> GPUStack v2.3.1（与 2.3.0 同）的 worker 必须保留 `--runtime=nvidia --privileged --security-opt label=disable`；缺整组会 `gpu_devices=[]`。升级 / 回退见 [release notes v2.3.1](gpustack/release-notes-v2.3.1-v1.md)（上一版 [v2.3.0](gpustack/release-notes-v2.3.0-v1.md) 为历史件）。

## 性能

**现役生产行（2026-11-16，Swift-1.5，模型行 48）**：8×T10，SMT Off，262K，镜像 `…opt15-final-v1`（与历史档同一枚），面板三档 8192/16384/32768、output 1024、并发 1、temp 1、topP 0.95：

| Prompt | TTFT ms | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| 8192 | 2030.70 | 4039.33 | **66.61** |
| 16384 | 4333.09 | 3783.84 | **66.48** |
| 32768 | 9368.64 | 3498.81 | **66.94** |

decode 三档均值 **66.68**（**三档各一次，未做多次中位**）。**面板「平均预填充」汇总字段本轮未取，故不写该字段**：三格算术均值 3773.99 与该字段口径不同（面板按「(Σ各格＋末格)÷(格数＋1)」算），不得冒充。served name `EXP-swift15-qwen3.8-flash-next-w4a16-autoround`、端口 40000、参数 25 项（`--block-size=16`，未夹带吸收优化）。现读运行时值、显存口径（可分配上限 **15929 MiB/卡**）与四组参数判定见 [Swift 现役调优页](docs/tuning/swift15-prod-20261116.md)。

**历史基线（2026-10-27 生产行，Intel AutoRound，opt15 薄层）**：8×T10，SMT Off，262K，A 路线，面板九档 512→131072、output 1024、并发 1：

| 配置 | 平均输出 | 平均预填 |
|---|---:|---:|
| 上游 VLLM-SM75 v0.1.7-beta2（仅 SM75 适配，FP8PLE 权重，2026-10-07/08） | 43.31 | 2742.54 |
| 历史 PIECEWISE 档（W4A16-AutoRound，2026-10-07/08） | 57.04 | 2762.51 |
| 2026-10-27 Intel opt15 生产行（FULL + 批 8192 + triton/gdn_tc + AR=0，W4A16）——**历史基线，已被 2026-11-16 Swift 行 48 取代** | 63.30 | 3485.13 |

- 历史窗口（2026-10-27 Intel 档）8K/16K/32K/64K/128K 输出 **63.52/63.15/63.00/62.48/61.50**；TTFT **2013/4127/9200/19758/42595 ms**。
- 历史窗口（2026-10-27 Intel 档）相对上游 **+46.2%**（跨场次，且权重不同；本交付 PLE 为 BF16、上游 FP8PLE 为 FP8，不是降精度换速度）；同权重 FP8PLE 对照是 +33.3%（57.73）。**这两个百分比属历史对照，不与现役 66.68 相减。**
- 直测尺的 2019/8643 ms 与 59.6–61.6 tok/s 是另一把尺，**不与面板 63.30 相减，也不与现役 66.68 相减**；逐项读数、负结果与硬件地板见 [历史调优文档](docs/tuning/flash-next-20261027.md)。
- 63.30 是 2026-10-27 在 opt15 薄层上的同机读数；公开一键镜像 `-v7` 已内置同一门控包，按同一组生产参数可复现，读数以本机同尺为准。final-v1 / sqlprompt-audit 只改 console 与面板扩展，不改引擎参数，63.30 档位读数不变。不加这些参数时是历史 PIECEWISE 档（57.04）。
- **现役 66.68 与 63.30 的差异来自模型权重（Swift-1.5 对 Intel AutoRound），不是镜像或平台升级**：两者同一枚 final-v1 镜像；Swift 线试跑期的同版对照结论为 +5.5%（对 final-v1 Intel 档 63.4）。平台 v2.3.1 升级后未重跑历史九档，**不得把 66.68 与 63.30 的差记作升级收益**。

## 门控优化包

镜像内含 `vllm_sm75`，通过 `vllm.general_plugins` 自动安装；不设开关时等于原基线。

- `VLLM_SM75_OPT=auto`：只开已通过同尺 A/B（≥1%、3–5 次中位）的项；当前为 `gemv_m1`（decode +1.4%）与 `gdn_tc`（预填 +6.5/12.0/10.1%，需 `--gdn-prefill-backend=triton`）。
- `VLLM_SM75_OPT=0`：全关；逐项用 `VLLM_SM75_<ITEM>=0/1` 覆盖。
- 逐项状态（2 项转正、4 项判负、2 项仅测量）与负结果：[vllm_sm75/README.md](vllm_sm75/README.md)、[调优文档](docs/tuning/flash-next-20261027.md)。

## 文档

- [docs/README.md](docs/README.md)：文档索引。
- [run-book](gpustack/run-book-v1.md)：0→1 部署、粘贴块、排错。
- [tuning（现役）](docs/tuning/swift15-prod-20261116.md)：Swift-1.5 行 48 基线（decode 均 66.68）、现读运行时值、显存口径 15929 MiB/卡、四组参数判定与空间边界。
- [tuning（历史）](docs/tuning/flash-next-20261027.md)：2026-10-27 Intel opt15 的 63.30 基线、负结果、硬件地板。
- [compat tuning](docs/tuning/compat-turing032-20261116.md)：Turing 0.32 兼容件层（默认关）、容量档 block400、MTP 结论。
- [backend card](gpustack/backend-version-card-v1.md)：后端注册表单逐项（含 2026-10-29 内部部署增补）。
- [final-v1 配方](gpustack/Dockerfile.final-v1-20261029) / [SQL 提示词审计配方](gpustack/Dockerfile.sql-prompt-audit-v20261029)：最新镜像薄层。
- [compat matrix](gpustack/compat-matrix-v1.md)：能力落点与读数边界。
- [release notes](gpustack/release-notes-v2.3.1-v1.md)：GPUStack v2.3.1（现役）；[v2.3.0](gpustack/release-notes-v2.3.0-v1.md) 为历史件。
- [design](docs/design/)：设计与调研笔记。
- [vllm_sm75](vllm_sm75/README.md)：门控包。

## 仓库结构

```text
gpustack/   交付物：run-book、粘贴块、Dockerfile.v1、补丁门、脚本
docker/     上游构建件（基座 Dockerfile、helpers）
ultra/      面板与 BenchLocal 源码
vllm/       上游 SM75 适配与 attention overlay
vllm_sm75/  门控优化包
```

## 与上游、许可

- 基座：`vllm/vllm-openai:v0.30.0-cu129`（tag + digest 双钉）；上游 SM75 适配在 `vllm/`、`docker/`。
- 本仓新增交付物在 `gpustack/`；上游文件受冻结表保护，例外登记在 `NOTICE`。
- 许可：Apache-2.0；第三方来源见 `NOTICE` 与 `gpustack/backend-version-card-v1.md`。

## 致谢

本仓的推理、部署与面板能力建立在下列开源工作之上：

- **VLLM-SM75（fishensw）**：SM75/Turing 的 vLLM 适配与 Ultra 控制台来源（Apache-2.0）；
  上游基座 vLLM 0.30.0，逐项来源见 `NOTICE`。
- **vLLM、FlashInfer、PyTorch、Triton、NCCL**：推理运行时、注意力/通信内核与集合通信基础。
- **GPUStack**：模型部署、后端版本与运维控制面。
- **AutoRound（Intel）** 与 **Qwen3.8-Flash-Next**：W4A16 量化工具链与基础模型；本仓不重新分发权重。
- **DeepSeek Harness（@deepseek-ai/dsh）**：控制台“工作台”的会话与插件框架。
- **llm_speedtest（gengchaogit）**：控制台“模型测试”页的测速工具；其致谢包括 Bilibili 创作者“纸鸢随风”、
  DeepSeek-R1-0528 开源测速脚本与 llm-inference-bench。
- 也感谢 vLLM、SGLang、Ollama、llama.cpp 等推理引擎社区公开的基准与调优实践。

第三方许可、版本与逐项来源见 `NOTICE` 与 `gpustack/backend-version-card-v1.md`。

## GPUStack v2.3.1（现役）

2026-11-14 升到 `quay.io/gpustack/gpustack:v2.3.1`（commit `458030e`）：平台侧只换 server/worker 镜像，自定义后端版本、模型定义与凭据留在 data_dir 里原样可用；worker 运行时参数整组保留。升级 / 回退与**未复核门禁**见 [release notes v2.3.1](gpustack/release-notes-v2.3.1-v1.md) 与 run-book §0.2。

- **新特性门槛**：Decision Model Serving（`/v1/systemone` ＋ Decision Playground，类型化 choice/score/boolean 判定）**要求 vLLM ≥ 0.32.0**；本交付 SM75 优化版是 **0.30 ⇒ 用不了**。
- **未复核，不当已修写**：上游 #6660「后端 4xx 转 503」是否真修（release 只写 non-streaming，我方流式复现过）；`GPUSTACK_MODEL_VRAM_CLAIM` 硬编码是否可删（v2.3.1 支持含 PLE/Engram 的 CPU-offloaded 显存估算，但未对比新旧估算值前不得删）。
- **历史（2.3.0，2026-10-23 实测）**：离线文档、Anthropic `/v1/messages`、模型 YAML 导出/导入、revision / 回滚已测；这些面在 v2.3.1 下未逐项重测。PD 分离、共享 KV、vGPU / MIG、多副本负载路由在 8×T10 单副本形态下仍不适用。见 [release notes v2.3.0](gpustack/release-notes-v2.3.0-v1.md) 与 run-book §0.1。

[English](README.en.md)
