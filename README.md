# vLLM-SM75 × GPUStack 自定义后端（v0.1.7-beta2 线）

把 SM75 版 vLLM 做成 GPUStack 的自定义后端，让 T4 / T10 这类 Turing 老卡在 GPUStack 里跑大模型。
本仓只服务这一条线；上游的 Ultra／面板独立部署形态不在本仓的交付范围内（见文末"与上游的关系"）。

---

## 1 这是什么、我们加了什么

官方 vLLM 在 Turing 上没有加速路径（Marlin、FP8 W8A8 都不支持，7.5 只是它能跑起来的最低门槛）。上游 VLLM-SM75 做了 SM75 适配；我们在它之上加了两层代码：

- **pcieipc**：八卡 PCIe 直连 allreduce（把 FlashInfer 0.7.0 的 PCIe-IPC 回填进镜像内的 0.6.18）；
- **incple**：AutoRound / INC 量化白名单补丁。

另有一组环境变量把底座已有的 QSA、FlashQLA、自建调度器打开——那是**启用**，不是我们实现。

## 2 运行环境（实测下限，不保证兼容上限）

| 件 | 本次实测通过的版本 |
|---|---|
| 操作系统 | Ubuntu 24.04.5 LTS（内核 6.8.0-146-generic） |
| NVIDIA 驱动 | 580.178.04（CUDA 13.0） |
| Docker | 29.8.0（storage `overlayfs`，cgroup v2） |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.2.3 |
| 硬件 | 8×Tesla T10 16 GiB ＋ 251 GiB 主机内存 |

低于上表我们不保证、也未验证；高于它同样不保证，请自行实测。逐项说明见 `gpustack/run-book-v1.md` 第 0 节。

## 3 快速开始（照做即可）

详细步骤与排错见 **`gpustack/run-book-v1.md`**；这里只给最小序列。

```bash
# 0) 先把两枚基础镜像拉下来（否则第一段构建会卡在 60 GB 下载上）
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.2.3

# 1) 拉本仓
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend

# 2) 两段构建：第一段（补丁门＋基座）→ 第二段（四层：pcieipc／incple／时区／面板）
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-gpustack-v1.sh --build
docker build --file gpustack/Dockerfile.v1 --target panel-layer \
  --build-arg BASE_TAG=vllm-sm75:v0.1.7-beta2 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v1 .

# 3) 起 GPUStack（server 与 worker 的具体命令见 run-book §3／§5）
#    同机部署 worker 不需要 token；跨机用 GET /v2/clusters/1/registration-token 取

# 4) 注册后端版本：版本名 0.1.7-sm75.1-custom，镜像填上面那枚 tag，
#    入口覆盖填：python3 /opt/vllm-sm75/panel-sidecar.py vllm serve
#    （也可以用 gpustack/set-panel-autostart-v1.sh 一次写好后端入口与面板三 env）

# 5) 部署模型：表单里整块粘
#    gpustack/args-paste-v1.txt（24 行）＋ gpustack/env-paste-v1.txt（25 行）
bash gpustack/lint-paste-v1.sh      # 末行必须是 LINT_PASTE_OK env=25 args=24
```

**第二段必须取 `--target panel-layer`**：它是超集，含前三层＋面板负载＋BenchLocal 校验器运行时，一枚镜像同时承担"跑模型／起面板／起校验容器"三种角色。

## 4 控制面板（可选，与模型同一个镜像、同一个模型行）

镜像里已带面板（node v22.23.2 ＋ `/opt/sm75-workbench`）。要让面板随模型一起起来，在模型行加三格：

```
SM75_PANEL_AUTOSTART=1
SM75_CONSOLE_HOST=0.0.0.0
SM75_CONSOLE_ROOT=/var/lib/gpustack/panel
```

- 面板口 **1616**（`SM75_CONSOLE_PORT` 可改）；登录令牌在 `$SM75_CONSOLE_ROOT/key`；数据落在 worker 数据目录，重启不丢。
- 不设 `SM75_PANEL_AUTOSTART` 时入口包装纯透传，引擎行为与不带这一层完全一致。
- 单独起面板、三枚 BenchLocal 校验容器与雷达图步骤，见 run-book 第 11 节。

> 已知边界：面板里的 **使用统计／工作台／插件** 三个 native 页与 **BenchLocal 的 harness** 依赖 DSH 工作台（`/opt/harness`，585 MB）。该组件目前只在 Ultra 工作台形态下可用；本仓正在做"扩展开关随模型起"的版本，完成前请按 run-book 第 11 节以独立容器方式使用。

## 5 参数与文档

| 想知道什么 | 看哪里 |
|---|---|
| 从零到可推理的每一步 | `gpustack/run-book-v1.md` |
| 每条参数为什么这么设、怎么调 | run-book **第 12 节**（逐条讲解） |
| 只想复制粘贴 | run-book **第 13 节** ＋ `gpustack/args-paste-v1.txt`／`env-paste-v1.txt` |
| 后端注册表单逐项 | `gpustack/backend-version-card-v1.md` |
| 能力落点与"为什么不是验收线" | `gpustack/compat-matrix-v1.md` |
| 面板自启一键接线 | `gpustack/set-panel-autostart-v1.sh`（默认 dry-run，`--apply` 才改） |

## 6 速度参考（面板形态，不是 GPUStack 形态的验收线）

同一台 8×T10、同一套测法（"大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）"，并发 1，九档 512→131072）：

| 配置 | 平均输出 | 平均预填充 |
|---|---|---|
| VLLM-SM75 v0.1.7-beta2（基线，已含 SM75 适配） | 43.31 tok/s | 2742.54 |
| 同一权重，叠上整组优化 · FP8PLE | 57.73 tok/s | 2749.96 |
| 叠上整组优化 · W4A16-AutoRound | 57.04 tok/s | 2762.51 |

输出侧 +33.3%，预填充几乎不动。**这三张表是在面板形态上量的**，不是 GPUStack 形态的验收线；GPUStack 形态的九档读数另见 run-book 第 10a 节（52.81，且其配置窗口与面板档不同）。怎么读、哪些数不能互相比，见 `gpustack/compat-matrix-v1.md` 最后一节。

## 7 目录速览

```
gpustack/        本线全部交付物：run-book、粘贴块、Dockerfile.v1、补丁门、脚本、证据
docker/          上游构建件（基座 Dockerfile、helpers）——受上游冻结门保护
ultra/source/    面板与 BenchLocal 源码（层 4 从这里取）
vllm/            上游 SM75 适配源码与我们回填的 attention overlay
```

## 8 与上游的关系、许可

- 基座：`vllm/vllm-openai:v0.30.0-cu129`（按 tag＋digest 双钉），上游 SM75 适配见 `vllm/` 与 `docker/`。
- 本仓新增的交付物在 `gpustack/`；上游文件受 `gpustack/gate-tree-unchanged-v1.sh` 的 551 件冻结门保护，改动需在 `NOTICE` 登记例外。
- 许可：Apache-2.0（见 `LICENSE`）；第三方件的许可与来源见 `NOTICE` 与 `gpustack/backend-version-card-v1.md`。

[English](README.en.md)
