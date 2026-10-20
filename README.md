# vLLM-SM75 × GPUStack 自定义后端（v0.1.7-beta2 线）

把 SM75 版 vLLM 做成 GPUStack 的自定义后端，让 T4 / T10 这类 Turing 老卡在 GPUStack 里跑大模型。
本仓只服务这一条线；上游的 Ultra／面板独立部署形态不在本仓的交付范围内（见文末"与上游的关系"）。

---

## 1 这是什么、我们加了什么

官方 vLLM 在 Turing 上没有加速路径（Marlin、FP8 W8A8 都不支持，7.5 只是它能跑起来的最低门槛）。上游 VLLM-SM75 做了 SM75 适配；我们在它之上加了两层代码：

- **pcieipc**：八卡 PCIe 直连 allreduce（把 FlashInfer 0.7.0 的 PCIe-IPC 回填进镜像内的 0.6.18）；
- **incple**：AutoRound / INC 量化白名单补丁。

另有一组环境变量把底座已有的 QSA、FlashQLA、自建调度器打开——那是**启用**，不是我们实现。

**交付默认路线（2026-10-20 定）= 一条命令从 vLLM 基座构建的最快档（`-v6`）**：在既有四层链末端加一层运行期包钉版，把 transformers/tokenizers 钉到 **5.15.1/0.22.2**；同机同尺 GPUStack 面板直连 **56.79/57.15/57.17（均值 57.04）**（A 路线，输出 1024，并发 1，SMT Off），达到历史 AutoRound 57.04。**同一枚镜像两种用法**：GPUStack 后端（入口 `panel-sidecar.py vllm serve`）或单独面板（入口 `runtime-entrypoint.py`，见 run-book §11）。回退：`--no-tf-pin` 产出基座原版 v4（56.26/56.97）；有旧 Ultra 镜像的机器也可用 `build-oldpanel-backend-v1.sh` 薄层路线（57.15，等价备选）。

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
| CPU/内存 | 高单核 CPU ＋ 高频内存；参考 EPYC 7402 24C/24T、8×32 GB DDR4-2400；PLE 表常驻主机内存，容量要覆盖 PLE 表＋KV＋系统 |
| SMT | BIOS **SMT=Off**；同镜像同档 8192 54.69→59.35 tok/s（2026-10-20），代价逻辑核 48→24 |
| P2P/虚拟化 | **物理机直装 Ubuntu** 才实测（56/56、13.1–13.2 GB/s）；G292-Z20 的 PLX switch 整体直通 VM 仅理论、未实操 |

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

# 2) 一条命令：上游 vLLM 基座 → SM75 基座 → 自定义后端（默认最快档）
#    默认在最后一层把 transformers/tokenizers 钉到 5.15.1/0.22.2（实测 57.04 tok/s）；
#    产物 tag：vllm-sm75-gpustack:v0.1.7-beta2-v6。
#    回退基座原版 5.17.0/0.23.2（56.26/56.97）：bash gpustack/build-backend-v1.sh --no-tf-pin
#    （产物 tag 为 -v4；基座已在本地就复用，--rebuild-base 强制重建，--drop-base 删中间 tag）
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh

# 3) 起 GPUStack（server 与 worker 的具体命令见 run-book §3／§5）
#    同机部署 worker 不需要 token；跨机用 GET /v2/clusters/1/registration-token 取

# 4) 注册后端版本（默认 = 最快档）
#    版本名 0.1.7-sm75.1-fast-custom，镜像填上一步产物 vllm-sm75-gpustack:v0.1.7-beta2-v6，
#    入口覆盖填：python3 /opt/vllm-sm75/panel-sidecar.py vllm serve
#    （也可以用 gpustack/set-panel-autostart-v1.sh 一次写好后端入口、面板 env 与 SM75_EXTENSIONS）
#    回退路径：版本 0.1.7-sm75.1-custom ＋ 镜像 vllm-sm75-gpustack:v0.1.7-beta2-v4

# 5) 部署模型：表单里整块粘
#    gpustack/args-paste-videooff-v1.txt（24 行，A 路线：保留图像、关闭视频）＋ env-paste-v1.txt（25 行）
#    纯文本改用 gpustack/args-paste-v1.txt（24 行）
#    要面板与三种测试，再粘 gpustack/env-paste-panel-v1.txt（5 行，含统一端口 1615）
bash gpustack/lint-paste-v1.sh --args gpustack/args-paste-videooff-v1.txt   # 末行须为 LINT_PASTE_OK env=25 args=24
```

**链条末端取 `--target runtime-pin-layer`（默认）**：它是超集，含前四层＋面板负载＋BenchLocal 校验器运行时＋随仓测试资产＋运行期钉版，一枚镜像同时承担"跑模型／起面板／起校验容器"三种角色，也能单独起面板。

## 4 控制面板（可选，与模型同一个镜像、同一个模型行）

镜像里已带面板（node v22.23.2 ＋ `/opt/sm75-workbench`）。要让面板和扩展随模型一起起来，在 **GPUStack 模型部署表单的“环境变量”** 里加下面五格（与 25 行 env 块同一处，整块粘贴；这不是宿主 shell 的 `export`）：

```
SM75_PANEL_AUTOSTART=1
SM75_CONSOLE_HOST=0.0.0.0
SM75_CONSOLE_ROOT=/var/lib/gpustack/panel
SM75_CONSOLE_PORT=1615
SM75_EXTENSIONS=panel,harness,benchlocal,speedtest,sql
```

- `SM75_EXTENSIONS` 是扩展开关：`panel`（面板）、`harness`（Ultra 工作台 DSH）、`benchlocal`（三枚校验器）、`speedtest`／`sql`（测试资产启用）；只想要面板就写 `panel`。
- 后端版本的入口覆盖仍要填：`python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`（`gpustack/set-panel-autostart-v1.sh --apply` 会连同上面五格一起写好）。
- 面板口**统一写 1615**：旧引擎薄层镜像的 ENV 本来就是 1615，显式写死让 v4 回退路径也起在同一端口，换镜像不用改书签/防火墙。登录令牌在 `$SM75_CONSOLE_ROOT/key`；数据落在 worker 数据目录，重启不丢。
- 不设 `SM75_PANEL_AUTOSTART` 时入口包装纯透传，引擎行为与不带这一层完全一致。
- **GPUStack 形态下面板自动进入外部引擎模式**：隐藏“模型库／运行配置／设置”，启停与模型连接交给 GPUStack；速度/SQL 测试、BenchLocal、监控照常使用。

> Ultra 工作台形态（使用统计／工作台／插件）由扩展模块在模型容器内启动 DSH；DSH 的 `/dsh/home`、`/dsh/workspace` 落在 worker 数据目录，重启保留。DSH 的独立容器用法见 run-book 第 11 节。

## 5 参数与文档

| 想知道什么 | 看哪里 |
|---|---|
| 从零到可推理的每一步 | `gpustack/run-book-v1.md` |
| 每条参数为什么这么设、怎么调 | run-book **第 12 节**（逐条讲解） |
| 只想复制粘贴 | run-book **第 13 节** ＋ `gpustack/args-paste-videooff-v1.txt`（默认 A 路线）／`args-paste-v1.txt`（纯文本）／`env-paste-v1.txt` |
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

输出侧 +33.3%，预填充几乎不动。**这三张表是在面板形态上量的**，不是 GPUStack 形态的验收线；GPUStack 形态的九档读数另见 run-book 第 10a/14 节（旧九档 52.81 与本次 6 档 56.26 不是同一路径/档位）。怎么读、哪些数不能互相比，见 `gpustack/compat-matrix-v1.md` 最后一节。

> 上面的 57.73/57.04 是 2026-10-07/08 的 **SMT On** 历史面板表；下面 2026-10-20 补充是 **SMT Off**。旧 v5/v4 51.1/51.8 也是 SMT On，SMT Off A/B 待复核，不引用。

### 2026-10-20 GPUStack 形态补充（不替换上面的面板表）

同机 8×T10、SMT Off、AutoRound W4A16、output 1024、并发 1、面板直连：

| 口径 | 8192 | 16384 | 32768 | 均值 |
|---|---:|---:|---:|---:|
| **默认交付 `-v6`（v4 栈＋运行期钉 5.15.1/0.22.2）＋A 路线** | **56.79** | **57.15** | **57.17** | **57.04** |
| 备选：旧引擎薄层后端（transformers 5.15.1）＋A 路线（image16/video0） | 56.92 | 57.57 | 56.96 | 57.15 |
| 旧引擎薄层后端、text-only，两轮 | 57.56 / 57.07 | 56.98 / 58.11 | 57.22 / 58.04 | 57.25 / 57.74 |
| v4 后端（transformers 5.17.0）panel sidecar 直连，六档 | 56.97 | — | — | 56.26 |
| v4 后端 8180 网关，四档 | 53.57 | — | — | 54.71 |
| 旧 Ultra 面板镜像 text-only 直连（独立容器），五档 | 59.35 | 58.73 | 59.43 | 58.59（五档） |

- **默认档 `-v6` 由仓库一条命令从 vLLM 基座构建**：在四层链末端加层 5，把 `transformers==5.15.1`、`tokenizers==0.22.2` 成对钉住；与 v4 的完整 pip 清单差集恰为这两项。实测 **57.04**，与历史 AutoRound 57.04 一致。
- **归因口径**：关键变量是**这两个包成对钉版**，不是“旧镜像整包”。证据：v4 栈钉对＝57.04；旧栈钉对＝57.15；旧栈换 5.17/0.23＝55.36；v4 栈只换 transformers 5.15.1（tokenizers 未配对）＝51.1；v5 的反向结果因此不能当作“旧包更慢”的证据。**不要单独改 transformers，必须两项一起。**
- **3% 未归因**：旧面板独立容器同模型 text-only 8192 59.35 / 五档 58.59，比 GPUStack 面板直连高约 3%。候选只剩 `pstate-supervisor`（旧容器有、GPUStack sidecar 没起；采样 99.3% P0，只解释极小部分）与 `cudagraph_capture_sizes [1,2,4]` vs `[1,2,4,8,16]`；`--auto-sleep-idle-timeout` 经用户判定不影响、已撤。**不得写成已解释。**
- **A 路线首次启动要 JIT 编译 FlashInfer 多模态 kernel**：`init engine` 211 s、总启动约 9–10 分钟；缓存 `/root/.cache/flashinfer/0.6.18/75/` 不持久，重建即重编。
- **第三线判据（旧栈单换 5.17/0.23）**：54.91/55.81（合并 55.36），比旧栈钉对低 1.8–2.4 ⇒ **钉版组合错向会掉速；方向与“必须 5.15.1/0.22.2”一致**。
- 引用这些数时必须带全条件（后端引擎包/路径/SMT/多模态），详细见 run-book 第 14 节、完整教程与 `调度/SM75性能变量与历史档位速查-20261019-v1.md`。

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
