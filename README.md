# vLLM-SM75 × GPUStack 自定义后端（v0.1.7-beta2 线；GPUStack 2.3.0 实测兼容）

把 SM75 版 vLLM 做成 GPUStack 的自定义后端，让 T4 / T10 这类 Turing 老卡在 GPUStack 里跑大模型。
**本仓兼得两种形态**：既交付 **GPUStack 自定义后端**（模型行里跑 `vllm serve`），也交付 **Ultra／面板独立部署形态**（同一枚镜像用 `runtime-entrypoint.py` 单独起面板与 Ultra 工作台）。两者共用同一条构建链、同一套 SM75 适配与扩展；选哪种只取决于入口与 env（见 §1、§3、§4 与文末"与上游的关系"）。

---

## 1 这是什么、我们加了什么

官方 vLLM 在 Turing 上没有加速路径（Marlin、FP8 W8A8 都不支持，7.5 只是它能跑起来的最低门槛）。上游 VLLM-SM75 做了 SM75 适配；我们在它之上加了两层代码：

- **pcieipc**：八卡 PCIe 直连 allreduce（把 FlashInfer 0.7.0 的 PCIe-IPC 回填进镜像内的 0.6.18）；
- **incple**：AutoRound / INC 量化白名单补丁。

另有一组环境变量把底座已有的 QSA、FlashQLA、自建调度器打开——那是**启用**，不是我们实现。

**交付默认路线（2026-10-21 定）= 一条命令从上游 vLLM 0.30.0 构建的最快档（`-v7`）**：在既有四层链末端加一层运行期包钉版，把 transformers/tokenizers 钉到 **5.15.1/0.22.2**；同机同尺 GPUStack 面板直连 **56.83–57.04**（PIECEWISE 历史档，A 路线，输出 1024，并发 1，SMT Off；2026-10-21 复核）；生产行 4 改用 **FULL_AND_PIECEWISE** 后同尺九档均值 **62.28**（2026-10-26）。**同一枚镜像两种用法**：GPUStack 后端（入口 `panel-sidecar.py vllm serve`）或单独面板（入口 `runtime-entrypoint.py`，见 run-book §11）。回退：`--no-tf-pin` 产出基座原版 v4（56.26/56.97）；有旧 Ultra 镜像的机器也可用 `build-oldpanel-backend-v1.sh` 薄层路线（57.15，等价备选）。

### 1.1 门控优化包 `vllm_sm75`（2026-10-26 起）

镜像里多了一个与 `vllm` 同级的顶层包 `site-packages/vllm_sm75/`，它自己注册了
`vllm.general_plugins` 入口点，所以**不需要改 vLLM 任何文件**，引擎每次启动会自动调用
`vllm_sm75.install()`。全关时它一行子模块都不加载，行为与没有这个包完全一致。

四级门控（逐级短路）：

| 级 | 条件 | 关掉的办法 |
|---|---|---|
| ① | 计算能力必须是 7.5（Turing） | 非 7.5 的卡自动跳过 |
| ② | `VLLM_SM75_OPT` = `auto` / `0` / `1` | `VLLM_SM75_OPT=0` 一刀全关＝现基线 |
| ③ | 逐项开关 `VLLM_SM75_<ITEM>` = `auto` / `0` / `1` | 单项关：`VLLM_SM75_GEMV_M1=0` |
| ④ | 启动参数显式指定 | 最高优先 |

`auto` **不等于全开**：只开登记表里已经过 L4 性能门的项（门＝同尺 A/B ≥ **1%**、3–5 次中位）。
登记表是包里的 `manifest.json`，当前状态：

| 项 | 开关 | 状态 | 说明 |
|---|---|---|---|
| `gemv_m1` | `VLLM_SM75_GEMV_M1` | ✅ verified，`auto` 生效 | decode 的 M=1 GEMV，3 个形状换用 Triton 内核；L2 相对误差 5e-6/1.9e-4/0，端到端 **64.32 → 65.24 tok/s（+1.4%）** |
| `gdn_tc` | `VLLM_SM75_GDN_TC` | ✅ verified，`auto` 生效 | GDN 预填换成张量核内核（`mma.sync m16n8k8`）。**必须同时加 `--gdn-prefill-backend=triton`**（该项替换的是 FLA 的 chunk 内核）；预填 +6.5%/+12.0%/+10.1%（8K/32K/126K），logprob 差 0.0035 nats |
| `ar_prefill` | `VLLM_SM75_AR_PREFILL` | ❌ 已判负（默认关） | 让 PCIe-IPC 也服务预填的 21 MB all-reduce：实测反而慢 15–22%，NCCL ring 在这个尺寸更好 |
| `ar_overlap` | `VLLM_SM75_AR_OVERLAP` | ❌ 已判负（默认关） | 把 `RowParallelLinear` 按 token 切块、异步 all-reduce，想用下一块 GEMM 盖住通信。同镜像 A/B：开 2306.6/10081.5 ms vs 关 2266.7/10021.7 ms（约 7.9K/31.6K 预填，5 次中位），无收益；且每步约 98 次 AR 里它只能碰到 48 次（MoE 的 48 次与 embedding/PLE 在外） |
| `hc_fuse` / `gdn_fuse` / `qsa_merge` | 同名 | 已融合，永不加载 | 复核证明上游已经融合好，不需要重写 |
| `qsa_slice` | `VLLM_SM75_QSA_SLICE_ROWS` | ❌ 已判负（默认关） | QSA 选页切片 1024→4096 行：8K 2019.1→2000.7 ms、32K 持平；不是预填空隙的来源 |
| `qsa_timing` / `ar_q8_probe` | `VLLM_SM75_QSA_TIMING` / `VLLM_SM75_AR_Q8_PROBE` | 🔬 仅测量（默认关，永不 auto） | 前者给 QSA 路径计时（实测全部 <2 ms）；后者只量化不改通信，用于量 8-bit 集合通信的精度代价——按"不降精度"约束已作废，仅留档 |

新增改动一律遵守：**只新增、不覆盖**镜像里已有文件；每项独立开关；默认关；全关等于原基线。

调优基线与逐项判定（含 9 项负结果与硬件天花板）见 [`docs/tuning/flash-next-20261027.md`](docs/tuning/flash-next-20261027.md)；外部技术对照（vLLM/SGLang/DeepSeek）见 [`docs/design/external-techniques-20261027.md`](docs/design/external-techniques-20261027.md)。

## 2 运行环境（实测下限，不保证兼容上限）

| 件 | 本次实测通过的版本 |
|---|---|
| 操作系统 | Ubuntu 24.04.5 LTS（内核 6.8.0-146-generic） |
| NVIDIA 驱动 | 580.178.04（CUDA 13.0） |
| Docker | 29.8.0（storage `overlayfs`，cgroup v2） |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.3.0（2026-10-23 实测；上一版 v2.2.3 可回退） |
| 硬件 | 8×Tesla T10 16 GiB ＋ 251 GiB 主机内存 |
| CPU/内存 | 高单核 CPU ＋ 高频内存；参考 EPYC 7402 24C/24T、8×32 GB DDR4-2400；PLE 表常驻主机内存，容量要覆盖 PLE 表＋KV＋系统 |
| SMT | BIOS **SMT=Off**；同镜像同档 8192 54.69→59.35 tok/s（2026-10-20），代价逻辑核 48→24 |
| P2P/虚拟化 | **物理机直装 Ubuntu** 才实测（56/56、13.1–13.2 GB/s）；G292-Z20 的 PLX switch 整体直通 VM 仅理论、未实操 |

低于上表我们不保证、也未验证；高于它同样不保证，请自行实测。逐项说明见 `gpustack/run-book-v1.md` 第 0 节。

> **v2.3.0 升级实测（2026-10-23）**：同一枚 `-v7` 镜像、同一模型定义原样迁移；8 卡与入口覆盖保持，实例端口由平台重新分配（本机 40052→40000）。**worker 必须保留 `--runtime=nvidia --privileged --security-opt label=disable` 这三项**，缺整组就会 `gpu_devices=[]`（整组缺失对照；逐项未验），调度器报 “does not support backend version”。升级与回退见 run-book §0.1。

## 3 快速开始（照做即可）

详细步骤与排错见 **`gpustack/run-book-v1.md`**；这里只给最小序列。

```bash
# 0) 先把两枚基础镜像拉下来（否则首次构建会卡在约 42 GiB 的上游基座拉取上）
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.3.0

# 1) 拉本仓
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend

# 2) 一条命令：上游 vLLM 基座 → 自定义后端（默认最快档）
#    默认 --two-stage（已验证，会产出 vllm-sm75 中间基座 tag）；
#    --allinone 为单次构建可选路径（生成器可复现，未完整验收）。
#    默认在最后一层把 transformers/tokenizers 钉到 5.15.1/0.22.2（实测 56.83–57.04 tok/s）；
#    产物 tag：vllm-sm75-gpustack:v0.1.7-beta2-v7。
#    回退基座原版 5.17.0/0.23.2（56.26/56.97）：bash gpustack/build-backend-v1.sh --no-tf-pin（产物 -v4-nopin）
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh --two-stage

# 3) 起 GPUStack（server 与 worker 的具体命令见 run-book §3／§5）
#    同机部署 worker 不需要 token；跨机用 GET /v2/clusters/1/registration-token 取

# 4) 注册后端版本（默认 = 最快档）
#    版本名 0.1.7-sm75.1-fast-custom，镜像填上一步产物 vllm-sm75-gpustack:v0.1.7-beta2-v7，
#    入口覆盖填：python3 /opt/vllm-sm75/panel-sidecar.py vllm serve
#    （也可以用 gpustack/set-panel-autostart-v1.sh 一次写好后端入口、面板 env 与 SM75_EXTENSIONS）
#    回退路径：版本 0.1.7-sm75.1-custom ＋ 镜像 vllm-sm75-gpustack:v0.1.7-beta2-v4-nopin

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
| GPUStack 2.3.0 升级与新特性 | `gpustack/release-notes-v2.3.0-v1.md` |
| 面板自启一键接线 | `gpustack/set-panel-autostart-v1.sh`（默认 dry-run，`--apply` 才改） |

## 6 速度参考（面板形态，不是 GPUStack 形态的验收线）

### 2026-10-27 优化线（SM75 基座 v13 + `opt15` 薄层，直测同尺，GPUStack 形态）

在由上游 **vLLM 0.30.0** 构建的 SM75 基座 `v0.1.7-beta2-v13` 上叠门控优化包与调参（`--max-num-batched-tokens=8192` +
`--gdn-prefill-backend=triton` + `VLLM_FIREFLY_AR=0`），直测同尺（随机前缀绕缓存、`ignore_eos`、流式 TTFT，5 次中位）：

| 指标 | 优化后 | 对照（4096 + flashqla） | 变化 |
|---|---:|---:|---:|
| 预填 8K TTFT | **2019 ms** | 2266.7 ms | **−10.9%** |
| 预填 32K TTFT | **8643 ms** | 10021.7 ms | **−13.7%** |
| 解码 8K 上下文 / 256 输出 | **59.6–61.6 tok/s** | ≈60 tok/s | 持平（`gemv_m1` 单独 +1.4%） |

**两种取舍**：上面的 2019/8643 是**预填优先**档（批 8192 + triton + AR=0），解码在直测尺上 59.6–61.6；若更看重解码，用默认的 **FULL + flashqla + AR=0**（不换 triton），2026-10-26 生产行 4 面板九档均值 **62.28**，8K/16K/32K = **62.50/62.24/62.21**。两把尺不能相减。

读数镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15`（`v13` 由上游 vLLM 0.30.0 构建，`opt15` 是装门控包的薄层）；`16384` 批会 CUDA OOM，`gdn_tc` 必须配 triton 后端。按本仓默认两段构建产出的 `-v7` 已内置同一门控包，把上面三处参数改齐即可复现，实际读数以本机同尺为准。
逐项读数、负结果与硬件下限见 [`docs/tuning/flash-next-20261027.md`](docs/tuning/flash-next-20261027.md)。

> 下面 2026-10-07/08 与 10-20 的表是**历史面板尺**（面板 WebSocket、output 1024），与上面的直测尺不能混算。

同一台 8×T10、同一套测法（"大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）"，并发 1，九档 512→131072）：

| 配置 | 平均输出 | 平均预填充 |
|---|---|---|
| VLLM-SM75 v0.1.7-beta2（基线，已含 SM75 适配） | 43.31 tok/s | 2742.54 |
| 同一权重，叠上整组优化 · FP8PLE | 57.73 tok/s | 2749.96 |
| 叠上整组优化 · W4A16-AutoRound | 57.04 tok/s | 2762.51 |

前两行是同一枚 FP8PLE 权重；第三行是另一枚 W4A16-AutoRound（PLE 为 BF16，FP8PLE 档为 FP8，本交付精度更高）。**以第一行上游基座为基线**：同权重 FP8PLE 是 **+33.3%**（43.31→57.73，隔离整组优化的效果），历史 PIECEWISE 档是 **+31.7%**（43.31→57.04）；后者含权重差异，不能记到单一机制名下，也不是降精度换速度。当前默认已改为 FULL 图：2026-10-26 面板九档 **62.28**，比 57.04 高 **9.2%**；拿 62.28 对 43.31 粗算是 **+43.8%**，但两者不同场次，只作量级参考。

输出侧 +33.3%，预填充几乎不动。**这三张表是在面板形态上量的**，不是 GPUStack 形态的验收线；GPUStack 形态的九档读数另见 run-book 第 10a/14 节（旧九档 52.81 与本次 6 档 56.26 不是同一路径/档位）。怎么读、哪些数不能互相比，见 `gpustack/compat-matrix-v1.md` 最后一节。

> 上面的 57.73/57.04 是 2026-10-07/08 的 **SMT On** 历史面板表；下面 2026-10-20 补充是 **SMT Off**。旧 v5/v4 51.1/51.8 也是 SMT On，SMT Off A/B 待复核，不引用。

### 2026-10-20 GPUStack 形态补充（不替换上面的面板表）

同机 8×T10、SMT Off、AutoRound W4A16、output 1024、并发 1、面板直连：

| 口径 | 8192 | 16384 | 32768 | 均值 |
|---|---:|---:|---:|---:|
| **`-v6` 批次（2026-10-20；v4 栈＋运行期钉 5.15.1/0.22.2）＋A 路线** | **56.79** | **57.15** | **57.17** | **57.04** |
| 备选：旧引擎薄层后端（transformers 5.15.1）＋A 路线（image4/video0） | 56.92 | 57.57 | 56.96 | 57.15 |
| 旧引擎薄层后端、text-only，两轮 | 57.56 / 57.07 | 56.98 / 58.11 | 57.22 / 58.04 | 57.25 / 57.74 |
| v4 后端（transformers 5.17.0）panel sidecar 直连，六档 | 56.97 | — | — | 56.26 |
| v4 后端 8180 网关，四档 | 53.57 | — | — | 54.71 |
| 旧 Ultra 面板镜像 text-only 直连（独立容器），五档 | 59.35 | 58.73 | 59.43 | 58.59（五档） |

- **上表是 `-v6` 批次；当前默认 `-v7`**：同一构建链在四层链末端加层 5，把 `transformers==5.15.1`、`tokenizers==0.22.2` 成对钉住；与 v4 的完整 pip 清单差集恰为这两项。v7 同引擎层复核为 **56.83–57.04**（run-book §2 的 2026-10-21 定版说明），与历史 AutoRound 57.04 同档；不要把 v6 批次的三格与 v7 的范围当同一读数混算。
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
- 本仓新增的交付物在 `gpustack/`；上游文件受 `gpustack/gate-tree-unchanged-v1.sh` 的冻结表保护（当前基线表条目数见 `gpustack/SHA256SUMS-upstream-20261019-v2.txt`；历史 551 件见 `NOTICE`），改动需在 `NOTICE` 登记例外。
- 许可：Apache-2.0（见 `LICENSE`）；第三方件的许可与来源见 `NOTICE` 与 `gpustack/backend-version-card-v1.md`。

## 9 GPUStack 2.3.0 新特性与边界（2026-10-23 实测）

- **已实测**：离线文档 `/docs`、Anthropic 原生 `/v1/messages`、模型 YAML 导出/导入、部署 revision/回滚。
- **API 已验证、待窗口实测**：定时伸缩（只验证接受与持久化，未执行真实时间窗口）。
- **已跑通 3 请求 smoke**：智能 Benchmark 首次需拉 `gpustack/benchmark-runner:v0.0.9`；本机拉完后 3/3 成功（output 107.53 tok/s、TTFT 937 ms、TPOT 28.70 ms；单点，不代表负载曲线）。
- **当前不适用**：PD 分离（运行时常驻约 96 GB ×2 套需 16 卡；权重文件 168.8 GiB，含 CPU offload）、共享 KV（单副本）、vGPU/MIG（T10 无 MIG）、多副本负载路由。
- **升级硬要求**：worker 保留 `--runtime=nvidia --privileged --security-opt label=disable`；回退必须恢复 `postgresql/data` 备份，不能只起旧容器。
- 详细证据与未实测项见 run-book §0.1 与 `gpustack/release-notes-v2.3.0-v1.md`。

[English](README.en.md)
