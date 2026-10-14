# vLLM-SM75 v0.1.7 beta2


## 0. 本仓另有用途：GPUStack 对接线（要做这个的话，从这里开始）

本仓在上游 VLLM-SM75 之上多了一条交付线：把 SM75 版 vLLM 作为 **GPUStack 的自定义后端**部署。
要的是这个，就别按下面第 5 节「标准构建与启动」走（那是面板／Ultra 形态），直接看：

- 操作步骤（拉仓 → 探活与构建 → 起 server/worker → 注册后端版本 → 部署模型 → 四道验收）：`gpustack/run-book-v1.md`
- 表单要整块粘的两枚：`gpustack/args-paste-v1.txt`（24 行）、`gpustack/env-paste-v1.txt`（**25 行**，第 25 行 `GPUSTACK_MODEL_VRAM_CLAIM` 是硬性项，省掉模型会永久排队且界面不报错）
- 每条参数为什么这么设、换硬件怎么算：`gpustack/backend-version-card-v1.md`、`gpustack/compat-matrix-v1.md`
- 改完先自检：`bash gpustack/lint-paste-v1.sh`，末行须逐字 `LINT_PASTE_OK env=25 args=24`

本节与 `README.en.md` 的同名一节，是上游冻结基线之外本仓**有意改动**的两处，
新摘要已就地同步进 `gpustack/SHA256SUMS-upstream.txt` 对应两行，并在 NOTICE 第三节登记。

[English](README.en.md) · [发布说明](docs/releases/v0.1.7-beta2.md) · [参数模板](docs/configuration-v0.1.7.md) · [BenchLocal](docs/benchlocal.md)

## 0.5 性能实测（同一台机、同一把尺）

测试台：BenchLocal，**并发数 1、超时 30 s、提示词长度九档 512 → 131072**，全部 `1/1 并发成功`。
硬件：`G292Z20 8*T10 SM-75`。下面三张图的备注文字按测试台原文照录，未改写。

### 测试参数配置（三张图完全相同）

工具：**大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）**，即本镜像自带的测速台。

| 项 | 值 | 项 | 值 |
| --- | --- | --- | --- |
| 最小提示词长度 | 512 | 最大提示词长度 | 131072 |
| 步长 | 128 | 步长倍数 | 2（指数级：128→256→512→…1024…） |
| 期望输出长度 | **1024** | 并发数 | 1 |
| 请求超时 | 30000 ms | Temperature | 1 |
| Top P | 0.95 | Presence Penalty | 0 |
| Frequency Penalty | 0 | | |

### 裸上游基线（未打本仓任何优化层）

![上游基准](gpustack/assets/bench-upstream-baseline-w4.png)

备注 `G292Z20 8*T10 SM-75 Qwen3.8-Flash-Next-W4`｜平均总输出 **43.31 tokens/s**｜平均总预填充 2742.54 tokens/s｜ITL 22.89–23.62 ms

### 优化后模型一：FP8PLE

![FP8PLE](gpustack/assets/bench-sm75-0.17-fp8ple.png)

备注 `G292Z20 8*T10 SM-75 0.17 Qwen3.8-Flash-Next-W4 FP8PLE`｜平均总输出 **57.73 tokens/s**｜平均总预填充 2749.96 tokens/s｜ITL 17.22–17.61 ms

### 优化后模型二：W4A16-AutoRound

![W4A16-AutoRound](gpustack/assets/bench-sm75-0.17-w4a16-autoround.png)

备注 `G292Z20 8*T10 SM-75 0.17 Qwen3.8-Flash-Next-W4A16-AutoRound`｜平均总输出 **57.04 tokens/s**｜平均总预填充 2762.51 tokens/s｜ITL 17.24–18.24 ms

**这三张表怎么读（口径按用户确认，非本仓推断）**：第一张与第二张是**同一枚模型 FP8PLE**——第一张是只拉上游 `v0.1.7-beta2`、不打本仓任何优化层的裸基线，测试台备注里的 `W4` 与实际权重名不一致，以本句为准。所以 **43.31 → 57.73 是同模型直比 ＝ 输出侧 +33.3%**，而两档的平均预填充（2742.54 与 2749.96）几乎持平 ⇒ **本仓优化层的收益集中在解码侧，不在预填充侧**。第三张 `W4A16-AutoRound` 是另一枚模型，**没有配对的裸上游基线**，不要拿它跟前两张做百分比。另：第三张的 512 档（54.87、ITL 18.24）是首格冷启动离群点，别当稳态。

## 1. 简要说明

面向 NVIDIA Turing / SM75（Tesla T4、T10 等）的 vLLM 0.30.0 适配，支持 Qwen3.8 27B FP8 / AWQ / NVFP4 和 Flash Next W4A16，保留 GDN、FP8 KV、MTP、DFlash2、Firefly、CPU KV、自动休眠和监控等功能。

统一使用 **`vllm-sm75:v0.1.7-ultra-beta2`**。Ultra 提供模型管理、运行配置、对话工作台和模型测试；设置 `SM75_EDITION=standard` 可仅运行推理 API。

QQ 交流群：**878924874**

## 2. 更新说明

- **DSpark**：增加 27B FP8 / NVFP4 的 DSpark k5 模板，适配草稿 KV 分组与图捕获，支持 256K 上下文配置。
- **缓存配置**：完善前缀缓存、CPU KV 共享内存检查及 LMCache 参数提示，支持在面板和 Docker 启动参数中设置。
- **BenchLocal**：内置九类测试及全部执行依赖，默认六项；支持参数调整、逐题详情、雷达图和结果导入导出，只需构建一个 Ultra 镜像。
- **配置管理**：完善模型选择与模板启动，增加运行配置删除，优化卡片操作布局与 API 地址 / Key 复制入口。
- **监控与界面**：统一吞吐趋势的活跃采样平均值，修复字体、主题、测试状态颜色及结果布局。

## 3. 标准运行与 Ultra 功能对比

| 功能 | 标准运行 | Ultra |
|---|---|---|
| SM75 推理核心、OpenAI 兼容 API | 相同 | 相同 |
| 普通推理 / MTP / DFlash2 / DSpark | 启动参数及模板 | 面板、启动参数及模板 |
| GPU KV、CPU KV、YaRN、可选 LMCache | 启动参数 | 面板及启动参数 |
| P-State 常驻省电、自动休眠 | 环境变量 | 每份运行配置独立设置 |
| 引擎监控、投机开关 | `:8000/monitor` | 控制台集成 |
| 模型与配置管理 | 命令行 | 模型库、模板、导入导出、删除 |
| 对话、工具工作区、插件、用量 | 外部客户端 | Harness 工作台、Watcher |
| 性能测试、SQL 测试、BenchLocal | 外部工具 | 控制台模型测试 |
| 默认端口 | API 8000 | 管理 1615、API 8000 |

## 4. 性能参考

以下为已有版本的 T10 实测参考，单位 tok/s；更多条件见 [27B](docs/validation/v0.1.6-full-images.md) 与 [Flash Next](docs/flash-next-performance.md)。

| 模型与配置 | 输入 tokens | Prefill | Decode |
|---|---:|---:|---:|
| 27B FP8 · TP4 · DFlash2 draft7 · v0.1.6 Ultra | 8,192 | 1281.34 | 60.49 |
| 27B FP8 · TP4 · DFlash2 draft7 · v0.1.6 Ultra | 32,768 | 1207.62 | 63.47 |
| 27B FP8 · TP4 · DFlash2 draft7 · v0.1.6 Ultra | 131,072 | 978.08 | 58.73 |
| Flash Next W4A16 · TP8 · MTP3 · v0.1.7 beta | 8,192 | 3127.89 | 69.86 |
| Flash Next W4A16 · TP8 · MTP3 · v0.1.7 beta | 261,632 | 2108.20 | 65.08 |

## 5. 快速构建与启动

### 构建

需要 Linux x86_64、Git、Docker / BuildKit；运行机安装支持 Turing 的 NVIDIA 驱动及 NVIDIA Container Toolkit。构建需访问镜像仓库、PyPI、npm、GitHub 和编译器 / 浏览器的官方下载站，建议预留 200 GiB 构建空间。

镜像固定 vLLM 0.30.0、CUDA 12.9、Torch 2.13.0、Node 22.23.2、Harness 0.1.7-alpha.2。BenchLocal 题包、Go、Rust、浏览器及 Hermes 版本随源码锁定。

```bash
git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
EDITION=ultra MAX_JOBS=1 bash docker/build.sh
```

构建最终只输出 `vllm-sm75:v0.1.7-ultra-beta2`，包含 BenchLocal 九项测试所需环境。

已有上一版 Ultra，可复用其 CUDA / Torch / Harness 基础层：

```bash
docker build -f docker/Dockerfile.ultra \
  --build-arg BASE_IMAGE=vllm-sm75:v0.1.7-ultra-beta \
  -t vllm-sm75:v0.1.7-ultra-beta2 .
```

完整基础构建及 Unraid 构建资源隔离见 [构建文档](docker/BUILD.md)。

### 启动 Ultra

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
export VLLM_API_KEY='replace-with-your-own-long-random-key'
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1  # 替换为宿主实际路径
EDITION=ultra bash docker/run.sh

docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

打开 `http://<服务器IP>:1615`，使用生成的管理凭据登录，在模型库选择模型和配置模板。模型 API 为 `http://<服务器IP>:8000/v1`；可在运行配置的“API 地址 / Key”中复制。

启动脚本会复用 **同一个 Ultra 镜像**准备 BenchLocal 执行环境，无需额外下载或构建执行镜像。代码修复、命令行任务和 Agent 任务使用独立 CPU 容器限制资源，不挂载模型、会话或 Docker 套接字；详见 [BenchLocal 使用说明](docs/benchlocal.md)。

### 推荐参数

| 场景 | 配置 | 效果 |
|---|---|---|
| 27B FP8＋DSpark | `SM75_PROFILE=qwen38-27b-fp8-dspark5`，指定 `MODEL` / `DRAFT_MODEL` | TP4、k5、FP8 KV、256K、seq4 |
| 27B NVFP4＋DSpark | `SM75_PROFILE=qwen38-27b-nvfp4-dspark5-256k` | TP4、k5、FP8 KV、每卡 GPU KV 5 GiB |
| CPU 前缀缓存 | `CPU_KV_GIB=16`、`SHM_SIZE=32g` | 为整组 TP 设置 16 GiB CPU KV，启动前检查共享内存 |
| 常驻省电 | `POWER_MODE=pstate`，挂载匹配的 NVAPI 驱动库 | 保留模型和 KV，空闲降频，请求时恢复驱动自动状态 |
| 仅推理 API | 同一镜像设置 `SM75_EDITION=standard` | 使用相同模型模板、参数和推理核心 |

`256K` 是输入与输出合计上限，`seq4` 是共享 KV 的并发调度上限。CPU KV 用于前缀卸载与复用；GPU KV 配额仍需满足活动请求。详细参数和启动例子见 [配置模板](docs/configuration-v0.1.7.md)。

升级时沿用原 `/data`、缓存、`/dsh/home` 和 `/dsh/workspace` 挂载，保留账号、配置、会话与工作区。
