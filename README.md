# vLLM-SM75 v0.1.7 beta2


## 0. 在 GPUStack 里用它

这个仓把 SM75 版 vLLM 做成了 GPUStack 的自定义后端，让 T4 / T10 这类老卡能在 GPUStack 里跑大模型。
官方 vLLM 里 Turing 没有加速路径，Marlin 和 FP8 W8A8 都不支持，7.5 只是它肯跑起来的最低门槛。
上游 VLLM-SM75 做的是 SM75 适配。我们在它之上带了两层代码——八卡 PCIe 直连通信（pcieipc）和
AutoRound 量化白名单（incple），又用一组环境变量把底座的 QSA、FlashQLA、自建调度打开。
下面 0.5 量的就是这一整组加上去的效果。

前提：8 张 16 GiB 卡 ＋ 251 GiB 主机内存。要跑的档权重 168.7 GiB，比显存总和大，靠主机内存补，
所以内存不够会在加载阶段被杀，现象看起来像镜像坏了。

操作看 `gpustack/run-book-v1.md`，从头做到尾：构建镜像 → 起 server 和 worker → 注册后端版本 →
部署模型 → 四道验收。表单里要整块粘的两枚参数是 `gpustack/args-paste-v1.txt`（24 行）和
`gpustack/env-paste-v1.txt`（25 行，第 25 行的显存配额是硬性项，漏了模型会一直排队且不报错）。
粘完跑一次 `bash gpustack/lint-paste-v1.sh`，末行出现 `LINT_PASTE_OK env=25 args=24` 才算粘齐。

每条参数为什么这么设、换硬件怎么重算，看 `gpustack/backend-version-card-v1.md` 和
`gpustack/compat-matrix-v1.md`。

[English](README.en.md) · [发布说明](docs/releases/v0.1.7-beta2.md) · [参数模板](docs/configuration-v0.1.7.md) · [BenchLocal](docs/benchlocal.md)

## 0.5 速度

同一台 8×T10，同一套测法（镜像自带的 BenchLocal，并发 1，提示词九档 512→131072，每档 1/1 成功）：

| 配置 | 平均输出 | 平均预填充 |
|---|---|---|
| VLLM-SM75 v0.1.7-beta2（基线，已含 SM75 适配） | 43.31 tok/s | 2742.54 |
| 同一权重，叠上上面那一整组 · FP8PLE | 57.73 tok/s | 2749.96 |
| 叠上这一整组 · W4A16-AutoRound | 57.04 tok/s | 2762.51 |

输出侧 +33.3%，预填充几乎没动，提升在解码这一段。这三张表怎么读、哪些数不能互相比、
以及它们为什么不是 GPUStack 形态的验收线，见 `gpustack/compat-matrix-v1.md` 最后一节。



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
