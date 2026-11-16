# Release notes · GPUStack v2.3.1 × SM75（2026-11-14）

范围：把本仓的 SM75 自定义 vLLM 后端接到 GPUStack **v2.3.1 正式版**，平台侧只换 server/worker 镜像，自定义后端版本、模型定义与凭据全部留在 data_dir 里原样可用。上一版记录见 [`release-notes-v2.3.0-v1.md`](release-notes-v2.3.0-v1.md)（2026-10-23，历史件）。

> 本件的读数与状态截至 **2026-11-16**。凡未在本机复核的项一律写在 §5 门禁表里，**不当"已修/已支持"陈述**。

## 1 版本与来源

| 项 | 值 |
|---|---|
| GPUStack | `quay.io/gpustack/gpustack:v2.3.1`，commit `458030e`（2026-11-14 本机实测） |
| 上一版 | `v2.3.0`（`63a03ba`，2026-10-23 实测）；再上一版 `v2.2.3`（**仍是本仓源码行号的取证基线**） |
| 现役容器 | server `sm75-gpustack-server-v231`、worker `sm75-gpustack-worker-v231` |
| 回退容器 | 旧 v2.3.0 的 `sm75-gpustack-server-v3` / `sm75-gpustack-worker`，**stopped 保留** |
| data_dir | server 与 worker **共用同一个** `${GS_DATA_DIR}`（升级只换镜像、不动这个目录） |
| SM75 后端镜像 | `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1`（`7267ff1c3db0`，**不变**） |
| 后端版本名 | `0.1.7-sm75.1-final-v1-custom`（**不变**） |
| 入口覆盖 | `python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`（**不变**） |
| 镜像 digest / Alembic 版本 | **本轮未现读**，不写数（见 §5） |

## 2 升级要点

1. **先 `pg_dump` 备库**。2.3.1 可能做 schema 迁移，**迁移后 2.3.0 不一定读得回来**，备份是回退的唯一前提。本机备份为一枚 SQL 导出件（约 4.3 MB，升级前导出）。
2. 停 worker 与 server（先停模型实例或把 `replicas` 调 0，确认八卡释放）；旧容器改名保留并 `docker update --restart=no`，**不要删**。
3. 拉 `quay.io/gpustack/gpustack:v2.3.1`。
4. **先升 server**（同 8180、同 data_dir、bridge ＋ 只映射 `${HOST_IP}:8180:80`、全新目录首启须显式 `--token`），等 `/healthz` 200。
5. **后升 worker**：`--runtime=nvidia --privileged --security-opt label=disable` 与 host 网络**必须整组保留**，缺整组会 `gpu_devices=[]`、调度器报 “does not support backend version”。`TZ=Asia/Shanghai` **必须显式传**，漏了容器走 UTC、日志时间戳差 8 小时。
6. `--worker-name` 保持不变 ⇒ worker 身份与模型行的 `gpu_selector` 都不用改；容器名加 `-v231` 后缀不影响功能。
7. **确认模型行仍指向我们的 `-custom` 版本**：2.3.1 捆绑的内置后端版本变了，别被默认值带跑。
8. 模型实例端口若变化，同步更新客户端 `baseURL`（端口以 `GET /v2/model-instances/{id}` 的 `port` 为准，实例重建会重分配）。
9. **回退**：停 `-v231` 两容器 → **恢复升级前的 pg_dump 备份** → 起旧 v2.3.0 容器 → `docker update --restart=unless-stopped`。**不能把旧容器直接接到已迁移的库上。**

完整命令形制见 [`run-book-v1.md`](run-book-v1.md) §0.2（升级）与 §3／§5（server/worker 起法）。

## 3 本机实测（2026-11-14 升级 → 2026-11-16 复核）

| 项 | 读数 |
|---|---|
| 管理面 | `8180/healthz` 200 |
| worker | READY，八卡均识别为 `cuda` |
| 后端与模型定义 | 原样保留（data_dir 未动）；模型行 26 升级后重调度恢复、`40000/v1/models`=200 |
| 现役生产模型 | **行 48 Swift-1.5**：served name `EXP-swift15-qwen3.8-flash-next-w4a16-autoround`，端口 **40000**，`/health` 200、一次真实推理非空 |
| 行 48 参数 | **25 项**（Swift 原配置，`--block-size=16`，未夹带任何吸收优化） |
| 行 26 | Intel AutoRound，`replicas=0` 待命，params/env 与吸收前备份**逐字全等** |
| 面板同尺速度 | 三档 8192/16384/32768、output 1024、并发 1、temp 1、topP 0.95：decode **66.61 / 66.48 / 66.94（均 66.68）**；TTFT 2030.70 / 4333.09 / 9368.64 ms；prefill 4039.33 / 3783.84 / 3498.81 tok/s |

**速度归属声明**：66.68 是**模型行 48（Swift-1.5 权重）的读数**，**不是平台升级带来的收益**。本轮未在 v2.3.1 上重跑历史九档（512→131072），因此**不得把 66.68 与 2026-10-27 的 63.30 相减当作升级效果**——两者模型权重、尺子与窗口都不同。逐项读数与现读运行时值见 [`../docs/tuning/swift15-prod-20261116.md`](../docs/tuning/swift15-prod-20261116.md)。

## 4 新特性与门槛

- **Decision Model Serving（决策模型）**：`/v1/systemone` ＋ Decision Playground，回答 `choice` / `score` / `boolean` 类型化判定，用于模型路由、分类、轻量评判。后端代码 `gpustack/routes/plugins/decision_service/`。
  **门槛：vLLM 路径须 ≥ 0.32.0** 才提供 System One 契约。**本交付的 SM75 优化版是 0.30 ⇒ 用不了。** 模型须 `Category=Decision` 才进 Playground 下拉，否则显示「暂无数据」。llama.cpp 路径用决策 GGUF 自动归类，与本交付无关。
- **CPU-offloaded 查表资源的显存估算**（上游 #6417 / #6492）：从估算 VRAM 里扣掉卸载到 CPU 的表权重（**明确包含 PLE 与 Engram**），并在参与 worker 上预留 RAM，RAM/VRAM 可分别覆盖。与本交付的 `--engram-config={"cpu_offload":true}` 直接相关 ⇒ 见 §5 的 `GPUSTACK_MODEL_VRAM_CLAIM` 门禁。
- **后端 4xx 被转成 503 的修复**（上游 #6660）：与本仓曾提的 issue #6595 同症状。**release 只写 non-streaming**，而我方复现里流式也受影响 ⇒ **须重验，见 §5**。
- 顺带修到的（上游 release 声明，本机未逐项复验）：cluster-specific image settings 被 server image configuration 覆盖；inference backend usage-check 失败被当成 not in use；extended KV cache 配置变更不触发资源重估（本交付改过 `kv-cache-memory-bytes`，值得留意）。
- 安全修复两条值得注意：模型配置求值时的服务端代码执行（GHSA-gjf9-3p3r-9h9c）、集群管理与注册 token 授权收紧（GHSA-9vwx-hfm8-g59g）。**后者可能改变 worker token 的可用操作范围**。
- 2.3.0（2026-10-23）已实测的面（离线文档 `/docs`、Anthropic `/v1/messages`、模型 YAML 导出/导入、revision/回滚）在 v2.3.1 下**未逐项重测**；PD 分离、共享 KV、vGPU/MIG、多副本负载路由在 8×T10 单副本形态下仍不适用。

## 5 已知边界与未复核门禁

| # | 项 | 状态与判据 |
|---|---|---|
| U1 | #6660「后端 4xx→503」是否真修（含流式） | **未复核**。release 只写 non-streaming；我方流式复现过。本地删 fallback target 的规避措施**先保留**，验完再决定是否恢复 |
| U2 | `GPUSTACK_MODEL_VRAM_CLAIM=120779177984` 是否仍需要 | **未复核**。v2.3.1 的 CPU-offloaded 估算可能让它变成多余，但**未对比新旧估算值前不得删**——删早了会回到「平台按启发式算出超额需求、模型一直排队且界面不给原因」的老坑（2026-10-14 实撞）。当前仍按硬性项执行 |
| U3 | Alembic/schema 版本、镜像 digest | **本轮未现读**，不写数。复核方式：只读 `docker inspect` 与 server 启动日志 |
| U4 | worker token 授权范围收紧（GHSA-9vwx-hfm8-g59g） | **未复验**。若升级后 worker 注册异常，先查这条 |
| U5 | 本仓源码行号 | 仍按上游 **v2.2.3** 取证；v2.3.0 只复核了 run-book §0.1、v2.3.1 只复核了 §0.2 的兼容面，**旧行号均未逐行复核，不要混用** |
| U6 | `description` 等元数据 | 2.3.0（2026-10-23）实测不进入部署 spec、不触发 revision、不随 spec 回滚；**v2.3.1 下未复核** |
| U7 | 定时伸缩真实窗口 | 2.3.0（2026-10-23）只验证 API 接受与持久化，真实窗口未执行（副本模型运行时会与生产争 8 卡）；**v2.3.1 下仍未执行** |
| U8 | 智能 Benchmark | 2.3.0（2026-10-23）预拉 `gpustack/benchmark-runner:v0.0.9` 后跑通 3 请求 smoke（单点，auto-tune 未跑）；该镜像是 Docker Hub 件，**离线部署需提前导入**；v2.3.1 下未重跑 |
| U9 | 内置 vLLM 版本更新 | 与本交付无关（用自定义镜像），但要确认 GPUStack 没有改掉 runner 镜像默认值（`GPUSTACK_RUNTIME_DEPLOY_CORRECT_RUNNER_IMAGE=false` 是镜像默认，保持） |

## 6 与本仓交付层的关系

- 平台升级**不改**本仓任何镜像配方：`final-v1`、`sqlprompt-audit`、`compat1` 三层的父镜像与 COPY 面均不变。
- `compat1`（Turing 0.32 兼容件薄层，全部开关默认关）**已构建、未注册、未切行**，GPU 门禁 G1–G7 未跑，不得用于生产行；见 [`../docs/tuning/compat-turing032-20261116.md`](../docs/tuning/compat-turing032-20261116.md)。
- **0.32 整栈不迁**：全配非 MTP 的面板正式验收平均输出 **59.47**，对本栈 2026-10-27 Intel 档 63.30 为 −6.05%、对 2026-11-16 Swift 档 66.68 为 −10.81%（净负）⇒ 只吸收兼容件（默认关），不整栈迁移。
