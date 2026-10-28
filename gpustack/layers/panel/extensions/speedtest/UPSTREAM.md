# 测速扩展上游同步记录

- 上游仓：`gengchaogit/llm_speedtest`
- 本次对齐提交：`07895cffd90bb9ff4847695ed7f90d097832518c`（`update gpu type`，2026-10-21T01:10:19Z）
- 取件方式：`gh api repos/gengchaogit/llm_speedtest/contents/<path>?ref=07895cff…`（GitHub API，raw 媒体类型；递归树 `truncated=false`）
- 上游版本：v3.5（上一版为 v3.0 / 快照提交 `eb199403`，2026-09-13）
- 记录日期：2026-10-28

## 1 文件对照（09-13 快照 → 07895cff）

| 文件 | 旧 sha256 | 新 sha256 | 旧→新 字节 | 行数 |
|---|---|---|---|---|
| python/LLM_Speed_Test_v3_Python_Backend.html | e15781a3…b75 | 63294c07…824 | 410,340 → 714,025 | 8,032 → 12,781 |
| LLM_Speed_Test_v3_Leaderboard.html | be5fde56…515 | 9fbed5db…4d8b | 201,637 → 585,305 | 5,011 → 11,393 |
| python/llm_test_backend.py | 18b7b1d3…178a | 0683534…8870 | 99,407 → 237,296 | 2,304 → 5,368 |
| python/requirements.txt | — | — | 50 → 53（新增 `h2`） | 6 → 7 |
| python/README_Backend.md | — | — | 11,303 → 16,056 | 253/287 行改写 |
| supabase_get_model_tags.sql | — | — | 1,019 → 1,019（逐字节相同） | 不变 |

包内 v20261021 文件与该表新 sha256 一致；旧文件保持原位，未覆盖。

## 2 上游变更摘要

两版之间共 12 个提交，主体为 `c0619ddf`（v3.5 大版本）。主要功能面：

- 测速模式：一维扫频之外新增 **2D 并发×上下文矩阵** 与 **稳态持续解码** 两种模式（Python 页新增 93 个 DOM 编号、32 个函数：`runAllTasksMatrixSequence`、`handleMatrixCellResult`、`renderMatrixHeatmapSkeleton`、`runAllTasksSustainedSequence` 等）。
- 任务类型：新增 `general / code / structured / prose / prefill_16k / reasoning` 任务集与对应系统提示词（`TASK_TYPE_CHOICES`、`generate_task_prompt`、`get_system_prompt_for_task`）。
- 熔断：新增流式周期重复检测 `StreamingLoopDetector`（死循环自动中止）。
- 后端：新增 CLI 模式（`--cli`）、Prometheus `/metrics` 抓取（`parse_prometheus_metrics`、`detect_hardware`）、校验哈希与直传 Supabase（`compute_verification_hash`、`upload_benchmark_to_supabase`）；HTTP/2 连接池（`httpx.AsyncClient(http2=True)`）。路由集合不变（11 条，增删 0）。
- 排行榜：新增帕累托前沿、六维雷达、2D 矩阵透视、贡献者榜与筛选胶囊（新增 183 个 DOM 编号、87 个函数）；移除 MyMemory/Google 翻译外链，改纯本地双语；Chart.js 由浮动版本固定为 `4.4.1`。
- 两个 HTML 的 DOM 编号**只增不减**（Python 页 138→231、排行榜 66→249），旧编号全部保留。

## 3 本地集成与合并结果

包内两份 HTML 与 09-13 上游原件逐字节相同；本地集成分三层，均不改页面字节：

| 本地项 | 位置 | 合并结果 |
|---|---|---|
| 面板代理注入外观/凭据脚本 | `ultra/source/console/original-tests.mjs` | 依赖 `<head>`：新页存在；注入幂等标记不冲突 |
| 默认参数覆盖（512 / 131072 / 128 / 2 / 512 / 30000 / 1 / 0.1 / 0 / 0） | 同上，`let d_<name>` 与 `const savedMinLength` 精确替换 | 10 个 `d_` 变量与 `savedMinLength` 在新页各命中 1 次 |
| 引擎地址/模型名/密钥注入 | 同上：`backendUrlInput.value = \`ws://${host}:${port}\`;`、`d_openai_apiUrl`、`d_openai_modelName`、`d_api_key` | 新页各命中 1 次 |
| 采样参数注入 | `public/benchmark-sampling.js`，依赖 `temperatureInput/topPInput/presencePenaltyInput/frequencyPenaltyInput/modelName/resetParams` | 6 个编号在新页均存在 |
| 结果采样脚本挂载 | 代理在 `</body>` 前插入 | `</body>` 命中 1 次 |
| 响应式宽度（v11 薄层） | `gpustack/layers/panel-responsive/public/speedtest.html`，仅 2 行 CSS | 已按同两行派生新件，`diff --strip-trailing-cr` 断言仅这 2 行不同 |
| 品牌 | `sm75-brand` 面板壳插件，不注入页面 | 新旧页均无 `SM75` 字样，无需合并 |
| 面板入口与就绪标记 | `manifest.json` 的 `original_ui=1`；服务端固定读取 `LLM_Speed_Test_v3_Python_Backend.html` / `llm_test_backend.py` | 新件带版本后缀，部署时切换（见 §5） |
| 面板自动基准运行器 | `benchmark-runner.py` 导入 `execute_single_request` 并改写 `httpx.AsyncClient.stream` | 新签名仅在尾部追加可选 `client`，前 11 个位置参数不变；`httpx` 模块属性保留 |

结论：新版本可直接承接现有全部本地集成，无需改动面板代码。

## 4 新增文件

| 文件 | 用途 | sha256 |
|---|---|---|
| `LLM_Speed_Test_v3_Python_Backend-v20261021.html` | 新版测速页（上游原样） | `63294c075ae7…090824` |
| `LLM_Speed_Test_v3_Leaderboard-v20261021.html` | 新版排行榜（上游原样） | `9fbed5db2c2b…a3d8b` |
| `llm_test_backend-v20261021.py` | 新版后端（上游原样） | `06835345c2d7…a078870` |
| `manifest-v20261021.json` | 供部署时替换 `manifest.json`（commit/sha/original_ui） | — |
| `deps/h2`、`deps/hpack`、`deps/hyperframe`（含 dist-info） | 新后端 `http2=True` 的必需依赖（纯 Python） | 见 `deps-h2-20261021.sha256`（46 件） |
| `deps-h2-20261021.sha256` | 上述依赖逐文件哈希清单 | — |
| `gpustack/layers/panel-responsive/public/speedtest-v20261021.html` | 新版 + 2 行响应式 CSS（CRLF），供 v12 薄层使用 | `fe139724183c…085048c` |

依赖版本：h2 4.4.1、hpack 4.2.0、hyperframe 6.1.0（均为 py3-none-any，无编译件）。旧后端不导入这些模块，新增不影响旧版。

## 5 部署映射（阶段 2，本阶段未执行）

1. 备份：`LLM_Speed_Test_v3_Python_Backend.html`、`LLM_Speed_Test_v3_Leaderboard.html`、`llm_test_backend.py`、`manifest.json`、`panel-responsive/public/speedtest.html`。
2. 扩展目录：三件 v20261021 副本复制为规范名；`manifest-v20261021.json` 复制为 `manifest.json`（`original_ui` 必须保持 1，避免触发在线安装）。
3. 响应式薄层：`speedtest-v20261021.html` 复制为 `speedtest.html`（或把 Dockerfile 指向新文件名）。
4. 重建面板层并在镜像内复核 sha（沿用 `build-panel-responsive-v1.sh` 的断言方式），再按里程碑报主控。
5. 回退：还原上述备份即可；版本化副本始终保留。

## 6 已知边界

- 排行榜的数据来自浏览器直连 Supabase（`cvezaerrczywfzqcaqmx.supabase.co`，上游地址）与 jsDelivr Chart.js，内网无外网时页面可打开但无数据；旧版同样如此。
- 上游新增的独立 `index.html`（2D 矩阵免部署版，779,199 B）未纳入本包，面板仍以 Python 后端页为唯一测速入口。
- 新页新增 `llmPerfTestTaskType` 本地键；面板 `speedResetScript` 只清理 API 地址/密钥，不涉及该键。
