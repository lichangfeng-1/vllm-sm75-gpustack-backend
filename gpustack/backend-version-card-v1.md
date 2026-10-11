# GPUStack 自定义后端注册卡 · vllm-sm75-gpustack（任务 7）

一句话：在 GPUStack 2.2.3 上注册一枚自定义 vLLM 后端，把 SM75 八卡档位原样带进平台，入口覆盖与执行命令都留空。

本文所有源码行号一律取 **GPUStack v2.2.3 正式版源码**（本地 `gpustack源码/v2.2.3-正式`）；开发态检出那套树行号不同，别混用。

## 一、注册表单逐项

| 字段（界面名 ／ 接口键） | 填什么 | 为什么 |
|---|---|---|
| 后端类型 type | 自定义（custom） | 不走内置镜像目录，镜像由本仓构建 |
| 版本名 backend_version | `0.1.7-sm75.1-custom` | 数字形态＋`-custom` 后缀。GPUStack 用 `should_default_to_ray`（`gpustack/utils/command.py:416-433`，阈值常量 `:413`＝`0.18.0`）选分布式执行器：版本名不含 `-custom` 一律回落 Ray，而标准镜像不带 Ray；解析抛异常也回落 Ray（`except Exception: return True`）。加 `-custom` 就是把"镜像里没有 Ray"告诉平台。单 worker 部署不进该分支（`gpustack/worker/backends/vllm.py:261/280` 需要 distributed_follower／distributed_leader 才触发），所以本卡只承诺单机 |
| 镜像名称 image_name | `vllm-sm75-gpustack:v0.1.7-beta2-v1` | 完整 tag 写进这一个字段，不拆成"仓库＋标签"两处 |
| 框架 custom_framework | `CUDA` | 该字段用于部署与兼容性检查（`gpustack/schemas/inference_backend.py:41`，字段声明 `:49`） |
| 入口覆盖 entrypoint | 留空 | 七项能力的落点全部不需要覆盖入口（见 `compat-matrix-v1.md`）；入口一改，`vllm serve` 形态即失效（`gpustack/worker/backends/vllm.py:171-177`） |
| 执行命令 run_command | 留空 | 留空＝沿用 GPUStack 内置默认命令与版本门参数；填了才进替换分支（`gpustack/worker/backends/base.py:817-821`），内置拼装被顶掉 |

## 二、执行命令默认模板（留空时 GPUStack 实际生成的形态）

容器命令 `vllm serve`（`gpustack/worker/backends/vllm.py:171`），参数序列（`vllm.py:593-659`）：

```
vllm serve  <模型路径>  <GPUStack 版本门参数>  <后端参数粘贴块>  --host <worker IP>  --port <端口>  --served-model-name <对外模型名>
```

- 模型路径是第一个位置参数（`vllm.py:613`），所以粘贴块里没有 `${MODEL}`：GPUStack 对后端参数不做模板渲染，占位符会字面直达引擎（审计 A-01）。渲染只在 run_command 与 env 取值两条路上发生（`gpustack/schemas/inference_backend.py:201-232`），后端参数走纯切分（`gpustack/worker/backends/base.py:1138-1152`），所以 `${MODEL}` 和 `{{model_path}}` 写进参数块都是字面量。
- 尾补三条由 `extend_args_no_exist` 只在用户没写时补（`vllm.py:705-710`），粘贴块里同样不写，写了两遍即重复参数。
- `--max-model-len` 必须由粘贴块提供：用户参数里没有它时，平台按推导值注入 8192（`vllm.py:705-716` 段的 max-model-len 分支，审计 B-04），262144 档会被静默砍到 8192。
- 若一定要显式填执行命令，可用模板变量只有 `{{model_path}} {{port}} {{worker_ip}} {{model_name}} {{gpu_count}} {{gpu_ids}}`（正式版逐串替换的实现见 `gpustack/schemas/inference_backend.py:201-227`），写成 `vllm serve {{model_path}}` 与默认同形。注意这里合法的是双花括号 `{{ }}`，参数块里出现的必须是零个 `{{`（占位符未回填即字面直达引擎）。

## 三、三句声明（逐字，部署时照此执行）

- 仅单机部署，分布式未验证
- 部署对话框不启用投机解码
- MTP 暂不支持，后续可以草稿模型绝对路径参数行开启

第二句的机制原因：UI 的投机解码注入位排在用户参数之前（`vllm.py:531-563`、`:627`），勾选后与参数块同族参数重复。第三句的落地路径：后续在"后端参数"里另起一行写草稿模型的绝对路径，形如 `--speculative-config={"method":"mtp","num_speculative_tokens":3,"model":"<草稿模型绝对路径>"}`；注意 GPUStack 的模型目录里没有 `runtime/mtp-int4-g32`，唯一一份在 `/data/8xt10-test/models/fp8ple/runtime/mtp-int4-g32`（盘点第五节＋风险清单 R6），要用得先把 GPUStack 模型来源指向该仓。

## 四、部署对话框逐项

| 字段 | 填什么 |
|---|---|
| 后端版本 | `0.1.7-sm75.1-custom` |
| 后端参数 | 整块粘贴 `gpustack/args-paste-v1.txt`（24 行，一行一条） |
| 环境变量 | 整块粘贴 `gpustack/env-paste-v1.txt`（24 行，一行一条） |
| 投机解码 | 不启用 |
| 副本数／worker | 1 台 8 卡 worker，`--tensor-parallel-size=8` 已在参数块里 |

## 五、手改后复检

```
bash gpustack/lint-paste-v1.sh
```

- 做什么：把两枚粘贴块再跑一遍结构护栏（env 形态＋键唯一＋恰 24 行；args 形态＋无尾补三条＋无占位符＋无投机＋flag 唯一＋`--max-model-len=262144` 在列）。
- 为什么：注册卡与教程里的块是人手复制的，生成器那道门只保护生成那一次。
- 预期输出：两行"绿｜"加末行 `LINT_PASTE_OK env=24 args=24`，退出码 0。
- 失败怎么办：按文案点名的那道门改回粘贴块本身，不要改护栏脚本；退出码 14＝块里有 GPUStack 会尾补的参数，15＝env 键重复，16＝内容断言红，2＝文件或行尾问题（含 CR 时按 `git -c core.autocrlf=false clone` 重新检出）。

## 六、阶段二门禁项（本地判不了，实机用只读探针）

| 门禁 | 秒级探针 | 判据 |
|---|---|---|
| 镜像在服务器可用 | `docker images vllm-sm75-gpustack` | 有该 tag；否则先 load |
| 容器按 GPUStack 形态起来 | GPUStack 实例页看容器状态＋`docker ps` | host 网络、特权、`--shm-size=64g`、restart 不自动拉起 |
| shm 真值 | 容器内 `df -h /dev/shm` | 64 GiB；不是则 `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` 未生效 |
| env 真值 | 引擎进程 `/proc/<pid>/environ` | 24 条逐一在位；面板回显不算证据 |
| 档位未被钳 | `GET /v1/models` ＋引擎启动日志的 max_model_len | 262144；出现 8192 即参数块没进 |
| 时区 | 容器内 `date +%z` | `+0800` |
| 对外名回显 | `GET /v1/models` | 与部署时模型名一致 |
| 省电项 | 无探针 | 已实测判死，GPUStack 方案不含（见 `compat-matrix-v1.md`） |
