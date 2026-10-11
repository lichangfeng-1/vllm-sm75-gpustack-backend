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
- 尾补三条由 `extend_args_no_exist` 只在用户没写时补（正式版 `gpustack/worker/backends/vllm.py:654-659`；修复轮 1 的 M-3 把此处原来误写的 `vllm.py:705-710` 改到真实位置——`:705` 起的其实是 max-model-len 那个函数，见下一条），粘贴块里同样不写，写了两遍即重复参数。
- `--max-model-len` 必须由粘贴块提供：用户参数里没有它时，平台按推导值注入 8192（`gpustack/worker/backends/vllm.py:705-716` 的 `_build_max_model_len_arguments`，判定在 `:708-712`、注入在 `:714-715`，审计 B-04），262144 档会被静默砍到 8192。
- 若一定要显式填执行命令，可用模板变量只有 `{{model_path}} {{port}} {{worker_ip}} {{model_name}} {{gpu_count}} {{gpu_ids}}`（正式版逐串替换的实现见 `gpustack/schemas/inference_backend.py:201-227`），写成 `vllm serve {{model_path}}` 与默认同形。注意这里合法的是双花括号 `{{ }}`，参数块里出现的必须是零个 `{{`（占位符未回填即字面直达引擎）。

## 三、三句声明（逐字，部署时照此执行）

- 仅单机部署，分布式未验证
- 部署对话框不启用投机解码
- MTP 暂不支持，后续可以草稿模型绝对路径参数行开启

第二句的机制原因：UI 的投机解码注入位排在用户参数之前（`vllm.py:531-563`、`:627`），勾选后与参数块同族参数重复。第三句的落地路径：后续在"后端参数"里另起一行写草稿模型的绝对路径，形如 `--speculative-config={"method":"mtp","num_speculative_tokens":3,"model":"<草稿模型绝对路径>"}`；注意 GPUStack 的模型目录里没有 `runtime/mtp-int4-g32`，唯一一份在 `/data/8xt10-test/models/fp8ple/runtime/mtp-int4-g32`（盘点第五节＋风险清单 R6），要用得先把 GPUStack 模型来源指向该仓。

## 四、部署对话框逐项

取得两枚粘贴块只用这一条形：`git -c core.autocrlf=false clone <url>`（`<url>` 替换为你的地址）。这不是风格问题，原因写在 §五 末「为什么偏偏这两枚 `.txt` 块会中招」。

| 字段 | 填什么 |
|---|---|
| 后端版本 | `0.1.7-sm75.1-custom` |
| 后端参数 | 整块粘贴 `gpustack/args-paste-v1.txt`（24 行，一行一条，**行内任何位置不得含空格或制表**；值一律等号形，JSON 值必须写成 `--flag={…}` 紧贴等号，不许写 `--flag {…}`） |
| 环境变量 | 整块粘贴 `gpustack/env-paste-v1.txt`（24 行，一行一条 `KEY=VALUE`，键与值两侧、值内都不得含空格或制表） |
| 投机解码 | 不启用 |
| 副本数／worker | 1 台 8 卡 worker，`--tensor-parallel-size=8` 已在参数块里 |

这三条写法规矩为什么是硬性的（修复轮 1 补，此前只在护栏文案里写过，属隐性约束）：GPUStack 会把表单里的每一行再用 shlex 切一遍（正式版 `gpustack/worker/backends/base.py:1138-1152` `_flatten_backend_param` → `gpustack/utils/command.py:338-351` `safe_split`），而 JSON 掩码**只在 `=` 紧跟 `{` 或 `[` 时才触发**（`gpustack/utils/command.py:226-301` `_mask_json_segments`，触发条件在 `:245`）。所以从 vLLM 官方文档或别处档位抄来的空格形 `--engram-config {"cpu_offload":true}` 会拆成两个 token、值退化成 `{cpu_offload:true}`（非法 JSON）；引擎不一定报错，只是把这一项吃掉——静默降能。抄完先跑 §五 的护栏，这类形状会被 L5 直接判红。

## 五、手改后复检

```
bash gpustack/lint-paste-v1.sh
```

- 做什么：把两枚粘贴块再跑一遍结构护栏。门序 L0-L13：前置（存在性／CR／末行换行）→ env 行形态 → env 键唯一 → env 占位符 → env 行数==24 → args 行形态 → 尾补三条排除 → args 占位符 → args 投机 → args flag 唯一 → 关键项在列且 `--max-model-len=262144` → **L11 args 行数==24** → **L12 env 与脚本内置常量表逐行逐字同序** → **L13 args 同上**。改值、乱序、增删行、原地留一行空行都会点名到具体行号。
- 为什么：注册卡与教程里的块是人手复制粘贴的，生成器那道门只保护生成那一次；表单里那一份没有第二道锁。
- 三条写法规矩（与 §四 同口径，护栏判死的面）：①一行一条，行内任何位置不得含空格或制表；②值一律等号形，JSON 值必须 `--flag={…}` 紧贴等号；③空格形会被 shlex 拆成两个 token、JSON 值当场坏且不报错，机制见 `gpustack/utils/command.py:226-301`（触发条件 `:245`）与 `:338-351`。
- 预期输出：两行"绿｜"加末行 `LINT_PASTE_OK env=24 args=24`，退出码 0。
- 失败怎么办：按文案点名的那道门改回粘贴块本身，不要改护栏脚本；退出码 14＝块里有 GPUStack 会尾补的参数，15＝env 键重复，16＝内容断言红（含 L11-L13 的行数与逐字有序），2＝文件或行尾问题（含 CR 时按下面的克隆形重新检出）。
- 确要改档（改值／增删行／换顺序）时**两处一起改**：粘贴块本身，与 `gpustack/lint-paste-v1.sh` 里的 `ENV_EXPECT`／`ARGS_EXPECT` 常量表；只改一处则 L12/L13 必红——这是设计而不是故障。args 侧还要重跑 `python gpustack/gen-args-paste-v1.py` 确认产物与块一致（生成器的 A1 与 lint 的 L5 同形，含空白的形状在生成侧就被判死）。

### 为什么偏偏这两枚 `.txt` 块会中招

仓内 `.gitattributes` 给 `*.sh`／`*.py`／`*.md`／`*.json`／`Dockerfile*` 等显式写了 `text eol=lf`，**唯独没有 `*.txt` 规则**；而 `.gitattributes` 属交接约束第 1 条的 14 个上游禁写面，本线不能替它补一条。于是在 Windows 上用默认 `core.autocrlf=true` 检出时，只有 `gpustack/env-paste-v1.txt` 与 `gpustack/args-paste-v1.txt` 各 24 行的行尾被换成 CRLF（同批检出的 `.sh`／`.md` 不受影响），而 CRLF 块贴进表单＝每行尾部多带一个 CR。护栏对它是**响亮失败**：L0 前置直接退 2 并打印 CR 个数，不会静默放行；唯一修法是用钦定形重新检出——`git -c core.autocrlf=false clone <url>`（`<url>` 替换为你的地址）。

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
