# 能力适配矩阵 · SM75 能力在 GPUStack 下的落点（任务 7）

口径：逐行抄录规格第四节（`design-vllm-sm75-gpustack-backend-v1.md` 第 39-53 行），"本轮定稿"列记 2026-10-11 服务器盘点后的结论。源码行号取 GPUStack v2.2.3 正式版。

| 能力 | 落点 | 规格原判 | 本轮定稿 |
|---|---|---|---|
| 速度环境变量 22 条（含 pcieipc 开关、QSA 系列） | 模型 env 粘贴块 `gpustack/env-paste-v1.txt`（2.2.3 `base.py:457` 模型 env 优先于继承的宿主 env） | 不需要覆盖入口 | 成立，无需入口覆盖 |
| MTP 投机、量化格式、上下文长度等 | 后端参数粘贴块 `gpustack/args-paste-v1.txt`（普通参数） | 不需要覆盖入口 | 成立；MTP 本线暂不启用，`--speculative-config` 已从块内剔除（用户 2026-10-11 裁决），后续用草稿模型绝对路径参数行开启 |
| 时区 | env `TZ=Asia/Shanghai` ＋镜像内 tzdata 兜底 | 不需要覆盖入口 | 成立；镜像侧时区层＝`Dockerfile.v1` 层 3（任务 5），env 与镜像两层独立生效，模型 env 的 TZ 优先（`base.py:457-458`） |
| 共享内存 64 GiB | env `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB=64`（2.2.3 `base.py:676`，缺省 10） | 不需要覆盖入口 | 成立，无需入口覆盖；实机真值用容器内 `df -h /dev/shm` 验 |
| 省电常驻/自动休眠（pstate 包装脚本） | 需入口包装＋宿主 libnvidia-api | 规格 2026-10-10 原口径「待服务器盘点」；规格现文（`:49`）已于 2026-10-11 原地改成「实测判死」，本列照原口径留痕、判死细节见右列，两份文档不再打架（修复轮 1 的 M-2） | **实测判死（2026-10-11 服务器盘点）**：宿主三标准路径全 ABSENT，只存私有副本且靠 bind 供给现役引擎；GPUStack runner 没有 bind 通道 ⇒ "宿主自带文件进 P8"这一前提不成立，GPUStack 方案不含省电项。证据：`sdd-workspace/plan-v1/server-inventory-v1.md` 第三节 |
| CPU KV 卸载预检 | 需入口包装 | 随省电项入口覆盖附带生效 | 随省电项一并落空：它的前提是"入口覆盖已启用"，而入口覆盖恒为默认（下一节），本线不提供该能力。**这是推导不是实测**，收益与代价的账见第三节「CPU KV 卸载是否在位」那一行，阶段二用秒级探针复核后再定稿（修复轮 1 的 M-7） |
| 档位模板展开（profile） | 需入口包装 | GPUStack 场景不做 | 维持不做；档位由参数粘贴块整块表达，展开器不进 GPUStack 链路 |

## 一、省电行为什么判死（证据链）

1. 宿主三个标准路径 `stat` 结果（盘点第三节原文）：

```
ABSENT /usr/lib64/libnvidia-api.so.1
ABSENT /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1
ABSENT /usr/lib/libnvidia-api.so.1
在 /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu 下 maxdepth 4 find 'libnvidia-api*' = 零命中
```

2. 全盘定位只找到私有副本五枚，逐字节同哈希（sha 前 12 `4a199f9b259a`），现役容器里那份也来自宿主 bind，不是宿主标准路径。
3. 宿主驱动已是 580.178.04，运行时注入的库会自动跟上驱动版本，而私有副本不会 ⇒ 副本本身才是脆弱点；把它再 bind 进 GPUStack runner 只是把脆弱点搬进新链路。
4. GPUStack 的 runner 容器配置由平台生成，2.2.3 的 worker 侧没有面向用户的自定义挂载字段（`gpustack/worker/backends/base.py` 内 `volumes`／`add_volume`／`custom_mount` 类关键字零命中）⇒ 省电包装脚本要求的"入口包装＋宿主库"两条同时缺一条。若阶段二在平台侧找到未列出的挂载入口，按本节证据链复核后再决定是否重开省电项，本线不预设它存在。

⇒ 结论：GPUStack 交付方案不含省电项，注册卡的入口覆盖与执行命令两栏恒为留空，不留可选段。

## 二、入口覆盖恒为默认

| 注册卡字段 | 取值 | 依据 |
|---|---|---|
| 入口覆盖 entrypoint | 留空 | 上表七项能力没有一项需要覆盖入口即成立；覆盖后 `vllm serve` 形态失效（`vllm.py:171-177`） |
| 执行命令 run_command | 留空 | 留空即沿用内置默认拼装与版本门参数；填了会进替换分支（`base.py:817-821`） |

## 三、实机门禁项（阶段二只读探针，本地不判）

| 门禁 | 探针 | 判据 |
|---|---|---|
| env 22＋2 条真进引擎进程 | 容器内 `tr '\0' '\n' < /proc/1/environ`（引擎非 PID 1 时取实际 pid） | 24 条逐一在位；面板回显不算证据 |
| shm 64 GiB | 容器内 `df -h /dev/shm` | 64 GiB；不是则 `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` 未生效 |
| 档位未被静默钳 | 引擎启动日志的 `max_model_len` | 262144；出现 8192 即参数块没进（`vllm.py:705-716`） |
| 容器时区 | 容器内 `date +%z` | `+0800` |
| CPU KV 卸载是否在位（修复轮 1 的 M-7 新增） | 容器内 `tr '\0' '\n' < /proc/<引擎 pid>/cmdline` 数 `prefix-match-unit` 与 offload 相关参数，再在引擎启动日志里找 `SimpleCPUOffloadScheduler: Allocating` 这一行，以及 `/metrics` 的 `external_prefix_cache_hits_total` | 三段判据：①日志有 Allocating 行 ⇒ 卸载层在位，首表把它写成"随入口覆盖落空"是低估，须回改右列；②参数进了 argv 但日志无 Allocating 行 ⇒ 入口包装确有必要，维持"落空"；③argv 里根本没有相关参数 ⇒ 参数块没进，先查「档位未被静默钳」那一格。方向要说清：这笔能力**收益与代价都有实测**——挤爆 GPU KV 后重放 65.5 s→1.5 s（外部命中 +154,072），但九格 decode 口径下连接器开销约 −8.5%（53.01 对基准 57.96，关掉后回到 57.05＝−1.57%），所以"落空"是两头一起落空、不是单方面损失。为什么单列这一格：首表把它随省电项划成"落空"是**推导**而非实测（首表已如实标注），本行就是推翻或坐实该推导的秒级只读探针。出处＝本机过程件 `调试档案-20261009-多模态与262K/结论件-CPUKV三档-v6.md` 第 7 节、`调试档案-20261010-启动日志解读与速度归因/报告-mm2e-关闭CPU卸载-v1.txt` 第 97-101 行（未随包发布，留作后续会话回查） |
| 省电项 | 不提供探针 | 已判死，阶段二不排这一格 |
