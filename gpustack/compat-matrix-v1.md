# 能力适配矩阵 · SM75 能力在 GPUStack 下的落点（任务 7）

这张表回答一个问题：上游为 SM75 做的那些能力，接到 GPUStack 上之后各自落在哪里、还剩下哪些。"怎么给"是配置位置，"结论"是在 GPUStack 形态下实测或推导的结果。源码行号取 GPUStack v2.2.3 正式版。

| 能力 | 落点 | 规格原判 | 本轮定稿 |
|---|---|---|---|
| 速度环境变量 22 条（pcieipc 卡间通信开关、QSA 量化注意力一族） | 模型 env 粘贴块 `gpustack/env-paste-v1.txt`（2.2.3 `base.py:457`：模型 env 优先于宿主 env） | 不需要改启动入口 | 能用，不用改入口 |
| MTP 投机、量化格式、上下文长度等 | 后端参数粘贴块 `gpustack/args-paste-v1.txt`（普通参数） | 不需要改启动入口 | 能用。MTP 投机解码这一版先不启用，`--speculative-config` 已从块里拿掉，以后用草稿模型的绝对路径参数行开启 |
| 时区 | env `TZ=Asia/Shanghai` ＋镜像内 tzdata 兜底 | 不需要覆盖入口 | 成立；镜像里也写了一层时区（`Dockerfile.v1` 层 3），两层各自独立生效，模型 env 的 TZ 优先（`base.py:457-458`） |
| 共享内存 64 GiB | env `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB=64`（2.2.3 `base.py:676`，缺省 10） | 不需要改启动入口 | 能用；实际值在容器里跑 `df -h /dev/shm` 看 |
| 省电常驻/自动休眠（pstate 包装脚本） | 需改启动入口＋宿主 libnvidia-api | 待服务器盘点 | **用不了。** 2026-10-11 在服务器上查过：宿主的三个标准路径里都没有这个库，只有一份私有副本，靠挂载喂给正在跑的引擎。而 GPUStack 自己起的 runner 容器没有挂东西的口子，所以"用宿主自带的库进 P8"这条路走不通，GPUStack 方案不含省电。查的过程见本文第一节。 |
| CPU KV 卸载预检 | 需改启动入口 | 跟省电项一起用不了 | 它得先能改启动入口，而下一节会说到这个口子一直是默认的。**这一条是推出来的，没有实测**；它到底能省多少、要付多少，见第三节那一行。 |
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
3. 宿主驱动已是 580.178.04，运行时注入的库会自动跟上驱动版本，而私有副本不会，所以副本本身才是脆弱点；把它再 bind 进 GPUStack runner 只是把脆弱点搬进新链路。
4. GPUStack 的 runner 容器配置由平台生成，2.2.3 的 worker 侧没有面向用户的自定义挂载字段（`gpustack/worker/backends/base.py` 内 `volumes`／`add_volume`／`custom_mount` 类关键字零命中），所以省电包装脚本要求的"入口包装＋宿主库"两条同时缺一条。若阶段二在平台侧找到未列出的挂载入口，按本节证据链复核后再决定是否重开省电项，本线不预设它存在。

结论：GPUStack 交付方案不含省电项，注册卡的入口覆盖与执行命令两栏恒为留空，不留可选段。

## 二、入口覆盖恒为默认

| 注册卡字段 | 取值 | 依据 |
|---|---|---|
| 入口覆盖 entrypoint | 留空 | 上表七项能力没有一项需要覆盖入口即成立；覆盖后 `vllm serve` 形态失效（`vllm.py:171-177`） |
| 执行命令 run_command | 留空 | 留空即沿用内置默认拼装与版本门参数；填了会进替换分支（`base.py:817-821`） |

## 三、实机门禁项（阶段二只读探针，本地不判）

| 门禁 | 探针 | 判据 |
|---|---|---|
| env 22＋3 条真进引擎进程 | 容器内 `tr '\0' '\n' < /proc/1/environ`（引擎非 PID 1 时取实际 pid） | 25 条逐一在位；面板回显不算证据 |
| shm 64 GiB | 容器内 `df -h /dev/shm` | 64 GiB；不是则 `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` 未生效 |
| 档位未被静默钳 | 引擎启动日志的 `max_model_len` | 262144；出现 8192 即参数块没进（`vllm.py:705-716`） |
| 容器时区 | 容器内 `date +%z` | `+0800` |
| CPU KV 卸载是否在位 | 容器内三处对照：`tr '\0' '\n' < /proc/<引擎 pid>/cmdline` 里数 `prefix-match-unit` 与 offload 相关参数；引擎启动日志里找 `SimpleCPUOffloadScheduler: Allocating`；`/metrics` 里的 `external_prefix_cache_hits_total` | 三种结果对应三种结论。**有 Allocating 行**：卸载层其实在位，上面那张表把它算成"用不了"是低估，要改。**参数进了命令行但没有 Allocating 行**：确实需要改启动入口，维持"用不了"。**命令行里根本没有这些参数**：参数块没进去，先查上面「档位未被静默钳」那一格。这笔能力收益和代价都量过：把 GPU KV 挤爆之后重放从 65.5 秒降到 1.5 秒（外部命中 +154,072），但常开着要付连接器开销，九格 decode 口径下约 −8.5%（53.01 对 57.96，关掉后回到 57.05，即 −1.57%）。这里的 57.96 是另一把尺的读数，与本文最后一节表里的 57.73（FP8PLE）同号不同物，不能合并也不能相减。 |
| 省电项 | 不测 | 上面已经判了用不了，不必再排探针 |

## 四、仓根那三张速度表怎么读

这三张表是在**镜像自带的 Ultra／面板形态**上跑的，不是在 GPUStack 形态上跑的。测速台是镜像自带的
"大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）"，三张图的配置完全相同：

**这一组数所凭的档**：面板／Ultra 形态的构建档，其中 `transformers` 是 5.15.1。GPUStack 交付线自 2026-10-18 起沿用基座原装的 `transformers 5.17.0`（其余四钉不变），两档只在这一点上不同。面板形态这一档没动，所以三个数对它仍然有效；但它**从来不是 GPUStack 形态的读数**，别把 57.73 说成 GPUStack 的速度，也别把两档混着引。

| 项 | 值 | 项 | 值 |
| --- | --- | --- | --- |
| 最小提示词长度 | 512 | 最大提示词长度 | 131072 |
| 步长 | 128 | 步长倍数 | 2（指数级：128→256→512→…1024…） |
| 期望输出长度 | **1024** | 并发数 | 1 |
| 请求超时 | 30000 ms | Temperature | 1 |
| Top P | 0.95 | Presence Penalty | 0 |
| Frequency Penalty | 0 | 硬件 | `G292Z20 8*T10 SM-75` |

三张图的备注文字是测速台自己打出来的原文，未改写：

![基线](assets/bench-upstream-baseline-w4.png)

`G292Z20 8*T10 SM-75 Qwen3.8-Flash-Next-W4`——平均总输出 **43.31 tokens/s**，平均总预填充
2742.54 tokens/s，每 token 间隔 22.89–23.62 ms。

![FP8PLE](assets/bench-sm75-0.17-fp8ple.png)

`G292Z20 8*T10 SM-75 0.17 Qwen3.8-Flash-Next-W4 FP8PLE`——平均总输出 **57.73 tokens/s**，
平均总预填充 2749.96 tokens/s，每 token 间隔 17.22–17.61 ms。

![W4A16-AutoRound](assets/bench-sm75-0.17-w4a16-autoround.png)

`G292Z20 8*T10 SM-75 0.17 Qwen3.8-Flash-Next-W4A16-AutoRound`——平均总输出 **57.04 tokens/s**，
平均总预填充 2762.51 tokens/s，每 token 间隔 17.24–18.24 ms。

**前两张是同一枚权重 FP8PLE 的对照。** 第一张只装 VLLM-SM75 的 `v0.1.7-beta2`，没有叠我们这一层的优化。
这一档**不是官方 vLLM 的读数**：`v0.1.7-beta2` 本身已经含 SM75 适配，而官方 vLLM 在 SM75 上我们没有测过，
所以不能把它当"官方 vLLM 的速度"。据维护者确认，这一档同时也是 P2P 没用上时的速度。
备注里的 `W4` 与权重实际名字不一致，两张表指的是同一枚 FP8PLE。

**同权重直比，平均总输出从 43.31 到 57.73 tokens/s，+33.3%**；两档平均预填充 2742.54 与 2749.96 基本没动，
所以提升集中在解码侧，不在预填充侧。

**这 33.3% 不能算在 P2P 一项上。** 基线那一档缺的东西有好几样：我们自己带的两层代码（pcieipc 八卡 PCIe 直连通信、
incple AutoRound 量化白名单），加上靠环境变量打开的底座能力（QSA 一族、`flashqla_sm75` 预填充核、自建调度器）。
这几样是一起上的，我们没有做过逐族单变量对照，所以只能记作"含 P2P 在内的整组优化"——既不能全记给 P2P，
也不能全记给我们那两层代码。

**第三张是另一枚权重 W4A16-AutoRound**，没有配对的基线档，因此不给百分比。它 512 长度那一格的
54.87（每 token 间隔 18.24 ms）是首格冷启动，不是稳态值。

**最后一条**：这三个数是面板形态的，不是 GPUStack 形态的验收线。GPUStack 形态要另测，
方法见 `gpustack/run-book-v1.md` 第 10 节。
