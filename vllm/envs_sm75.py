# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM75 扩展 env —— 对上游 vllm/envs.py 的纯增量, 不再整文件覆盖。

底座保留上游原样 vllm/envs.py; 本模块只装 SM75 自定义的 env getter,
构建时由 install_sm75_overlay.py 把一行 `import vllm.envs_sm75; apply()`
append 到上游 envs.py 尾部触发注入。注入做两件事:

1) 把 EXTENSIONS getter 灌进 vllm.envs.environment_variables —— 上游
   __getattr__ / __dir__ / is_set / validate_environ / enable_envs_cache
   全围绕该 dict, 灌入后 envs.VLLM_FIREFLY 等属性访问自动生效(无需改调用方)。
2) 包一层 vllm.envs.compile_factors, 把 INSTALL_IGNORED(idle auto-sleep 计时器)
   从 hash factors pop 掉，并加入固定的 overlay 编译版本，隔离后端/权重布局变更。

好处: 上游 envs.py 后续升级随便改(内容级), 本文件只跟「dict + __getattr__
+ compile_factors 返回 dict」这个稳定机制耦合, 冲突面从整文件缩到一处注入。
"""

import os

# Internal release fingerprint, not an environment setting. Bump when overlay
# changes can select a different kernel or packed-weight layout before tracing.
_COMPILE_REVISION = "0.1.6-1"

# 需从 compile_factors hash 中排除的 key: idle auto-sleep 计时器/路径只影响
# 调度与 checkpoint bookkeeping, 不改变编译图。切 test/prod sleep 计时不应
# 使 torch.compile 缓存失配。
INSTALL_IGNORED = {
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT",
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET",
    "VLLM_AUTO_SLEEP_RELOAD_PATH",
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL",
}


def _firefly_mode() -> str:
    """VLLM_FIREFLY 归一化: '1'=开(=auto) / '0'=关。

    未设默认 '0'(不激活); '1' 与 'auto' 等价(都算开); 其余值一律 '0'。
    开 = int4(AWQ/GPTQ) 走 int8 加速, fp8 走上游 marlin(fp8 加速走
    VLLM_FIREFLY_AR fp8 allreduce, 另见 PLAN-fp8-allreduce)。
    """
    v = os.getenv("VLLM_FIREFLY", "").strip().lower()
    return "1" if v in ("1", "auto", "on", "true", "yes") else "0"


def _firefly_ar_mode() -> bool:
    """VLLM_FIREFLY_AR 归一化: 默认关 (fail-closed), 仅显式开启才生效。

    用户红线 (2026-11-16): fp8 集合通信仅入库代码、默认关、永不默认启用。
    1/on/true/yes/fp8 = 开; 其余 (含未设与旧值 'auto') = 关。旧 'auto'
    (跟随 VLLM_FIREFLY) 语义已废除 —— firefly 权重路径开着也不会带开 AR。
    'fp8' 仍接受以兼容旧配置写法。FireflyAllReduce 只做 fp8, 双 backend
    (P2P / SHM, 见 VLLM_FIREFLY_AR_BACKEND)。
    """
    v = os.getenv("VLLM_FIREFLY_AR", "").strip().lower()
    return v in ("1", "on", "true", "yes", "fp8")


def _firefly_direct() -> bool:
    """VLLM_FIREFLY_DIRECT (0.32 生态名): 默认跟随 VLLM_FIREFLY; 显式覆盖优先。

    int4/fp8 权重 firefly 路径的门控; 与 0.32 包代码同名列兼容。未设时
    跟随 VLLM_FIREFLY ('1'/'auto' = 开), 显式 0 关 / 1 开。
    """
    v = os.getenv("VLLM_FIREFLY_DIRECT", "").strip().lower()
    if v in ("1", "auto", "on", "true", "yes"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    return _firefly_mode() == "1"


def _env_flag(name: str) -> bool:
    """严格布尔: 仅 1/on/true/yes 开; 未设/其余一律关 (fail-closed)。"""
    return os.getenv(name, "").strip().lower() in ("1", "on", "true", "yes")


def _firefly_skillet_spk() -> int:
    """VLLM_FIREFLY_SKILLET_SPK: split-K 因子, 仅接受 2/4/8/16, 其余回 8。"""
    try:
        v = int(os.getenv("VLLM_FIREFLY_SKILLET_SPK", "8").strip())
    except ValueError:
        return 8
    return v if v in (2, 4, 8, 16) else 8


def _firefly_idec_max_m() -> int:
    """VLLM_FIREFLY_IDEC_MAX_M: idecode GEMV 的 batch 上限, 只在 1..8 有效。"""
    try:
        v = int(os.getenv("VLLM_FIREFLY_IDEC_MAX_M", "8").strip())
    except ValueError:
        return 8
    return v if 1 <= v <= 8 else 8


def _firefly_ar_backend() -> str:
    """VLLM_FIREFLY_AR_BACKEND 归一化: 'auto'(默认, 运行时按 _can_p2p 选 P2P
    优先) / 'p2p'(强制 P2P) / 'shm'(强制 SHM)。

    auto = 有 P2P (NVLink/PCIe 直连) 选 P2P (data/flag 全 device 显存, 无 host
    bounce); 无 P2P (PHB, 如 T10) 回 SHM (/dev/shm + cudaHostRegister)。
    'p2p'/'shm' 强制指定 (P2P 初始化失败仍自动回 SHM 兜底)。
    """
    v = os.getenv("VLLM_FIREFLY_AR_BACKEND", "").strip().lower()
    if v in ("p2p",):
        return "p2p"
    if v in ("shm", "host", "shared"):
        return "shm"
    return "auto"


def _monitor() -> bool:
    """VLLM_MONITOR 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /monitor 挂单文件 HTML 监控页(自拉同源 /metrics 渲染, 无 CDN,
    见 entrypoints/serve/instrumentator/monitor.py); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_MONITOR", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


# SM75 自定义 env → getter。install 时灌进 vllm.envs.environment_variables,
# 之后 envs.<NAME> 属性访问 / is_set / validate_environ / __dir__ 自动生效。
EXTENSIONS: dict[str, object] = {
    # firefly(SM75) prefill 加速总开关, 默认关(_firefly_mode 归一化):
    #   未设 / 0 = 关(全走上游 marlin, 默认不激活)。
    #   1 / auto = 开(两者等价): int4(AWQ/GPTQ, W4A16) 走 int8 加速; fp8 走上游
    #     marlin(sm75 实测 firefly-fp8 不比 marlin 快, 已移除; fp8 加速改走
    #     VLLM_FIREFLY_AR fp8 allreduce, 见 PLAN-fp8-allreduce)。
    # 大 M 现反量化成 int8 走 CUTLASS(IMMA), 小 M/decode 保持 marlin。见
    # model_executor/layers/quantization/utils/firefly.py。
    "VLLM_FIREFLY": _firefly_mode,
    # 默认 1024: T10 上 p3_perf_sweep(6144x5120 层)测得 int8 反量化 ~1ms/层
    # (M 无关地板), crossover M≈854; M>1024 int8 才稳定快于 marlin(1.13-1.33x)。
    "VLLM_FIREFLY_MIN_M": lambda: int(
        os.environ.get("VLLM_FIREFLY_MIN_M", "1024")
    ),
    # int8 prefill 反量化量化步: "def"(默认) 除法(与两遍 pass2 逐 bit 一致);
    # "fast" 乘倒数(per-row r=1/c_n, 再省 ~30% 反量化, off-by-one ≤0.06%)。
    "VLLM_FIREFLY_DEQUANT_MODEL": lambda: (
        os.environ.get("VLLM_FIREFLY_DEQUANT_MODEL", "def")
    ),
    # fp8 allreduce (FireflyAllReduce, SHM backend, 无 P2P 如 T10)。auto(默认)
    # 跟随 VLLM_FIREFLY (firefly 开→AR 自动开); fp8 单独开; 0 单独关。TP2 每层 2 次
    # AllReduce 量减半 (fp16->fp8), 省 ~480-530ms/27B prefill。见
    # distributed/device_communicators/firefly_allreduce.py / PLAN-fp8-allreduce。
    "VLLM_FIREFLY_AR": _firefly_ar_mode,
    # 只对大消息走 FireflyAllReduce, 小消息回退 NCCL。decode 小消息 (几百 KB)
    # 时 firefly 的 amax 扫描 + 多轮 flag spin 固定开销可能超过砍半省下的传输
    # 时间 → 反而更慢。默认 1MB (fp16 字节): 27B decode M=1 单 AR 8KB << 1MB,
    # 走 NCCL; prefill M>=256 (2MB) 起走 firefly。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_MIN_SIZE": lambda: int(
        os.environ.get("VLLM_FIREFLY_AR_MIN_SIZE", "1048576")
    ),
    # FireflyAllReduce 传输 backend: auto(默认, 运行时 _can_p2p 选 P2P 优先) /
    # p2p(强制) / shm(强制)。P2P 有 NVLink/PCIe 直连时 data/flag 全 device 显存
    # 无 host bounce; 无 P2P (PHB 如 T10) 回 SHM。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_BACKEND": _firefly_ar_backend,
    # SHM 分块全双工流水线 (0.32 新增): 仅 2-GPU SHM 全双工链上有意义, 默认关。
    "VLLM_FIREFLY_AR_PIPE": lambda: _env_flag("VLLM_FIREFLY_AR_PIPE"),
    # ---- 兼容件开关 (2026-11-16 吸收 vllm-Turing 0.32; 全部默认关) ----
    # 0.32 生态同名总门: 未设跟随 VLLM_FIREFLY, 显式覆盖优先 (见 _firefly_direct)。
    "VLLM_FIREFLY_DIRECT": _firefly_direct,
    # NVFP4 权重 firefly prefill(大 M 现反 int8 走 IMMA)+ decode 走 Marlin。
    # 默认 0 => 不建快照、不改任何权重布局, 与 final-v1 逐字节等价。
    "VLLM_FIREFLY_NVFP4": lambda: _env_flag("VLLM_FIREFLY_NVFP4"),
    # NVFP4 decode 走 firefly int8(需 VLLM_FIREFLY_NVFP4=1 先开)。
    "VLLM_FIREFLY_IDEC_NVFP4": lambda: _env_flag("VLLM_FIREFLY_IDEC_NVFP4"),
    # int4(W4A16) decode 自研 GEMV: 改 load 期权重布局(自然 row-major), 默认关。
    "VLLM_FIREFLY_IDEC": lambda: _env_flag("VLLM_FIREFLY_IDEC"),
    "VLLM_FIREFLY_IDEC_MAX_M": _firefly_idec_max_m,
    # NVFP4 decode Skillet m16n8k8 GEMM(独享快照), 默认关。
    "VLLM_FIREFLY_SKILLET": lambda: _env_flag("VLLM_FIREFLY_SKILLET"),
    "VLLM_FIREFLY_SKILLET_SPK": _firefly_skillet_spk,
    # TurboQuant KV 在 SM75 的 prefill 快路径(FlashInfer ragged, 替代 FA2/SDPA)。
    # 0.32 包默认开; 本镜像按「兼容件默认关」翻转, 未设=关 => 与 final-v1 同。
    "VLLM_FIREFLY_TQ": lambda: _env_flag("VLLM_FIREFLY_TQ"),
    # EXL3 在线解码量化法注册(仅加载 EXL3 模型时需要), 默认关。
    "VLLM_SM75_EXL3": lambda: _env_flag("VLLM_SM75_EXL3"),
    # UltraQuant 4bit KV (--kv-cache-dtype ultraquant_4bit), 默认关;
    # 关时 arg_utils 对该 dtype 显式报错(fail-closed)。
    "VLLM_SM75_ULTRAQUANT": lambda: _env_flag("VLLM_SM75_ULTRAQUANT"),
    # DFlash2 0.32 改进(分组卷积 Triton 核 / tp>=2 context-FC 切分 / draft 量化
    # 传递), 默认关 => DFlash2 行为与 final-v1 一致。
    "VLLM_SM75_DFLASH2_OPT": lambda: _env_flag("VLLM_SM75_DFLASH2_OPT"),
    # 单文件 HTML 监控页开关, 默认开(_monitor 归一化)。
    # 开 = serve 在 /monitor 挂自包含 HTML 看板(纯前端 canvas 图表, 无 CDN,
    # 轮询同源 /metrics); 0/off/false/no = 只关闭 HTML 页面(控制 API 保留)。见
    # entrypoints/serve/instrumentator/monitor.py。
    "VLLM_MONITOR": _monitor,
    # A3(sm75 参考): custom allreduce 在 cuda graph capture 时的图输入策略。
    # auto=full decode 走 registered 快路径, piecewise/prefill 回退 staging
    # buffer(sm75 图私有大 buffer 无法经 CUDA IPC 导出); registered/staging
    # 可强制覆盖。
    "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE": lambda: os.getenv(
        "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE", "auto"
    ),
    # vllm-sm75 overlay: idle auto-sleep. Populated by EngineArgs from the
    # --auto-sleep-* CLI flags; consumed inside the engine-core process by
    # vllm.v1.engine.auto_sleep(那里用 os.environ.get 直读, 此处注册仅为让
    # validate_environ 不告警 + 被 compile_factors 看到后由 INSTALL_IGNORED 排除)。
    # Idle minutes (float) before the engine auto-sleeps; 0 disables.
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_IDLE_TIMEOUT", "0")
    ),
    # 'cpu' (sleep level 1, pinned CPU backup) or 'reload' (sleep level 2,
    # weights discarded and reloaded from the checkpoint on wake).
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET": lambda: os.getenv(
        "VLLM_AUTO_SLEEP_OFFLOAD_TARGET", "cpu"
    ),
    # Checkpoint path used to reload weights on wake for the 'reload' target.
    "VLLM_AUTO_SLEEP_RELOAD_PATH": lambda: os.getenv("VLLM_AUTO_SLEEP_RELOAD_PATH", ""),
    # Seconds between page-cache re-warm ticks while sleeping (reload mode);
    # keeps the checkpoint in the OS page cache so the wake-time reload read
    # is fast. 0 disables the background keeper (one-shot warm on sleep/wake
    # still happens).
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL", "600")
    ),
    # 按 rank 只读自己那组权重文件(启动加速)。1 = 开; 默认 0 = 完全走上游
    # 加载路径(等于现基线)。开关只改"读哪些文件", 不改张量内容与名字; 任何
    # 不匹配(模型 index sha、tp_size、分片缺失)都自动回退原路径并打 warning。
    # 见 model_executor/model_loader/sm75_rank_shards.py。此处用 os.getenv 直读,
    # 注册是为了让 validate_environ 不告警。
    "VLLM_SM75_EP_SHARDED_LOAD": lambda: os.getenv(
        "VLLM_SM75_EP_SHARDED_LOAD", "0"
    ),
    # 离线重排产物根目录(宿主 /home 下), 每个 rank 一个子目录。
    "VLLM_SM75_EP_SHARD_ROOT": lambda: os.getenv(
        "VLLM_SM75_EP_SHARD_ROOT", "/home/laryyli/flashnext-ep8-shards"
    ),
}


def apply() -> None:
    """把 SM75 扩展 env 注入上游 vllm.envs(幂等)。

    由 install_sm75_overlay.py append 到上游 envs.py 尾部的
    `import vllm.envs_sm75; vllm.envs_sm75.apply()` 触发。此刻 envs 模块的
    environment_variables / compile_factors 均已定义(append 在模块体最末)。
    用 sys.modules 取正在加载的 vllm.envs(即 append 所在模块本身), 不触发
    `import vllm` 的副作用链; 测试里 patch sys.modules["vllm.envs"] 即可复用。
    """
    import sys

    envs = sys.modules["vllm.envs"]

    # 1) 注入 getter —— dict.update 天然幂等(同 key 覆盖同值)。
    envs.environment_variables.update(EXTENSIONS)

    # 2) 包 compile_factors, 从 hash factors pop 掉 INSTALL_IGNORED(幂等: 已包跳过)。
    if not getattr(envs.compile_factors, "_sm75_wrapped", False):
        _orig = envs.compile_factors

        def _compile_factors_sm75():
            factors = _orig()
            for key in INSTALL_IGNORED:
                factors.pop(key, None)
            # The dashboard only attaches HTTP routes; it cannot change a
            # compiled graph. Keep its monitor-off signature so enabling the
            # UI reuses artifacts within the same overlay compile revision.
            # Do not change the getter: the actual monitor remains enabled.
            if "VLLM_MONITOR" in factors:
                factors["VLLM_MONITOR"] = False
            # AOT lookup happens before tracing the newly selected kernel.
            # Separate releases even when the vLLM version and "auto" backend
            # configuration are unchanged; both AOT and regular caches use it.
            factors["_vllm_sm75_compile_revision"] = _COMPILE_REVISION
            return factors

        _compile_factors_sm75._sm75_wrapped = True  # type: ignore[attr-defined]
        _compile_factors_sm75.__wrapped__ = _orig
        envs.compile_factors = _compile_factors_sm75
