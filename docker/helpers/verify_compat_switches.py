# SPDX-License-Identifier: Apache-2.0
"""镜像内验收: 2026-11-16 兼容件开关全部默认关 + 注册/导入门控正确。

在 final-v1 之上叠兼容件层后执行 (构建期, 无需 GPU):
  1) env 默认值: 全部兼容件开关未设=关 (fail-closed);
  2) 新模块可导入 (不触发 CUDA 编译);
  3) EXL3 量化法: 默认未注册 (与 final-v1 同), 显式 VLLM_SM75_EXL3=1 时注册;
  4) UltraQuant dtype 类型面: get_kv_quant_mode / CacheDType / backend enum 就位;
  5) arg_utils 的 UQ fail-closed 守卫在场。
失败即 raise (构建期 fail-fast)。
"""
import importlib
import os
import subprocess
import sys

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print("PASS", name)
    else:
        FAILS.append("%s %s" % (name, detail))
        print("FAIL", name, detail)


def main():
    import vllm.envs as envs

    # 1) 默认关
    for key in (
        "VLLM_FIREFLY_AR",
        "VLLM_FIREFLY_NVFP4",
        "VLLM_FIREFLY_IDEC_NVFP4",
        "VLLM_FIREFLY_IDEC",
        "VLLM_FIREFLY_SKILLET",
        "VLLM_FIREFLY_TQ",
        "VLLM_FIREFLY_AR_PIPE",
        "VLLM_SM75_EXL3",
        "VLLM_SM75_ULTRAQUANT",
        "VLLM_SM75_DFLASH2_OPT",
    ):
        check("default-off:%s" % key, getattr(envs, key) is False,
              repr(getattr(envs, key, None)))
    check("idec-max-m-default", envs.VLLM_FIREFLY_IDEC_MAX_M == 8)
    check("skillet-spk-default", envs.VLLM_FIREFLY_SKILLET_SPK == 8)
    # DIRECT 跟随 VLLM_FIREFLY; 本镜像 ENV VLLM_FIREFLY=auto => True (等价 final-v1)
    check("firefly-direct-follows", isinstance(envs.VLLM_FIREFLY_DIRECT, bool))

    # 2) EXL3 注册门控 (默认关 => KeyError; 开 => 可解析; 须在任何 import 之前)
    from vllm.model_executor.layers.quantization import get_quantization_config

    try:
        get_quantization_config("exl3")
        check("exl3-default-unregistered", False, "unexpectedly registered")
    except KeyError:
        check("exl3-default-unregistered", True)

    code = (
        "from vllm.model_executor.layers.quantization import get_quantization_config;"
        "cfg=get_quantization_config('exl3');"
        "print('EXL3-OK', cfg.__name__)"
    )
    env_on = dict(os.environ, VLLM_SM75_EXL3="1")
    p = subprocess.run([sys.executable, "-c", code], env=env_on,
                       capture_output=True, text=True)
    check("exl3-enabled-registers", p.returncode == 0 and "EXL3-OK" in p.stdout,
          (p.stdout + p.stderr)[-300:])

    # 3) 新模块导入
    mods = [
        "vllm.model_executor.layers.quantization.utils.firefly_nvfp4",
        "vllm.model_executor.layers.quantization.utils.firefly_skillet",
        "vllm.model_executor.layers.quantization.utils.firefly_idecode",
        "vllm.model_executor.layers.quantization.exl3",
        "vllm.model_executor.kernels.linear.nvfp4.marlin",
        "vllm.model_executor.kernels.linear.mixed_precision.marlin",
        "vllm.v1.attention.backends.turboquant_attn",
        "vllm.v1.attention.backends.ultraquant_attn",
        "vllm.v1.attention.ops.ultraquant.triton_unified_attention",
        "vllm.v1.attention.ops.flydsl_ultraquant_decode",
        "vllm.model_executor.models.qwen3_5",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            check("import:%s" % m, True)
        except Exception as e:  # noqa: BLE001
            check("import:%s" % m, False, repr(e))

    # 4) UltraQuant dtype 类型面
    from vllm.config.cache import CacheDType
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    check("uq-cache-dtype-literal", "ultraquant_4bit" in CacheDType.__args__)
    check("uq-torch-dtype-map", "ultraquant_4bit" in STR_DTYPE_TO_TORCH_DTYPE)
    check("uq-quant-mode", get_kv_quant_mode("ultraquant_4bit") == KVQuantMode.ULTRAQUANT_4BIT)
    check("uq-backend-enum", hasattr(AttentionBackendEnum, "ULTRAQUANT")
          and AttentionBackendEnum.ULTRAQUANT is not None)

    # 5) arg_utils fail-closed 守卫在场
    import inspect

    import vllm.engine.arg_utils as au

    src = inspect.getsource(au)
    check("uq-fail-closed-guard",
          "VLLM_SM75_ULTRAQUANT" in src and "ultraquant_4bit" in src)

    if FAILS:
        print("COMPAT-VERIFY: %d FAIL" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        raise SystemExit(1)
    print("COMPAT-VERIFY: ALL PASS")


if __name__ == "__main__":
    main()