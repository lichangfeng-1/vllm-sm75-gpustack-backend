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
    # DIRECT 未显式设置时跟随 VLLM_FIREFLY; 本镜像 ENV VLLM_FIREFLY=auto => True
    # (等价 final-v1)。断言两值一致, 而不是只断言类型。
    check("firefly-direct-follows",
          envs.VLLM_FIREFLY_DIRECT == bool(envs.VLLM_FIREFLY),
          "direct=%r firefly=%r" % (envs.VLLM_FIREFLY_DIRECT,
                                    envs.VLLM_FIREFLY))

    # 2) EXL3 注册门控 (默认关 => KeyError; 开 => 可解析; 须在任何 import 之前)
    from vllm.model_executor.layers.quantization import get_quantization_config

    try:
        get_quantization_config("exl3")
        check("exl3-default-unregistered", False, "unexpectedly registered")
    except (KeyError, ValueError):
        # 0.30: NotImplementedError 之外统一抛 ValueError("Invalid quantization
        # method: exl3"): 与 final-v1 行为一致即通过。
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
    # 构建期容器无 GPU 驱动: vllm 把 Triton 换成占位模块(无 dtype 属性), UQ 的
    # triton 模块不能在此环境 import (真机有驱动即正常)。按环境能力标记 SKIP。
    from vllm.triton_utils import tl as _tl
    triton_active = hasattr(_tl, "float16")
    for m in mods:
        needs_triton = ("ultraquant" in m) or ("turboquant" in m) or ("flydsl" in m)
        if needs_triton and not triton_active:
            # 不能 import, 但至少做语法编译, 避免结构性错误被 SKIP 掩盖。
            import importlib.util
            spec = importlib.util.find_spec(m)
            path = getattr(spec, "origin", None) if spec else None
            ok, detail = bool(path), "module file not found"
            if path:
                try:
                    with open(path, encoding="utf-8") as f:
                        compile(f.read(), path, "exec")
                    ok, detail = True, ""
                except SyntaxError as e:  # noqa: BLE001
                    ok, detail = False, repr(e)
            print("SKIP (no GPU driver / triton disabled):", m)
            check("compile:%s" % m, ok, detail)
            continue
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

    # 5) arg_utils fail-closed 守卫在场 (结构化断言, 不接受注释里的字符串)
    import inspect

    import vllm.engine.arg_utils as au

    import ast as _ast

    tree = _ast.parse(inspect.getsource(au))
    guard_ok = False
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.If):
            continue
        cond = _ast.dump(node.test)
        if "VLLM_SM75_ULTRAQUANT" not in cond or "ultraquant_4bit" not in cond:
            continue
        for sub in node.body:
            if (
                isinstance(sub, _ast.Raise)
                and isinstance(sub.exc, _ast.Call)
                and getattr(sub.exc.func, "id", "") == "ValueError"
            ):
                guard_ok = True
    check("uq-fail-closed-guard", guard_ok,
          "no 'if ... ultraquant_4bit ... not VLLM_SM75_ULTRAQUANT: raise ValueError'")

    if FAILS:
        print("COMPAT-VERIFY: %d FAIL" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        raise SystemExit(1)
    print("COMPAT-VERIFY: ALL PASS")


if __name__ == "__main__":
    main()