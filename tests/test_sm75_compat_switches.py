"""L1 静态测试: 2026-11-16 vllm-Turing 0.32 兼容件吸收 (默认关 + 门控在位)。

CPU-only, 无需 vllm/torch: 直接 import vllm/envs_sm75.py (纯 os 依赖)。
Run:  python -m pytest tests/test_sm75_compat_switches.py -q
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "envs_sm75_under_test", ROOT / "vllm/envs_sm75.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

KEYS = (
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
)


def _clear():
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.pop("VLLM_FIREFLY", None)
    os.environ.pop("VLLM_FIREFLY_DIRECT", None)


def _getter(key):
    return m.EXTENSIONS[key]


def test_all_compat_switches_default_off():
    _clear()
    for key in KEYS:
        assert _getter(key)() is False, key
    assert m.EXTENSIONS["VLLM_FIREFLY_IDEC_MAX_M"]() == 8
    assert m.EXTENSIONS["VLLM_FIREFLY_SKILLET_SPK"]() == 8


def test_ar_never_auto_enables_even_with_firefly_on():
    _clear()
    os.environ["VLLM_FIREFLY"] = "1"
    assert _getter("VLLM_FIREFLY_AR")() is False
    os.environ["VLLM_FIREFLY_AR"] = "auto"  # 旧默认值: 按红线视为关
    assert _getter("VLLM_FIREFLY_AR")() is False
    os.environ["VLLM_FIREFLY_AR"] = "fp8"
    assert _getter("VLLM_FIREFLY_AR")() is True
    os.environ["VLLM_FIREFLY_AR"] = "1"
    assert _getter("VLLM_FIREFLY_AR")() is True


def test_explicit_enable_values():
    _clear()
    for key in KEYS:
        if key == "VLLM_FIREFLY_AR":
            continue
        os.environ[key] = "1"
        assert _getter(key)() is True, key
        os.environ[key] = "on"
        assert _getter(key)() is True, key
        os.environ[key] = "0"
        assert _getter(key)() is False, key


def test_firefly_direct_follows_total_switch():
    _clear()
    os.environ["VLLM_FIREFLY"] = "auto"
    assert _getter("VLLM_FIREFLY_DIRECT")() is True
    os.environ["VLLM_FIREFLY"] = "0"
    assert _getter("VLLM_FIREFLY_DIRECT")() is False
    os.environ["VLLM_FIREFLY_DIRECT"] = "0"  # 显式覆盖优先
    os.environ["VLLM_FIREFLY"] = "1"
    assert _getter("VLLM_FIREFLY_DIRECT")() is False


def test_numeric_clamps():
    _clear()
    os.environ["VLLM_FIREFLY_IDEC_MAX_M"] = "99"
    assert m.EXTENSIONS["VLLM_FIREFLY_IDEC_MAX_M"]() == 8
    os.environ["VLLM_FIREFLY_IDEC_MAX_M"] = "4"
    assert m.EXTENSIONS["VLLM_FIREFLY_IDEC_MAX_M"]() == 4
    os.environ["VLLM_FIREFLY_SKILLET_SPK"] = "16"
    assert m.EXTENSIONS["VLLM_FIREFLY_SKILLET_SPK"]() == 16
    os.environ["VLLM_FIREFLY_SKILLET_SPK"] = "7"
    assert m.EXTENSIONS["VLLM_FIREFLY_SKILLET_SPK"]() == 8


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_new_files_exist_and_parse():
    files = [
        "vllm/model_executor/kernels/linear/nvfp4/marlin.py",
        "vllm/model_executor/layers/quantization/utils/firefly_nvfp4.py",
        "vllm/model_executor/layers/quantization/utils/firefly_nvfp4.cu",
        "vllm/model_executor/layers/quantization/utils/firefly_skillet.py",
        "vllm/model_executor/layers/quantization/utils/firefly_skillet.cu",
        "vllm/model_executor/layers/quantization/utils/firefly_idecode.py",
        "vllm/model_executor/layers/quantization/utils/firefly_idecode.cu",
        "vllm/model_executor/layers/quantization/utils/firefly_exl3.cu",
        "vllm/model_executor/layers/quantization/exl3/__init__.py",
        "vllm/model_executor/layers/quantization/exl3/exl3.py",
        "vllm/v1/attention/backends/turboquant_attn.py",
        "vllm/v1/attention/backends/ultraquant_attn.py",
        "vllm/v1/attention/ops/flydsl_ultraquant_decode.py",
        "vllm/v1/attention/ops/flydsl_kernels/ultraquant_decode_hd256.py",
        "vllm/v1/attention/ops/ultraquant/__init__.py",
        "vllm/v1/attention/ops/ultraquant/format.py",
        "vllm/v1/attention/ops/ultraquant/reference.py",
        "vllm/v1/attention/ops/ultraquant/triton_dequant.py",
        "vllm/v1/attention/ops/ultraquant/triton_store.py",
        "vllm/v1/attention/ops/ultraquant/triton_unified_attention.py",
        "vllm/v1/kv_cache_interface.py",
        "vllm/config/cache.py",
        "vllm/utils/torch_utils.py",
        "vllm/v1/attention/backends/registry.py",
        "vllm/platforms/cuda.py",
        "vllm/platforms/interface.py",
        "vllm/model_executor/models/qwen3_5.py",
        "docker/helpers/verify_compat_switches.py",
        "gpustack/Dockerfile.compat-v1-20261116",
    ]
    for rel in files:
        p = ROOT / rel
        assert p.is_file(), rel
        if rel.endswith(".py"):
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def test_gating_markers_present():
    npm = _read("vllm/model_executor/kernels/linear/nvfp4/marlin.py")
    assert "envs.VLLM_FIREFLY_NVFP4" in npm
    assert "workspace=layer.workspace" in npm  # 0.30 适配, 不得回 None
    mpm = _read("vllm/model_executor/kernels/linear/mixed_precision/marlin.py")
    assert "workspace=self.workspace" in mpm
    assert "VLLM_FIREFLY_IDEC_MAX_M" in mpm
    tq = _read("vllm/v1/attention/backends/turboquant_attn.py")
    assert "envs.VLLM_FIREFLY_TQ" in tq
    assert "def get_supported_kernel_block_sizes()" in tq  # 0.30 签名
    au = _read("vllm/engine/arg_utils.py")
    assert "VLLM_SM75_ULTRAQUANT" in au and "fail-closed" in au
    inst = _read("docker/helpers/install_sm75_overlay.py")
    for rel in ("nvfp4/marlin.py", "exl3/exl3.py", "ultraquant_attn.py",
                "turboquant_attn.py", "qwen3_5.py", "kv_cache_interface.py"):
        assert rel in inst, rel
    assert "EXL3 quant method registration" in inst
    df2 = _read("docker/speculative/vllm/model_executor/models/qwen3_dflash2.py")
    assert "_df2_opt()" in df2 and "VLLM_SM75_DFLASH2_OPT" in df2
    gdn = _read("docker/speculative/sm75_gdn_metadata.py")
    assert "spec_sequence_masks_cpu" in gdn
    dfl = _read("docker/speculative/vllm/model_executor/models/qwen3_dflash.py")
    assert "v_scale" in dfl


def test_dockerfile_copy_sources_exist():
    text = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            for source in line.split()[1:-1]:
                assert (ROOT / source).exists(), source


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)