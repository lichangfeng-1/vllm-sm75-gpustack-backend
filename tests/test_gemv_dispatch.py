"""Dispatch test for vllm_sm75.gemv_m1 using stub torch/triton modules.

The real module imports torch + triton at import time, so it cannot be imported
on a CPU-only workstation. This test installs minimal stubs, imports the module
fresh, and checks the wiring a GPU would otherwise be needed to catch:

  * install() patches UnquantizedLinearMethod / UnquantizedEmbeddingMethod once;
  * apply() only routes allowlisted (N, K) shapes to the custom op;
  * bias / M>1 / other shapes fall back to the original implementation.

Run: python tests/test_gemv_dispatch.py   (CPU only, no torch needed)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Tensor:
    def __init__(self, shape, dtype="float16", cuda=True, contig=True, dev="cuda:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._cuda = cuda
        self._contig = contig
        self.device = dev

    def dim(self):
        return len(self.shape)

    def stride(self, dim=None):
        out, acc = [], 1
        for d in reversed(self.shape):
            out.append(acc)
            acc *= d
        out = tuple(reversed(out))
        return out if dim is None else out[dim]

    def is_contiguous(self):
        return self._contig

    def is_cuda(self):
        return self._cuda

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def new_empty(self, shape):
        return _Tensor(shape, self.dtype, self._cuda)


class _CustomOp:
    def __init__(self, fn):
        self._fn = fn
        self.fake = None

    def __call__(self, *a, **kw):
        return self._fn(*a, **kw)

    def register_fake(self, fn):
        self.fake = fn
        return fn


class _Kernel:
    def __init__(self, fn):
        self.fn = fn
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*a, **kw):
            self.launches += 1
            return None

        return launch

    def __call__(self, *a, **kw):
        self.launches += 1
        return None


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.Tensor = _Tensor
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.empty = lambda shape, device=None, dtype=None: _Tensor(shape, dtype or "float16")

    cuda = types.ModuleType("torch.cuda")
    cuda.get_device_capability = lambda _dev=None: (7, 5)
    torch.cuda = cuda

    lib = types.ModuleType("torch.library")
    lib.custom_op = lambda name, mutates_args=(): (lambda fn: _CustomOp(fn))
    torch.library = lib

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.linear = lambda x, w, bias=None: ("F.linear", tuple(w.shape))
    nn.functional = functional
    torch.nn = nn

    triton = types.ModuleType("triton")
    triton.jit = lambda fn: _Kernel(fn)
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    tl.float32 = "tl.float32"
    for name in ("arange", "zeros", "load", "store", "sum"):
        setattr(tl, name, lambda *a, **k: None)
    triton.language = tl

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    mel = types.ModuleType("vllm.model_executor")
    mel.__path__ = []
    layers = types.ModuleType("vllm.model_executor.layers")
    layers.__path__ = []
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    vpe = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            return ("orig_linear", tuple(layer.weight.shape))

    class UnquantizedEmbeddingMethod:
        def apply(self, layer, x):
            return ("orig_embed", tuple(layer.weight.shape))

    linear.UnquantizedLinearMethod = UnquantizedLinearMethod
    vpe.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod

    mods = {
        "torch": torch, "torch.cuda": cuda, "torch.library": lib, "torch.nn": nn,
        "torch.nn.functional": functional, "triton": triton, "triton.language": tl,
        "vllm": vllm, "vllm.model_executor": mel,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.linear": linear,
        "vllm.model_executor.layers.vocab_parallel_embedding": vpe,
    }
    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    return saved, UnquantizedLinearMethod, UnquantizedEmbeddingMethod


class _Layer:
    def __init__(self, shape):
        self.weight = _Tensor(shape)


def run():
    saved, LinM, EmbM = _install_stubs()
    try:
        sys.modules.pop("vllm_sm75.gemv_m1", None)
        import vllm_sm75.gemv_m1 as mod

        layer = _Layer((2048, 2560))
        x = _Tensor((1, 2560))

        assert LinM().apply(layer, x)[0] == "orig_linear"
        before = LinM.apply
        mod.install()
        assert getattr(LinM, "_sm75_gemv_patched", False), "class not marked patched"
        assert LinM.apply is not before, "apply not replaced"
        mod.install()
        print("PASS install patches once and stays idempotent")

        assert isinstance(mod.PLANS, dict) and not mod.PLANS, mod.PLANS
        assert LinM().apply(layer, x)[0] == "orig_linear"
        print("PASS empty PLANS falls back to the original method")

        mod.PLANS[(2048, 2560)] = (1, 256, 4)
        out = LinM().apply(layer, x)
        assert isinstance(out, _Tensor) and out.shape == (1, 2048), out
        print("PASS allowlisted shape routes to the custom op")

        assert LinM().apply(layer, x, _Tensor((2048,)))[0] == "orig_linear"
        print("PASS bias falls back")

        out = mod.gemv_m1(_Tensor((4, 2560)), _Tensor((2048, 2560)))
        assert out[0] == "F.linear", out
        print("PASS M>1 falls back inside the op")

        out = EmbM().apply(_Layer((2048, 2560)), x)
        assert isinstance(out, _Tensor), out
        print("PASS embedding method routes to the custom op")

        out = LinM().apply(_Layer((12, 2560)), _Tensor((1, 2560)))
        assert out[0] == "orig_linear", out
        print("PASS non-allowlisted shape falls back")
    finally:
        for name, mod_ in saved.items():
            if mod_ is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod_
        sys.modules.pop("vllm_sm75.gemv_m1", None)


if __name__ == "__main__":
    run()
    print("\nall gemv dispatch checks passed")
