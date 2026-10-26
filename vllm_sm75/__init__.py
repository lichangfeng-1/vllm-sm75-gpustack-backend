"""SM75 (Turing) optional optimizations — gated, individually switchable.

Gate order (short-circuits in this order):

1. compute capability must be 7.5 (checked lazily; CUDA may not be initialised at import time)
2. ``VLLM_SM75_OPT`` = ``auto`` | ``0`` | ``1``
3. per-item flag, e.g. ``VLLM_SM75_HC_FUSE`` = ``auto`` | ``0`` | ``1``
4. explicit launch arguments win over all of the above

With ``VLLM_SM75_OPT=0`` nothing in this package is imported beyond this module, and the
engine must behave byte-identically to the baseline image.
"""

import logging
import os

__all__ = ["install", "enabled", "opt_mode"]

logger = logging.getLogger(__name__)

_INSTALLED = False
_CC_OK = None


def _capability_ok():
    global _CC_OK
    if _CC_OK is None:
        try:
            import torch

            _CC_OK = bool(torch.cuda.is_available()) and torch.cuda.get_device_capability() == (7, 5)
        except Exception:  # noqa: BLE001 - import-time environments can be odd
            # Do not cache a transient failure: a CPU-only utility process must not
            # permanently disable the gate for the engine process that follows.
            return False
    return _CC_OK


def opt_mode():
    v = os.environ.get("VLLM_SM75_OPT", "auto").strip().lower()
    if v in ("auto", "0", "1"):
        return v
    # Fail closed: a typo such as VLLM_SM75_OPT=off must not enable the auto items.
    logger.warning("vllm_sm75: invalid VLLM_SM75_OPT=%r, treating it as '0'", v)
    return "0"


def enabled(item):
    """Whether ``item`` should be active right now."""
    if opt_mode() == "0":
        return False
    if not _capability_ok():
        return False
    flag = os.environ.get("VLLM_SM75_" + item.upper(), "auto").strip().lower()
    if flag == "0":
        return False
    if flag == "1":
        return True
    from .registry import AUTO_ENABLED

    return item in AUTO_ENABLED


def install():
    """Register every enabled item. Idempotent; must stay cheap when everything is off."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if opt_mode() == "0" or not _capability_ok():
        return
    from . import registry

    registry.install_all(enabled)
