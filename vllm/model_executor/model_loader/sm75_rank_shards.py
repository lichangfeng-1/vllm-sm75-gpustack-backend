# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 opt-in per-rank checkpoint file selection.

The Flash-Next checkpoint keeps its whole PLE table (95 GiB, 128 vocabulary
shards) in a single safetensors file.  Every TP rank only needs the 16 shards
that cover its own vocabulary range, but the stock loader opens that file in
every rank and drops 7/8 of the rows after reading them.

``工具/脚本-flashnext按rank重排权重-v2.py`` rewrites the PLE table into one
file per TP rank under ``VLLM_SM75_EP_SHARD_ROOT``.  With
``VLLM_SM75_EP_SHARDED_LOAD=1`` the default loader asks this module for its
file list instead of globbing the model directory.

The whole module is fail-closed: any missing or inconsistent shard state
returns ``None`` and the caller keeps the stock behaviour (logging why).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import struct

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_ENABLE = "VLLM_SM75_EP_SHARDED_LOAD"
ENV_ROOT = "VLLM_SM75_EP_SHARD_ROOT"
DEFAULT_ROOT = "/home/laryyli/flashnext-ep8-shards"

_SHARD_FORMAT = "sm75-rank-shards"
_INDEX_NAME = "model.safetensors.index.json"
_PLE_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")


def is_enabled() -> bool:
    value = os.environ.get(ENV_ENABLE, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def select_rank_files(model_path: str) -> tuple[str, list[str]] | None:
    """Return ``(rank_dir, absolute weight files)`` for this rank, or None.

    ``None`` means "use the stock loader": either the switch is off or the
    offline shards do not match the model being loaded.
    """
    if not is_enabled():
        return None
    try:
        return _select_rank_files(model_path)
    except Exception:
        logger.warning(
            "SM75 sharded load: failed to resolve rank shards, "
            "falling back to the stock file list",
            exc_info=True,
        )
        return None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_names(path: str) -> list[str]:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length < 64 << 20:
            raise ValueError(f"{path}: implausible header length {length}")
        header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return list(header)


def _fallback(reason: str) -> None:
    logger.warning("SM75 sharded load: %s; using the stock loader", reason)


def _select_rank_files(model_path: str) -> tuple[str, list[str]] | None:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group

    root = os.environ.get(ENV_ROOT, DEFAULT_ROOT)
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.isfile(manifest_path):
        _fallback(f"{manifest_path} not found")
        return None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != _SHARD_FORMAT:
        _fallback(f"unsupported manifest format {manifest.get('format')!r}")
        return None

    index_path = os.path.join(model_path, _INDEX_NAME)
    if not os.path.isfile(index_path):
        _fallback(f"{index_path} not found")
        return None
    if _sha256(index_path) != manifest.get("source_index_sha256"):
        _fallback("model index sha256 differs from the one the shards were built from")
        return None

    tp_size = get_tp_group().world_size
    if int(manifest.get("tp_size", -1)) != tp_size:
        _fallback(
            f"shards were split for tp_size={manifest.get('tp_size')}, "
            f"this run uses tp_size={tp_size}"
        )
        return None

    rank = get_tensor_model_parallel_rank()
    rank_meta = (manifest.get("ranks") or {}).get(str(rank))
    if not rank_meta:
        _fallback(f"manifest has no entry for rank {rank}")
        return None
    rank_dir = os.path.join(root, f"rank_{rank}")
    files_path = os.path.join(rank_dir, "files.json")
    rank_index_path = os.path.join(rank_dir, "index.json")
    for path in (files_path, rank_index_path):
        if not os.path.isfile(path):
            _fallback(f"{path} not found")
            return None
    with open(files_path, encoding="utf-8") as fh:
        rank_files = json.load(fh)
    if int(rank_files.get("rank", -1)) != rank:
        _fallback(f"{files_path} describes rank {rank_files.get('rank')}")
        return None

    ple_rel = rank_files.get("ple_file")
    if not ple_rel:
        _fallback(f"{files_path} has no ple_file")
        return None
    ple_path = os.path.join(root, ple_rel)
    if not os.path.isfile(ple_path):
        _fallback(f"{ple_path} not found")
        return None

    low, high = rank_files.get("ple_shard_range", [-1, -2])
    expected = set(range(int(low), int(high) + 1))
    names = _safetensors_names(ple_path)
    got = set()
    for name in names:
        match = _PLE_SHARD_RE.search(name)
        if match:
            got.add(int(match.group(1)))
    if got != expected or len(names) != len(expected):
        _fallback(
            f"{ple_path} holds PLE shards {sorted(got)[:2]}..{sorted(got)[-2:]} "
            f"({len(names)} tensors), expected {low}..{high}"
        )
        return None

    with open(rank_index_path, encoding="utf-8") as fh:
        rank_index = json.load(fh)
    allowed = set(rank_index.get("weight_map", {}).values())
    ple_source = manifest.get("ple_source_file")
    common = []
    for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        base = os.path.basename(path)
        if base == ple_source or base not in allowed:
            continue
        common.append(path)
    if not common:
        _fallback(f"no common weight files resolved under {model_path}")
        return None

    files = [ple_path, *common]
    logger.info(
        "SM75 sharded load: rank %d/%d reads %d files "
        "(PLE shards %d..%d, %.2f GiB + %d shared files)",
        rank,
        tp_size,
        len(files),
        low,
        high,
        os.path.getsize(ple_path) / 2**30,
        len(common),
    )
    return rank_dir, files
