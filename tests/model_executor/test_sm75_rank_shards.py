"""CPU-only checks for the opt-in per-rank shard file selection."""
import json
import os
import runpy
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "vllm/model_executor/model_loader/sm75_rank_shards.py"
LOADER = ROOT / "vllm/model_executor/model_loader/default_loader.py"

STATE = {"rank": 0, "tp": 8}


def _stub_vllm():
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: __import__("logging").getLogger(name)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: STATE["rank"]
    distributed.get_tp_group = lambda: types.SimpleNamespace(world_size=STATE["tp"])
    sys.modules.update({"vllm": vllm, "vllm.logger": logger, "vllm.distributed": distributed})


def _write_safetensors(path, tensors):
    """tensors: list of (name, nbytes). Returns the payload bytes in order."""
    header, offset, payload = {}, 0, b""
    for name, nbytes in tensors:
        blob = bytes([len(name) % 251]) * nbytes
        header[name] = {
            "dtype": "U8",
            "shape": [nbytes],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        payload += blob
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8 or 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(payload)
    return payload


class RankShards(unittest.TestCase):
    def setUp(self):
        _stub_vllm()
        self.mod = runpy.run_path(str(MODULE))
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.model = base / "model"
        self.root = base / "shards"
        self.model.mkdir()
        self.root.mkdir()
        index = {"weight_map": {"x.weight": "common.safetensors"}}
        (self.model / "model.safetensors.index.json").write_text(json.dumps(index))
        _write_safetensors(self.model / "common.safetensors", [("x.weight", 8)])
        _write_safetensors(self.model / "ple.safetensors", [("m.ngram_embedding.shard_0.weight", 8)])
        import hashlib

        self.index_sha = hashlib.sha256(
            (self.model / "model.safetensors.index.json").read_bytes()
        ).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_root(self, ranks=8, per_rank=2, tp=8, index_sha=None, fmt="sm75-rank-shards"):
        manifest = {
            "format": fmt,
            "source_index_sha256": index_sha or self.index_sha,
            "tp_size": tp,
            "ple_source_file": "ple.safetensors",
            "ranks": {},
        }
        for rank in range(ranks):
            rank_dir = self.root / f"rank_{rank}"
            rank_dir.mkdir()
            low = rank * per_rank
            names = [
                (f"m.ngram_embedding.shard_{low + i}.weight", 8) for i in range(per_rank)
            ]
            _write_safetensors(rank_dir / f"ple-rank-{rank}.safetensors", names)
            weight_map = {"x.weight": "common.safetensors"}
            weight_map.update({n: f"rank_{rank}/ple-rank-{rank}.safetensors" for n, _ in names})
            (rank_dir / "index.json").write_text(json.dumps({"weight_map": weight_map}))
            (rank_dir / "files.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "tp_size": tp,
                        "ple_file": f"rank_{rank}/ple-rank-{rank}.safetensors",
                        "ple_shard_range": [low, low + per_rank - 1],
                    }
                )
            )
            manifest["ranks"][str(rank)] = {"ple_shard_range": [low, low + per_rank - 1]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def _select(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_SM75_EP_SHARDED_LOAD": "1",
                "VLLM_SM75_EP_SHARD_ROOT": str(self.root),
            },
            clear=False,
        ):
            return self.mod["select_rank_files"](str(self.model))

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {"VLLM_SM75_EP_SHARDED_LOAD": "0"}, clear=False):
            self.assertIsNone(self.mod["select_rank_files"](str(self.model)))

    def test_rank_files_selected(self):
        self._make_root()
        STATE.update(rank=3, tp=8)
        out = self._select()
        self.assertIsNotNone(out)
        rank_dir, files = out
        self.assertEqual(Path(rank_dir), self.root / "rank_3")
        self.assertEqual(Path(files[0]), self.root / "rank_3/ple-rank-3.safetensors")
        self.assertEqual([Path(f).name for f in files[1:]], ["common.safetensors"])
        self.assertNotIn("ple.safetensors", [Path(f).name for f in files])

    def test_index_hash_mismatch_falls_back(self):
        self._make_root(index_sha="0" * 64)
        self.assertIsNone(self._select())

    def test_tp_size_mismatch_falls_back(self):
        self._make_root(tp=4)
        STATE.update(rank=0, tp=8)
        self.assertIsNone(self._select())

    def test_wrong_ple_range_falls_back(self):
        self._make_root()
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["ranks"]["0"]["ple_shard_range"] = [5, 6]
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "rank_0/files.json").write_text(
            json.dumps(
                {
                    "rank": 0,
                    "tp_size": 8,
                    "ple_file": "rank_0/ple-rank-0.safetensors",
                    "ple_shard_range": [5, 6],
                }
            )
        )
        self.assertIsNone(self._select())

    def test_missing_manifest_falls_back(self):
        (self.root / "manifest.json").unlink(missing_ok=True)
        self.assertIsNone(self._select())

    def test_auto_round_expert_suffixes_are_filtered(self):
        import importlib.util

        import re as _re

        sys.modules.setdefault("regex", _re)  # 容器里有 regex 包；本机用 re 顶替
        path = ROOT / "vllm/model_executor/model_loader/ep_weight_filter.py"
        spec = importlib.util.spec_from_file_location("epf", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        local = {0, 1, 2, 3}
        for suffix in ("weight", "weight_packed", "qweight", "qzeros"):
            self.assertTrue(
                mod.should_skip_weight(f"m.layers.0.mlp.experts.9.down_proj.{suffix}", local),
                suffix,
            )
            self.assertFalse(
                mod.should_skip_weight(f"m.layers.0.mlp.experts.1.down_proj.{suffix}", local),
                suffix,
            )
        # 量表不跳（某些后端要全体专家的 scales 算全局 amax）
        self.assertFalse(mod.should_skip_weight("m.layers.0.mlp.experts.9.down_proj.scales", local))
        self.assertFalse(mod.should_skip_weight("m.layers.0.self_attn.q_proj.weight", local))
        self.assertFalse(mod.should_skip_weight("m.layers.0.mlp.experts.9.down_proj.qweight", None))

    def test_loader_hook_is_wired_once(self):
        text = LOADER.read_text(encoding="utf-8")
        self.assertEqual(text.count("select_rank_files(model_name_or_path)"), 1)
        self.assertIn("from vllm.model_executor.model_loader.sm75_rank_shards import", text)
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY vllm/model_executor/model_loader/sm75_rank_shards.py", dockerfile)
        installer = (ROOT / "docker/helpers/install_sm75_overlay.py").read_text(encoding="utf-8")
        self.assertIn('"model_executor/model_loader/sm75_rank_shards.py"', installer)


if __name__ == "__main__":
    unittest.main()
