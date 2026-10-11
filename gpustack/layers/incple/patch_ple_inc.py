#!/usr/bin/env python3
"""incple 补丁：让 Qwen4Exp PLE/n-gram 嵌入支持 INC（Intel AutoRound 的 auto-round 档）。
目标模型 Intel/Qwen3.8-Flash-Next-W4A16-AutoRound 必需；不打则引擎启动报
NotImplementedError: Qwen4Exp PLE embedding does not support quantization config INCConfig
（0.1.7-beta2 现场 ngram_embedding.py 的白名单只有 None / ModelOpt* / Fp8Config）。

本仓适配（相对物料 调试档案-20261007-AutoRound映射/部署物料/patch_ple_inc.py，只动两处，其余逐字保留）：
  ① 退出码换成本任务登记的码位（交接约束第 4 条）：锚点缺失 4→11（＝11b）、
     回读契约/编译失败 5→12、目标解析失败 3→2（3 已被任务 2 探源占用，2＝用法/前置，
     与 install-pcieipc.sh 的定位类失败同口径）；
  ② 兜底解析从 `import vllm` 改为 importlib.util.find_spec（不执行包内代码，
     与 install-incple-v1.sh 的探测同形态；构建期主路径由 install 传 PLE_INC_TARGET，本兜底仅供手工单跑）。

幂等：已含 INCConfig 分支则打印"已应用"并直接退出 0，绝不重复插入。
契约：import 锚点唯一 + 分支锚点唯一 + py_compile 通过 + 回读分支在位 → PLE_INC_CONTRACT_OK。
原理（2026-10-07 实测，不是转述）：该 AutoRound 档 137 个 PLE 张量＝134 个 BF16 weight
＋3 个 I64 索引（layer_multipliers / ngram_heads_offsets / ngram_heads_vocab_sizes），
ple.* 底下 qweight/qzeros/scales/g_idx 形态的量化件为 0；同分片对照的量化 Linear 确为 I32 packed。
⇒ PLE/n-gram 表未量化存放，INC 配置下应走 Unquantized 分支。
影响面封闭：新分支只在 quant_config 是 INCConfig 时命中，FP8/ModelOpt/未量化档走原分支，行为不变。
插入形态：与上游既有同构形态对齐（vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
:45 的 INCConfig import 与 :778 的白名单 isinstance 判据），把 INCConfig 加进 PLE 白名单判据。

纪律：只插一支 if 与一行 import，**不整文件覆盖**（0.16 的 ngram_embedding.incple-*.py 是整文件替换，
跨版本漂移过，本轮不复用那个形态）。
目标文件路径在镜像内解析；测试可用环境变量 PLE_INC_TARGET 指向假现场。

退出码（本脚本自有；install-incple-v1.sh 原样透传，构建期由 docker 统一收进 20＝构建失败）：
  0   本次插入成功，或已含 INCConfig（幂等，打印"已应用"）
  2   目标解析失败（vllm 定位不到 / 目标文件不存在；兜底解析仅供手工单跑）
  11  锚点缺失或命中数≠1（11b，本任务登记码位；不敢下手，不写盘）
  12  写盘后回读契约失败或 py_compile 失败（incple 断言，与 install 的计数断言同码位）"""
import hashlib
import importlib.util
import os
import py_compile
import sys

MARKER = "isinstance(quant_config, INCConfig)"
BASELINE_PREFIX = "/usr/local/lib/python3.12/dist-packages/"


def resolve_target():
    """先认环境变量（测试/主路径用），再在镜像内按解释器实况解析，解析不到就把诊断打全。"""
    env = os.environ.get("PLE_INC_TARGET")
    if env:
        return env, None
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception as exc:  # noqa: BLE001 - 诊断要的是"为什么定位不了"
        return None, ("vllm 定位失败：%r（基座不是预期镜像，或解释器不对）" % (exc,))
    if spec is None:
        return None, "vllm 不在 sys.path 上＝基座不是预期镜像，或解释器不对"
    pkg_dir = None
    origin = getattr(spec, "origin", None)
    if origin and origin != "namespace" and os.path.isfile(origin):
        pkg_dir = os.path.dirname(os.path.abspath(origin))
    else:
        for d in (list(getattr(spec, "submodule_search_locations", None) or [])):
            d = os.path.abspath(d)
            if os.path.basename(d) == "vllm":
                pkg_dir = d
                break
    if pkg_dir is None:
        return None, "vllm 定位到但拿不到包目录（find_spec 既无 origin 也无 search_locations）"
    cand = os.path.join(pkg_dir, "models", "qwen4_exp", "nvidia", "ngram_embedding.py")
    if not os.path.isfile(cand):
        return None, ("vllm 可定位但 %s 不存在＝基座不含 qwen4_exp（不是 SM75 ultra 派生镜像）" % cand)
    return cand, None


SRC, err = resolve_target()
if err:
    print("PLE_INC_RESOLVE_FAIL " + err)
    print("两种根因：文件不在＝基座不含 qwen4_exp；vllm 定位不了＝基座不是预期镜像。都属基座问题，不在补丁范围。")
    sys.exit(2)
if not os.path.abspath(SRC).startswith(BASELINE_PREFIX) and not os.environ.get("PLE_INC_TARGET"):
    print("PLE_PATH_DRIFT 实际=%s（基线 %s…；布局不同不影响补丁正确性，只说明这台机不是参考环境）" % (SRC, BASELINE_PREFIX))

with open(SRC, encoding="utf-8") as f:
    src = f.read()
if MARKER in src:
    print("PLE_INC_ALREADY_PATCHED 已应用（幂等，不再插入） " + SRC)
    sys.exit(0)

IMP_ANCHOR = "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
IMP_ADD = "from vllm.model_executor.layers.quantization.inc import INCConfig\n"
n_imp = src.count(IMP_ANCHOR)
if n_imp != 1:
    print("PLE_INC_ANCHOR_FAIL import 锚点命中 %d 处（期望 1），不敢下手：%s" % (n_imp, SRC))
    sys.exit(11)
src = src.replace(IMP_ANCHOR, IMP_ANCHOR + IMP_ADD, 1)

BR_ANCHOR = ("        if not isinstance(quant_config, Fp8Config):\n"
             "            raise NotImplementedError(")
BR_ADD = ("        if isinstance(quant_config, INCConfig):\n"
          "            # AutoRound (auto-round -> inc) checkpoints keep the PLE/n-gram table\n"
          "            # unquantized (BF16); only Linear/expert layers carry int4 packs.\n"
          "            # Measured 2026-10-07 on Intel/Qwen3.8-Flash-Next-W4A16-AutoRound:\n"
          "            # 134/137 PLE tensors BF16, 3 I64 index tensors, 0 packed under ple.*.\n"
          "            return Qwen4ExpPLEUnquantizedEmbeddingMethod()\n"
          "        if not isinstance(quant_config, Fp8Config):\n"
          "            raise NotImplementedError(")
n_br = src.count(BR_ANCHOR)
if n_br != 1:
    print("PLE_INC_ANCHOR_FAIL 分支锚点命中 %d 处（期望 1），不敢下手：%s" % (n_br, SRC))
    sys.exit(11)
src = src.replace(BR_ANCHOR, BR_ADD, 1)

with open(SRC, "w", encoding="utf-8", newline="") as f:
    f.write(src)
try:
    py_compile.compile(SRC, doraise=True)
except py_compile.PyCompileError as exc:
    print("PLE_INC_CONTRACT_FAIL py_compile 未过（插入后语法坏了？）：%s" % exc)
    sys.exit(12)
with open(SRC, encoding="utf-8") as f:
    s2 = f.read()
if MARKER not in s2:
    print("PLE_INC_CONTRACT_FAIL 回读不到分支（写进去了但读不出来？）")
    sys.exit(12)
if s2.count(MARKER) != 1:
    print("PLE_INC_CONTRACT_FAIL 分支出现 %d 次（期望 1）" % s2.count(MARKER))
    sys.exit(12)
print("sha256:", hashlib.sha256(s2.encode()).hexdigest())
print("PLE_INC_CONTRACT_OK " + SRC)
