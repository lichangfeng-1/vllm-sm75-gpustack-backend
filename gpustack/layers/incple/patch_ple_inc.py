#!/usr/bin/env python3
"""incple 补丁：让 Qwen4Exp PLE/n-gram 嵌入支持 INC（Intel AutoRound 的 auto-round 档）。
目标模型 Intel/Qwen3.8-Flash-Next-W4A16-AutoRound 必需；不打则引擎启动报
NotImplementedError: Qwen4Exp PLE embedding does not support quantization config INCConfig
（0.1.7-beta2 现场 ngram_embedding.py 的白名单只有 None / ModelOpt* / Fp8Config）。

本仓适配（相对物料 档案/调试/20261007-AutoRound映射/部署物料/patch_ple_inc.py，其余逐字保留；
①②＝入库轮，③④＝修复轮 1 按审查 I-2 新增，适配处由两处增为四处，NOTICE 改动 3 已同步登记）：
  ① 退出码换成本任务登记的码位（交接约束第 4 条）：锚点缺失 4→11（＝11b）、
     回读契约/编译失败 5→12、目标解析失败 3→2（3 已被任务 2 探源占用，2＝用法/前置，
     与 install-pcieipc.sh 的定位类失败同口径）；
  ② 兜底解析从 `import vllm` 改为 importlib.util.find_spec（不执行包内代码，
     与 install-incple-v1.sh 的探测同形态；构建期主路径由 install 传 PLE_INC_TARGET，本兜底仅供手工单跑）；
  ③（修复轮 1，审查 I-2a）锚点匹配从 str.count 子串语义改为**整行精确匹配**——
     文件行必须与锚点行逐字相等（缩进参与比对，只去行尾符）；缩进超集/前缀污染/行尾污染
     一律计为未命中，漂移类在锚点门就以 11（＝11b）出声、零写盘，不再出现"假阳通过锚点门"；
  ④（修复轮 1，审查 I-2b）写盘改为"**临时件验绿后原子落位**"：锚点门通过后先把改后全文写进
     同目录临时件，py_compile 与回读契约都在临时件上验，绿才 os.replace 原子落位到目标件；
     红则丢弃临时件退 12、目标件保持不变——"写盘先于校验、失败无回滚"的窗口被消除，
     校验未过根本碰不到目标件，因此无需回滚。

幂等：已含 INCConfig 分支则打印"已应用"并直接退出 0，绝不重复插入。
契约：import 锚点行整行唯一 + 分支锚点行对整行唯一 + 临时件 py_compile 过 + 临时件回读分支唯一
→ PLE_INC_CONTRACT_OK（成功路径另有 PLE_INC_ATOMIC_PLACED 一行标记原子落位）。
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
  0   本次插入成功（临时件验绿后原子落位），或已含 INCConfig（幂等，打印"已应用"）
  2   目标解析失败（vllm 定位不到 / 目标文件不存在；兜底解析仅供手工单跑）；
      或目标件不可读写/临时件无法在工作目录创建（用法/前置类，同口径）
  11  锚点缺失或命中数≠1（11b，本任务登记码位；整行精确匹配，不敢下手，零写盘）
  12  临时件 py_compile 失败或临时件回读契约失败（incple 断言，与 install 的计数断言同码位；
      临时件已丢弃，目标件零写盘不变）"""
import hashlib
import importlib.util
import os
import py_compile
import stat
import sys
import tempfile

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


def discard(paths):
    """丢弃临时件与它的 pyc：静默幂等，红路径不留残渣。
    Windows 上只读件（如目标件本就是 0444，chmod 同位传染给临时件后）unlink 会被拒——
    先补写位再删，保证"红则删临时件"的承诺在任何位形态下兑现。"""
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except PermissionError:
                try:
                    os.chmod(p, stat.S_IWRITE)
                    os.unlink(p)
                except OSError:
                    pass
            except OSError:
                pass


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

# —— 整行精确匹配（修复轮 1 · I-2a）——
# 行比对＝去掉行尾换行符后逐字符相等，缩进参与比对。str.count 的子串语义会让
# "        if…"（8 空格）成为 12 空格缩进行的后端子串而假阳命中（审查 R1 实锤），
# 前缀污染（如行首多了 "pass; "）同理打穿锚点门；整行相等一并堵死这两类。
IMP_ANCHOR_LINE = "from vllm.model_executor.layers.quantization.fp8 import Fp8Config"
IMP_ADD_LINE = "from vllm.model_executor.layers.quantization.inc import INCConfig"
BR_ANCHOR_LINES = ("        if not isinstance(quant_config, Fp8Config):",
                   "            raise NotImplementedError(")
BR_ADD_LINES = ("        if isinstance(quant_config, INCConfig):",
                "            # AutoRound (auto-round -> inc) checkpoints keep the PLE/n-gram table",
                "            # unquantized (BF16); only Linear/expert layers carry int4 packs.",
                "            # Measured 2026-10-07 on Intel/Qwen3.8-Flash-Next-W4A16-AutoRound:",
                "            # 134/137 PLE tensors BF16, 3 I64 index tensors, 0 packed under ple.*.",
                "            return Qwen4ExpPLEUnquantizedEmbeddingMethod()",
                "        if not isinstance(quant_config, Fp8Config):",
                "            raise NotImplementedError(")

parts = src.split("\n")
imp_hits = [i for i, p in enumerate(parts) if p == IMP_ANCHOR_LINE]
if len(imp_hits) != 1:
    print("PLE_INC_ANCHOR_FAIL import 锚点整行命中 %d 处（期望 1），不敢下手：%s" % (len(imp_hits), SRC))
    sys.exit(11)
br_hits = [i for i in range(len(parts) - 1)
           if parts[i] == BR_ANCHOR_LINES[0] and parts[i + 1] == BR_ANCHOR_LINES[1]]
if len(br_hits) != 1:
    print("PLE_INC_ANCHOR_FAIL 分支锚点整行对命中 %d 处（期望 1），不敢下手：%s" % (len(br_hits), SRC))
    sys.exit(11)

i_imp, i_br = imp_hits[0], br_hits[0]
new_parts = list(parts)
for idx, kind in sorted([(i_imp, "imp"), (i_br, "br")], reverse=True):
    if kind == "imp":
        new_parts[idx + 1:idx + 1] = [IMP_ADD_LINE]
    else:
        new_parts[idx:idx + 2] = list(BR_ADD_LINES)
new_src = "\n".join(new_parts)

# —— 临时件验绿后原子落位（修复轮 1 · I-2b）——
# py_compile 与回读契约全部在临时件上完成；红＝丢弃临时件退 12，目标件一个字节都不动。
# pyc 显式落在临时件旁并随即删除，不在目标目录生成 __pycache__（审查 M-1 顺带收口）。
workdir = os.path.dirname(SRC) or "."
tmp_path = None
pyc_path = None
fail = None
try:
    try:
        orig_mode = os.stat(SRC).st_mode & 0o7777
    except OSError:
        orig_mode = 0o644
    fd, tmp_path = tempfile.mkstemp(prefix=".incple-patch-", suffix=".tmp.py", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(new_src)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp_path, orig_mode)
    pyc_path = tmp_path + ".pyc"
    py_compile.compile(tmp_path, cfile=pyc_path, doraise=True)
    with open(tmp_path, encoding="utf-8") as f:
        s2 = f.read()
    if MARKER not in s2:
        fail = "临时件回读不到分支（写进去了但读不出来？）"
    elif s2.count(MARKER) != 1:
        fail = "临时件分支出现 %d 次（期望 1）" % s2.count(MARKER)
    if fail is None:
        os.replace(tmp_path, SRC)
        tmp_path = None
except py_compile.PyCompileError as exc:
    fail = "临时件 py_compile 未过（插入后语法坏了？）：%s" % exc
except (OSError, UnicodeError) as exc:
    discard((pyc_path, tmp_path))
    print("PLE_INC_APPLY_FAIL 目标件 %s 不可读写或临时件创建失败（用法/前置类）：%r" % (SRC, exc))
    sys.exit(2)
if fail is not None:
    discard((pyc_path, tmp_path))
    print("PLE_INC_CONTRACT_FAIL " + fail + "（临时件已丢弃，目标件零写盘不变）")
    sys.exit(12)
discard((pyc_path,))

print("PLE_INC_ATOMIC_PLACED 临时件验绿，原子落位 -> " + SRC)
print("sha256:", hashlib.sha256(new_src.encode()).hexdigest())
print("PLE_INC_CONTRACT_OK " + SRC)
