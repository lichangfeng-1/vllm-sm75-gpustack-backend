#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gen-args-paste-v1.py —— 从上游档位模板生成 GPUStack 后端参数粘贴块（任务 7）
#
# 做什么：读上游 ultra/source/profiles/flash-next-tp8-256k-mtp3.json 的 args 数组，
#   丢掉 GPUStack 会自己补或本线明确不用的四组参数（首元素 ${MODEL}、${DRAFT_MODEL}
#   所在的 --speculative-config 整簇、--host/--port/--served-model-name 三簇），
#   其余按「一行一条参数」写成 flag=value 形态，落进 gpustack/args-paste-v1.txt；
#   写完前跑完断言集，任一红则不落盘。
#
# 为什么：GPUStack 2.2.3 对后端参数不做模板渲染，${MODEL} 这类占位符会字面直达引擎
#   （审计 A-01）；用户参数里没有 --max-model-len 时，平台会按推导值>8192 静默注入
#   --max-model-len 8192（vllm.py:705-716，审计 B-04），262144 档会被悄悄砍到 8192；
#   对话框启用投机解码会在用户参数之前注入同名参数造成重复（vllm.py:531-563/:627）。
#   所以这些项必须由生成器钉死并留证据，不能靠人手抄。
#
# 为什么用 flag=value 形态（不是 `--flag value` 两行、也不是空格分隔一行）：
#   1) 简报断言「每行以 -- 起」，只有把值并进同一行才成立；
#   2) GPUStack 把每行按 shlex 再切（base.py:1138 _flatten_backend_param →
#      command.py:338 safe_split），shlex 会吃掉 JSON 里的双引号，
#      空格形 --engram-config {"cpu_offload":true} 到引擎手里是 {cpu_offload:true}（坏 JSON）；
#      只有 `=` 紧跟 `{`/`[` 才触发 command.py:225-233 的 JSON 掩码，把原文还原回来；
#   3) 值本身紧凑单行（json.dumps separators=(',',':')，同上游 runtime-entrypoint.py:21 约定），
#      紧凑后无空格，等号形与 GPUStack 的掩码/归一化（command.py:169-215、base.py:67-116）都吃得下。
#
# 用法（路径请给目标解释器认得的形态；本机是原生 Windows Python，要 K:/… 或 C:\… 形，
#   Git Bash 的 /tmp/… 形会被解析成 C:\tmp\…）：
#   python gen-args-paste-v1.py                       # 默认：仓内档位模板 → 仓内粘贴块
#   python gen-args-paste-v1.py --profile P.json --out OUT.txt   # 负控夹具（OUT 必须在沙箱）
#
# 退出码（只用已登记码位，本任务不新增）：
#   0   生成并落盘（或 --profile 有效且断言全绿），打印逐行计数与丢弃清单
#   2   用法/前置红——模板缺失或不可解析、args 形态未知、落点守卫红（--out 落进上游作用域）
#   14  参数排除红——输出里出现了 GPUStack 尾补三条（--host / --port / --served-model-name）
#   16  断言集红——占位符残留 ${、投机参数、关键 flag 缺失或值不符、flag 集合与源模板不符、
#       flag 重复、行形态非法。简报只给 16 一个码覆盖这几类内容红，靠点名文案区分命中哪一道。
#
# 断言顺序（负控要各自命中自己的码，被更早的门抢走的都不算证据）：
#   P0 输入与落点前置（含 2）→ A0 非空 → A1 行形态 → A2 ${ 残留 → A3 投机参数 → A4 尾补三条（14）
#   → A5 flag 唯一 → A6 与源模板 flag 集合差集为空 → A7 关键项在列且 --max-model-len=262144
#   → 写盘 → 写后逐字节回读自证。
#   A4 排在 A6 之前：塞进 --port 的负控必须先命中 14，不能被 A6 的「多余 flag」以 16 抢走。
#   A6/A7 排在行数收口之前：抽掉 --max-model-len 行的负控必须命中 16，不能被行数门以 2 抢走。

import json
import os
import sys

EXIT_USAGE = 2
EXIT_EXCLUDED = 14
EXIT_ASSERT = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_PROFILE = os.path.join(
    DEFAULT_REPO_ROOT, "ultra", "source", "profiles", "flash-next-tp8-256k-mtp3.json"
)
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "args-paste-v1.txt")

# GPUStack 会自动尾补（vllm.py:654-659，用户写了则用户赢），粘贴块里出现即重复参数
TAIL_THREE = ("--host", "--port", "--served-model-name")
# 本线明确排除：MTP 暂不考虑（用户 2026-10-11 裁决），草稿模型走后续参数行
DROP_SPECULATIVE = "--speculative-config"
EXCLUDED_FLAGS = TAIL_THREE + (DROP_SPECULATIVE,)

# 简报点名的关键项（差集断言之外的显式加严，值必须逐字对上）
KEY_FLAGS = ("--max-model-len", "--tensor-parallel-size", "--engram-config", "--chat-template")
KEY_VALUES = {"--max-model-len": "262144"}

# 上游 14 个顶层项（交接约束第 1 条的禁写面；与 gate-tree-unchanged-v1.sh 的作用域同集合）
UPSTREAM_SCOPES = (
    "vllm", "docker", "ultra", "docs", "tests", "tools", "requirements",
    "build-dependencies", "LICENSE", "README.md", "README.en.md",
    ".dockerignore", ".gitattributes", ".gitignore",
)
REPO_MARKERS = (".git", os.path.join("gpustack", "SHA256SUMS-upstream.txt"))


def die(code, message):
    sys.stderr.write(message.rstrip("\n") + "\n")
    sys.exit(code)


def compact(value):
    """非字符串元素转成紧凑单行，与上游 runtime-entrypoint.py:21 同约定。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def flag_of(line):
    """参数行 → flag 名（等号形取等号左侧）。"""
    return line.split("=", 1)[0]


def repo_root_upwards(start):
    """从 start 逐级向上找仓标记，返回归一后的仓根；找不到返回 None。"""
    current = os.path.abspath(start)
    while True:
        for marker in REPO_MARKERS:
            if os.path.exists(os.path.join(current, marker)):
                return os.path.normcase(current)
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def within(root, path):
    """归一后的前缀包含判定（Windows 上 normcase 已折叠大小写与分隔符）。"""
    return path == root or path.startswith(root + os.sep) or path.startswith(root + "\\")


def guard_out_target(out_path):
    """落点守卫（交接约束第 8 条）：写盘之前判死，红则零落盘。

    词法形（abspath）与物理形（realpath，穿过链接点）两条途各向上遍历仓标记；
    任一途落进上游 14 顶层项之下即退 2。允许面＝gpustack/、docs-gpustack/ 与仓根新顶层。
    """
    if ".." in out_path.replace("\\", "/").split("/"):
        die(EXIT_USAGE, "错误：--out 含 .. 段，无法可靠判定真实落点（退出码 2），本脚本拒绝猜测。")
    legs = (
        ("词法途", os.path.normpath(os.path.abspath(out_path))),
        ("物理途", os.path.normpath(os.path.realpath(out_path))),
    )
    for leg_name, resolved in legs:
        leg_path = os.path.normcase(resolved)
        anchor_dir = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
        root = repo_root_upwards(anchor_dir)
        if root is None or not within(root, leg_path):
            continue
        top = leg_path[len(root) + 1:].replace("\\", "/").split("/")[0]
        if top in UPSTREAM_SCOPES:
            die(
                EXIT_USAGE,
                "错误：--out 落点不得写进上游作用域（退出码 2）：途=%s 落点=%s 位于仓根=%s 之下，"
                "顶层项「%s」属禁写面（交接约束第 1 条），自研件只能落 gpustack/、docs-gpustack/ 或仓根新顶层。"
                % (leg_name, resolved, root, top),
            )


def load_profile(path):
    if not os.path.isfile(path):
        die(EXIT_USAGE, "错误：档位模板不存在或不是文件（退出码 2）：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        die(EXIT_USAGE, "错误：档位模板不可读或 JSON 解析失败（退出码 2）：%s：%s" % (path, exc))
    args = data.get("args")
    if not isinstance(args, list) or not args:
        die(EXIT_USAGE, "错误：档位模板的 args 不是非空数组（退出码 2）：%s" % path)
    return args


def build_lines(raw_args):
    """按「一个 flag 加它的值」分组，丢弃首元素位置参数与四组排除项，产出参数行。"""
    dropped = []
    cursor = 0
    if not str(raw_args[0]).startswith("--"):
        dropped.append("首元素位置参数 %r（GPUStack 以模型路径位置参数注入）" % (raw_args[0],))
        cursor = 1
        while cursor < len(raw_args) and not str(raw_args[cursor]).startswith("--"):
            die(EXIT_USAGE, "错误：args 里出现第二个前导位置参数 %r，形态未知，拒绝猜测（退出码 2）"
                % (raw_args[cursor],))
    lines = []
    while cursor < len(raw_args):
        token = str(raw_args[cursor])
        if not token.startswith("--"):
            die(EXIT_USAGE, "错误：参数 %r 出现在任何 --flag 之前（游离值），拒绝猜测（退出码 2）" % token)
        flag = token.split("=", 1)[0]
        values = []
        if "=" in token:
            values = [token.split("=", 1)[1]]
        cursor += 1
        while cursor < len(raw_args) and not str(raw_args[cursor]).startswith("--"):
            values.append(compact(raw_args[cursor]))
            cursor += 1
        if flag in EXCLUDED_FLAGS:
            dropped.append("整簇 %s（%d 个元素）" % (flag, len(values) + 1))
            continue
        if len(values) > 1:
            die(EXIT_USAGE, "错误：%s 是多值参数簇（%d 个值），本粘贴块形态一行一值，拒绝猜测（退出码 2）"
                % (flag, len(values)))
        lines.append(flag if not values else "%s=%s" % (flag, values[0]))
    return lines, dropped


def assert_set(lines, source_flags):
    """断言集：任一红即不落盘。顺序即头注声明的顺序。"""
    if not lines:
        die(EXIT_ASSERT, "红：断言 A0 非空——生成 0 行参数（空粘贴块＝空门禁，后续所有计数判据都会裸跑）"
                         "（退出码 16）")
    for line in lines:
        if not line.startswith("--") or line.strip() != line or not line:
            die(EXIT_ASSERT, "红：断言 A1 行形态——存在不以 -- 起或含首尾空白的行（退出码 16）：%r" % line)
    joined = "\n".join(lines)
    if "${" in joined:
        die(EXIT_ASSERT, "红：断言 A2 占位符残留——输出含字面 ${（GPUStack 不做模板渲染，会字面直达引擎）"
                         "（退出码 16）。逐行如下：\n%s"
            % "\n".join("    " + line for line in lines if "${" in line))
    if DROP_SPECULATIVE in joined:
        die(EXIT_ASSERT, "红：断言 A3 投机参数——输出含 --speculative-config（本线 MTP 暂不考虑，"
                         "且 UI 投机注入位在用户参数之前会造成重复参数）（退出码 16）")
    for flag in TAIL_THREE:
        if flag in joined:
            die(EXIT_EXCLUDED, "红：断言 A4 参数排除——输出含 GPUStack 尾补参数 %s（含则重复注入，"
                               "退出码 14）。命中行：%s"
                % (flag, [line for line in lines if flag in line]))
    seen = []
    for line in lines:
        flag = flag_of(line)
        if flag in seen:
            die(EXIT_ASSERT, "红：断言 A5 flag 唯一——%s 出现多次（退出码 16）" % flag)
        seen.append(flag)
    missing = [flag for flag in source_flags if flag not in seen]
    unexpected = [flag for flag in seen if flag not in source_flags]
    if missing or unexpected:
        die(EXIT_ASSERT, "红：断言 A6 flag 差集——与源模板保留集不符（退出码 16）：%s"
            % "；".join(x for x in (
                "缺 %d 项：%s" % (len(missing), ",".join(missing)) if missing else "",
                "多 %d 项：%s" % (len(unexpected), ",".join(unexpected)) if unexpected else "",
            ) if x))
    values = {}
    for line in lines:
        if "=" in line:
            values[flag_of(line)] = line.split("=", 1)[1]
    for flag in KEY_FLAGS:
        if flag not in seen:
            die(EXIT_ASSERT, "红：断言 A7 关键项缺失——%s 不在列（退出码 16）" % flag)
    for flag, expected in KEY_VALUES.items():
        if values.get(flag) != expected:
            die(EXIT_ASSERT, "红：断言 A7 关键项取值——%s 实为 %r，期望 %r（防 GPUStack 静默钳 8192，"
                             "vllm.py:705-716）（退出码 16）" % (flag, values.get(flag), expected))


def source_kept_flags(raw_args):
    """源模板的保留 flag 顺序表（排除项已剔除，重复项去首保留）。"""
    out = []
    for token in raw_args:
        if not str(token).startswith("--"):
            continue
        flag = str(token).split("=", 1)[0]
        if flag in EXCLUDED_FLAGS or flag in out:
            continue
        out.append(flag)
    return out


def main(argv):
    profile, out = DEFAULT_PROFILE, DEFAULT_OUT
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("-h", "--help"):
            print("用法：python gen-args-paste-v1.py [--profile 模板.json] [--out 输出.txt]")
            print("退出码：0 通过 / 2 用法与前置与落点 / 14 尾补参数排除红 / 16 断言集红")
            return 0
        if arg in ("--profile", "--out"):
            if index + 1 >= len(argv) or not argv[index + 1]:
                die(EXIT_USAGE, "错误：%s 缺少参数值（退出码 2）" % arg)
            if arg == "--profile":
                profile = argv[index + 1]
            else:
                out = argv[index + 1]
            index += 2
            continue
        die(EXIT_USAGE, "错误：未识别的参数 %r（退出码 2）" % arg)

    guard_out_target(out)
    raw_args = load_profile(profile)
    lines, dropped = build_lines(raw_args)
    assert_set(lines, source_kept_flags(raw_args))

    payload = "".join(line + "\n" for line in lines)
    try:
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except OSError as exc:
        die(EXIT_USAGE, "错误：粘贴块写盘失败（退出码 2）：%s：%s" % (out, exc))

    # 写后逐字节回读自证：形态契约不能只靠"我写过"这句话
    try:
        with open(out, "rb") as handle:
            written = handle.read()
    except OSError as exc:
        die(EXIT_USAGE, "错误：粘贴块回读失败（退出码 2）：%s：%s" % (out, exc))
    if written != payload.encode("utf-8"):
        die(EXIT_USAGE, "错误：粘贴块回读字节与生成内容不一致（退出码 2），拒绝报成功。")
    if b"\r" in written:
        die(EXIT_USAGE, "错误：粘贴块含 CR 字节（退出码 2），请用 LF 检出：git -c core.autocrlf=false clone")
    if not written.endswith(b"\n"):
        die(EXIT_USAGE, "错误：粘贴块末行缺换行符（退出码 2）。")

    print("模板：%s" % profile)
    print("输出：%s（%d 行，%d 字节，CR 0，末行有换行）" % (out, len(lines), len(written)))
    print("丢弃 %d 组：%s" % (len(dropped), "；".join(dropped)))
    print("GEN_ARGS_PASTE_OK lines=%d flags=%d" % (len(lines), len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
