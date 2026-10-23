# -*- coding: utf-8 -*-
"""静态校验 v9 harness overlay：vLLM 的 At most N image(s) 必须映射到
IMAGE_OFFLOAD_REQUIRED，并带 offloadImages:1，交给自带 image-offload 插件。

用法：python3 verify-image-offload-v1.py dsh-llm-pi-ai-index.js
通过打印 HARNESS_OFFLOAD_VERIFY_OK；失败退出 1。
"""
import io
import sys


def need(text, needle):
    if needle not in text:
        raise SystemExit("FAIL: missing %r" % needle)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: verify-image-offload-v1.py dsh-llm-pi-ai-index.js")
    text = io.open(sys.argv[1], encoding="utf-8").read()

    cap = "if (/at most \\d+ image\\(s\\)/i.test(message)) return IMAGE_OFFLOAD_REQUIRED_CODE;"
    need(text, cap)
    need(text, "const code = classifyPiAiError(text);")
    need(text, "...code === IMAGE_OFFLOAD_REQUIRED_CODE ? { offloadImages: 1 } : {}")
    need(text, "if (offloadImages > 0) throw new LlmError(")

    # 必须排在 400/413 归类之前，否则 At most N image(s) 先被 INVALID_REQUEST 吃掉。
    cap_at = text.index(cap)
    invalid_at = text.index('if (/\\b400\\b|invalid.?request/i.test(message)) return "INVALID_REQUEST";')
    if cap_at > invalid_at:
        raise SystemExit("FAIL: image-cap branch must precede the 400 branch")
    print("HARNESS_OFFLOAD_VERIFY_OK")


if __name__ == "__main__":
    main()
