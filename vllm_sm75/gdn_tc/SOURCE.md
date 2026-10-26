Borrowed from vllm-turing (github cookjohn/vllm-turing, snapshot 290fa9c, 2026-10-24 copy at K:/sm75/杂项/vllm-turing-20261024).
Files: gdn.py -> __init__.py, _build.py, csrc/gdn_chunk.cu.
Changes: package path, env var VLLM_TURING_GDN* -> VLLM_SM75_GDN_TC*, off-switch file name.
Kernel itself is unmodified; it targets sm_75 with mma.sync m16n8k8 + ldmatrix.
