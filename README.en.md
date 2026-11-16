# vLLM-SM75 × GPUStack custom backend

Turns SM75 vLLM into a GPUStack custom backend so T4 / T10 Turing cards can serve large models. The same image can also run the standalone Ultra / panel; both forms share one build chain and the SM75 adaptation.

## What you get

- **GPUStack custom backend**: `vllm serve` in a GPUStack model row; 8×T10, 262K, A route (images 4 / video 0).
- **Standalone Ultra / panel**: same image, `runtime-entrypoint.py`, console + Ultra workbench + BenchLocal verifiers.
- **Two SM75 layers**: pcieipc (8-GPU PCIe direct allreduce) and incple (AutoRound / INC quantization whitelist); upstream QSA / FlashQLA / scheduler are enabled by environment variables.
- **Gated package `vllm_sm75`**: add-only, opt-in; `VLLM_SM75_OPT=auto` enables only verified items; `0` disables everything and equals the original baseline.

## Quick start

### 0 Prerequisites

- 8×Tesla T10 16 GiB each, host RAM ≥ 251 GiB, data disk ≥ 250 GB.
- Ubuntu 24.04, NVIDIA driver 580.178.04 (CUDA 13.0), Docker 29.8, NVIDIA Container Toolkit 1.20, GPUStack v2.3.1.
- Full versions: [Runtime environment](#runtime-environment).

### 1 Pull base images

```bash
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.3.1
```

### 2 Build

**0→1 (tutorial baseline: upstream vLLM → SM75 base → GPUStack backend)**

```bash
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh --two-stage
```

Starts from upstream vLLM 0.30.0; stage 1 produces the intermediate `vllm-sm75:v0.1.7-beta2` (cache only), stage 2 produces the public one-command image `vllm-sm75-gpustack:v0.1.7-beta2-v7`. Success marker: `BUILD_BACKEND_OK ... target=runtime-pin-layer mode=two-stage`.

**Upgrade to the latest production layers (optional; not the 0→1 upstream)**

The 0→1 output is `vllm-sm75-gpustack:v0.1.7-beta2-v7`; v7/opt15 are intermediate or delivery bases, **not the upstream**. Choose one:

- One command from upstream to latest (0→1 + upgrade): `bash gpustack/build-latest-from-zero-v1.sh` (use `SKIP_V7=1` if v7 already exists), producing `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1` and `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1-sqlprompt-audit`.
- With opt15 (internal optimized base), from the repo root:

```bash
docker build -f gpustack/Dockerfile.final-v1-20261029 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1 .
docker build -f gpustack/Dockerfile.sql-prompt-audit-v20261029 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1-sqlprompt-audit .
```

- `final-v1`: current production image (parent opt15; console maxTokens, speedtest v3.5 quota/upload, SQL finalanswer). **The live production model is row 48 Swift-1.5 on this very image (`7267ff1c3db0`) — what changed is the model weights and the row, not the image.**
- `sqlprompt-audit`: latest SQL prompt-audit layer (parent final-v1; fiscal-period format rule + tool-result truncation).
- `compat1` (optional experimental layer): Turing 0.32 compat thin layer (parent final-v1; all functional switches default-off; built, not registered), recipe `gpustack/Dockerfile.compat-v1-20261116`, see [tuning](docs/tuning/compat-turing032-20261116.md).

### 3 Start GPUStack, register the backend version

Full server / worker commands: [run-book §3/§5](gpustack/run-book-v1.md). Registration:

**0→1 baseline (tutorial):**
- Version: `0.1.7-sm75.1-fast-custom`
- Image: `vllm-sm75-gpustack:v0.1.7-beta2-v7`

**Latest upgrade layers (optional):**
- Version: `0.1.7-sm75.1-sqlprompt-audit-custom`
- Image: 0→1 path `vllm-sm75-gpustack:v0.1.7-beta2-v7-final-v1-sqlprompt-audit`; opt15 path `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1-sqlprompt-audit`
- Current production: `0.1.7-sm75.1-final-v1-custom` + `vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-final-v1`; model row **48 Swift-1.5** (served name `EXP-swift15-qwen3.8-flash-next-w4a16-autoround`, port 40000, 25 arguments) is live, row 26 (Intel AutoRound) stands by at replicas=0
- Entrypoint: `python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`

One-shot write of the entrypoint and panel env: `gpustack/set-panel-autostart-v1.sh --apply` (dry-run by default).

### 4 Deploy the model

Paste into the GPUStack model row:

- Arguments: `gpustack/args-paste-videooff-v1.txt` (24 lines, A route; text-only `args-paste-v1.txt`). **This is the 0-to-1 baseline block** (PIECEWISE / batch 4096 / `flashqla_sm75` / `--generation-config=vllm`); the production profile changes four more knobs (below and in the tuning docs), and live row 48 also carries `--override-generation-config`, which the 0-to-1 block does not contain (without it top_p/top_k are dropped).
- Environment: `gpustack/env-paste-v1.txt` (25 lines).
- Panel / verifiers: `gpustack/env-paste-panel-v1.txt` (8 lines, port 1615).

Check:

```bash
bash gpustack/lint-paste-v1.sh --args gpustack/args-paste-videooff-v1.txt
# must end with LINT_PASTE_OK env=25 args=24
```

To reproduce the **2026-10-27 Intel opt15 historical profile (63.30 tok/s)**, set `--compilation-config` to `FULL_AND_PIECEWISE`, add `--max-num-batched-tokens=8192`, `--gdn-prefill-backend=triton`, `VLLM_FIREFLY_AR=0`, `VLLM_SM75_OPT=auto`; see the [historical tuning doc](docs/tuning/flash-next-20261027.md). **The live production profile is Swift-1.5 row 48 (decode mean 66.68)** — same engine knobs, different weights, plus `--override-generation-config`; live runtime values and verdicts: [Swift tuning page](docs/tuning/swift15-prod-20261116.md).

## Runtime environment

| Item | Tested version |
|---|---|
| OS | Ubuntu 24.04.5 LTS (kernel 6.8.0-146-generic) |
| NVIDIA driver | 580.178.04 (CUDA 13.0) |
| Docker | 29.8.0 (storage overlayfs, cgroup v2) |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.3.1 (tested 2026-11-14, commit `458030e`; rollback chain v2.3.0 -> v2.2.3, and a rollback must restore the pre-upgrade pg_dump backup) |
| Hardware | 8×Tesla T10 16 GiB + ≥ 251 GiB host RAM |
| BIOS | SMT=Off; physical host with Ubuntu (P2P 13.1–13.2 GB/s) |

Below / above this table is not guaranteed; see [run-book §0](gpustack/run-book-v1.md).

> GPUStack v2.3.1 workers (same as 2.3.0) must keep `--runtime=nvidia --privileged --security-opt label=disable`; omitting the group leaves `gpu_devices=[]`. Upgrade / rollback: [release notes v2.3.1](gpustack/release-notes-v2.3.1-v1.md) ([v2.3.0](gpustack/release-notes-v2.3.0-v1.md) is the historical record).

## Performance

**Live production row (2026-11-16, Swift-1.5, model row 48)**: 8×T10, SMT Off, 262K, image `…opt15-final-v1` (the same image as the historical row), panel three tiers 8192/16384/32768, output 1024, concurrency 1, temp 1, topP 0.95:

| Prompt | TTFT ms | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| 8192 | 2030.70 | 4039.33 | **66.61** |
| 16384 | 4333.09 | 3783.84 | **66.48** |
| 32768 | 9368.64 | 3498.81 | **66.94** |

Decode mean over the three tiers: **66.68** (**one run per tier, no multi-run median**). **The panel's aggregate "average prefill" field was not captured this round, so it is not quoted here**: the arithmetic mean of the three tiers is 3773.99, which is a different formula from the panel field ((sum of tiers + last tier) / (tiers + 1)) and must not be passed off as it. Served name `EXP-swift15-qwen3.8-flash-next-w4a16-autoround`, port 40000, 25 arguments (`--block-size=16`, no absorbed optimizations mixed in). Live runtime values, the memory budget convention (**allocatable ceiling 15929 MiB per card**) and the four parameter verdict groups: [Swift tuning page](docs/tuning/swift15-prod-20261116.md).

**Historical baseline (2026-10-27 production row, Intel AutoRound, opt15 thin layer)**: 8×T10, SMT Off, 262K, A route, panel nine-tier 512→131072, output 1024, concurrency 1:

| Configuration | Avg output | Avg prefill |
|---|---:|---:|
| Upstream VLLM-SM75 v0.1.7-beta2 (SM75 adaptation only, FP8PLE weights, 2026-10-07/08) | 43.31 | 2742.54 |
| Historical PIECEWISE profile (W4A16-AutoRound, 2026-10-07/08) | 57.04 | 2762.51 |
| 2026-10-27 Intel opt15 production row (FULL + batch 8192 + triton/gdn_tc + AR=0, W4A16) — **historical baseline, superseded by Swift row 48 on 2026-11-16** | 63.30 | 3485.13 |

- Historical window (2026-10-27 Intel profile): 8K/16K/32K/64K/128K output **63.52/63.15/63.00/62.48/61.50**; TTFT **2013/4127/9200/19758/42595 ms**.
- Historical window (2026-10-27 Intel profile) vs upstream **+46.2%** (cross-session, different weights; delivered PLE is BF16 vs FP8 in the upstream FP8PLE set, so this is not a precision-for-speed trade); same-weights FP8PLE is +33.3% (57.73). **Both percentages are historical comparisons and must not be subtracted from the live 66.68.**
- The direct-ruler 2019/8643 ms and 59.6–61.6 tok/s are a separate ruler; **do not subtract them from the panel 63.30, nor from the live 66.68**; details and negative results: [historical tuning doc](docs/tuning/flash-next-20261027.md).
- 63.30 was measured on the 2026-10-27 opt15 thin layer; the public one-command `-v7` image ships the same gated package and can reproduce it with the same production knobs (use your own same-ruler numbers). final-v1 / sqlprompt-audit only change the console and panel extensions, not engine parameters; the 63.30 profile is unchanged. Without them it is the historical PIECEWISE profile (57.04).
- **The gap between the live 66.68 and 63.30 comes from the model weights (Swift-1.5 vs Intel AutoRound), not from an image or platform upgrade**: both run the same final-v1 image. The Swift line's same-version comparison during its trial run was +5.5% (against the final-v1 Intel profile at 63.4). The historical nine-tier sweep was not re-run after the v2.3.1 upgrade, so **the 66.68-vs-63.30 difference must not be booked as an upgrade gain**.

## Gated package

The image ships `vllm_sm75`, auto-installed through `vllm.general_plugins`; with no switch set it equals the original baseline.

- `VLLM_SM75_OPT=auto`: only items that passed the same-ruler A/B (≥1%, 3–5 run median); currently `gemv_m1` (decode +1.4%) and `gdn_tc` (prefill +6.5/12.0/10.1%, requires `--gdn-prefill-backend=triton`).
- `VLLM_SM75_OPT=0`: disable all; per-item override with `VLLM_SM75_<ITEM>=0/1`.
- Item status (2 verified, 4 rejected, 2 measurement-only) and negative results: [vllm_sm75/README.md](vllm_sm75/README.md), [tuning doc](docs/tuning/flash-next-20261027.md).

## Docs

- [docs/README.md](docs/README.md): docs index.
- [run-book](gpustack/run-book-v1.md): zero-to-serving, paste blocks, troubleshooting.
- [tuning (live)](docs/tuning/swift15-prod-20261116.md): Swift-1.5 row 48 baseline (decode mean 66.68), live runtime values, the 15929 MiB/card memory convention, four parameter verdict groups and headroom boundaries.
- [tuning (historical)](docs/tuning/flash-next-20261027.md): the 2026-10-27 Intel opt15 63.30 baseline, negative results, hardware floors.
- [compat tuning](docs/tuning/compat-turing032-20261116.md): Turing 0.32 compat layer (default-off), block400 capacity tier, MTP conclusion.
- [backend card](gpustack/backend-version-card-v1.md): registration form, field by field (includes the 2026-10-29 internal deployment addendum).
- [final-v1 recipe](gpustack/Dockerfile.final-v1-20261029) / [SQL prompt-audit recipe](gpustack/Dockerfile.sql-prompt-audit-v20261029): latest image layers.
- [compat matrix](gpustack/compat-matrix-v1.md): capabilities and reading boundaries.
- [release notes](gpustack/release-notes-v2.3.1-v1.md): GPUStack v2.3.1 (live); [v2.3.0](gpustack/release-notes-v2.3.0-v1.md) is the historical record.
- [design](docs/design/): design and research notes.
- [vllm_sm75](vllm_sm75/README.md): gated package.

## Repo layout

```text
gpustack/   deliverables: run-book, paste blocks, Dockerfile.v1, gates, scripts
docker/     upstream build assets (base Dockerfile, helpers)
ultra/      console + BenchLocal sources
vllm/       upstream SM75 adaptation and attention overlay
vllm_sm75/  gated optimization package
```

## Upstream and license

- Base: `vllm/vllm-openai:v0.30.0-cu129` (tag + digest pinned); upstream SM75 adaptation in `vllm/` and `docker/`.
- New deliverables live in `gpustack/`; upstream files are protected by the freeze table, exceptions recorded in `NOTICE`.
- License: Apache-2.0; third-party notices in `NOTICE` and `gpustack/backend-version-card-v1.md`.

## Acknowledgements

This repository builds on the following open-source work:

- **VLLM-SM75 (fishensw)**: the SM75/Turing vLLM adaptation and Ultra console source (Apache-2.0); upstream base vLLM 0.30.0; per-item provenance in `NOTICE`.
- **vLLM, FlashInfer, PyTorch, Triton, NCCL**: inference runtime, attention/communication kernels and collective communication.
- **GPUStack**: model deployment, backend versions and operations control plane.
- **AutoRound (Intel)** and **Qwen3.8-Flash-Next**: W4A16 quantization toolchain and base model; this repo does not redistribute weights.
- **DeepSeek Harness (@deepseek-ai/dsh)**: the session and plugin framework behind the console workbench.
- **llm_speedtest (gengchaogit)**: the speed test in the console's "Model Test" page; its credits include Bilibili creator "纸鸢随风", the DeepSeek-R1-0528 open speed-test script and llm-inference-bench.
- Thanks also to the vLLM, SGLang, Ollama and llama.cpp communities for public benchmarks and tuning practices.

Third-party licenses, versions and provenance: `NOTICE` and `gpustack/backend-version-card-v1.md`.

## GPUStack v2.3.1 (live)

Upgraded on 2026-11-14 to `quay.io/gpustack/gpustack:v2.3.1` (commit `458030e`): only the server/worker images change; custom backend versions, model definitions and credentials stay in the data_dir and remain usable as-is, and the worker runtime flags are kept as a group. Upgrade / rollback and the **unverified gates**: [release notes v2.3.1](gpustack/release-notes-v2.3.1-v1.md) and run-book §0.2.

- **Feature gate**: Decision Model Serving (`/v1/systemone` + Decision Playground, typed choice/score/boolean verdicts) **requires vLLM >= 0.32.0**; this delivery's SM75-optimized build is **0.30, so it cannot be used**.
- **Unverified, not claimed as fixed**: whether upstream #6660 ("backend 4xx turned into 503") is really fixed (the release notes only mention non-streaming, while our reproduction also hit streaming); whether the hardcoded `GPUSTACK_MODEL_VRAM_CLAIM` can be dropped (v2.3.1 supports CPU-offloaded lookup-table VRAM estimation including PLE/Engram, but it must not be removed before comparing old and new estimates).
- **Historical (2.3.0, tested 2026-10-23)**: offline docs, Anthropic `/v1/messages`, model YAML export/import, revisions / rollback were tested; those surfaces were not re-tested item by item under v2.3.1. PD disaggregation, shared KV, vGPU / MIG and multi-replica routing remain inapplicable on a single-replica 8×T10 form. See [release notes v2.3.0](gpustack/release-notes-v2.3.0-v1.md) and run-book §0.1.

[中文](README.md)
