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
- Ubuntu 24.04, NVIDIA driver 580.178.04 (CUDA 13.0), Docker 29.8, NVIDIA Container Toolkit 1.20, GPUStack 2.3.0.
- Full versions: [Runtime environment](#runtime-environment).

### 1 Pull base images

```bash
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.3.0
```

### 2 Build

```bash
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh --two-stage
```

Starts from upstream vLLM 0.30.0; stage 1 produces the intermediate `vllm-sm75:v0.1.7-beta2` (cache only), stage 2 produces `vllm-sm75-gpustack:v0.1.7-beta2-v7`. Success marker: `BUILD_BACKEND_OK ... target=runtime-pin-layer mode=two-stage`.

### 3 Start GPUStack, register the backend version

Full server / worker commands: [run-book §3/§5](gpustack/run-book-v1.md). Registration:

- Version: `0.1.7-sm75.1-fast-custom`
- Image: `vllm-sm75-gpustack:v0.1.7-beta2-v7`
- Entrypoint: `python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`

One-shot write of the entrypoint and panel env: `gpustack/set-panel-autostart-v1.sh --apply` (dry-run by default).

### 4 Deploy the model

Paste into the GPUStack model row:

- Arguments: `gpustack/args-paste-videooff-v1.txt` (24 lines, A route; text-only `args-paste-v1.txt`).
- Environment: `gpustack/env-paste-v1.txt` (25 lines).
- Panel / verifiers: `gpustack/env-paste-panel-v1.txt` (8 lines, port 1615).

Check:

```bash
bash gpustack/lint-paste-v1.sh --args gpustack/args-paste-videooff-v1.txt
# must end with LINT_PASTE_OK env=25 args=24
```

For the 63.30 tok/s production profile, set `--compilation-config` to `FULL_AND_PIECEWISE`, add `--max-num-batched-tokens=8192`, `--gdn-prefill-backend=triton`, `VLLM_FIREFLY_AR=0`, `VLLM_SM75_OPT=auto`; see the [tuning doc](docs/tuning/flash-next-20261027.md).

## Runtime environment

| Item | Tested version |
|---|---|
| OS | Ubuntu 24.04.5 LTS (kernel 6.8.0-146-generic) |
| NVIDIA driver | 580.178.04 (CUDA 13.0) |
| Docker | 29.8.0 (storage overlayfs, cgroup v2) |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.3.0 (v2.2.3 remains the rollback) |
| Hardware | 8×Tesla T10 16 GiB + ≥ 251 GiB host RAM |
| BIOS | SMT=Off; physical host with Ubuntu (P2P 13.1–13.2 GB/s) |

Below / above this table is not guaranteed; see [run-book §0](gpustack/run-book-v1.md).

> GPUStack 2.3.0 workers must keep `--runtime=nvidia --privileged --security-opt label=disable`; omitting the group leaves `gpu_devices=[]`. Upgrade / rollback: [release notes](gpustack/release-notes-v2.3.0-v1.md).

## Performance

2026-10-27 production row (8×T10, SMT Off, 262K, A route, panel nine-tier 512→131072, output 1024, concurrency 1):

| Configuration | Avg output | Avg prefill |
|---|---:|---:|
| Upstream VLLM-SM75 v0.1.7-beta2 (SM75 adaptation only, FP8PLE weights, 2026-10-07/08) | 43.31 | 2742.54 |
| Historical PIECEWISE profile (W4A16-AutoRound, 2026-10-07/08) | 57.04 | 2762.51 |
| **Current production row (FULL + batch 8192 + triton/gdn_tc + AR=0, W4A16, 2026-10-27)** | **63.30** | **3485.13** |

- 8K/16K/32K/64K/128K output **63.52/63.15/63.00/62.48/61.50**; TTFT **2013/4127/9200/19758/42595 ms**.
- Current vs upstream **+46.2%** (cross-session, different weights; delivered PLE is BF16 vs FP8 in the upstream FP8PLE set, so this is not a precision-for-speed trade); same-weights FP8PLE is +33.3% (57.73).
- The direct-ruler 2019/8643 ms and 59.6–61.6 tok/s are a separate ruler; **do not subtract them from the panel 63.30**; details and negative results: [tuning doc](docs/tuning/flash-next-20261027.md).
- 63.30 was measured on the 2026-10-27 opt15 thin layer; the `-v7` delivery image ships the same gated package and can reproduce it with the same production knobs (use your own same-ruler numbers). Without them it is the historical PIECEWISE profile (57.04).

## Gated package

The image ships `vllm_sm75`, auto-installed through `vllm.general_plugins`; with no switch set it equals the original baseline.

- `VLLM_SM75_OPT=auto`: only items that passed the same-ruler A/B (≥1%, 3–5 run median); currently `gemv_m1` (decode +1.4%) and `gdn_tc` (prefill +6.5/12.0/10.1%, requires `--gdn-prefill-backend=triton`).
- `VLLM_SM75_OPT=0`: disable all; per-item override with `VLLM_SM75_<ITEM>=0/1`.
- Item status (2 verified, 4 rejected, 2 measurement-only) and negative results: [vllm_sm75/README.md](vllm_sm75/README.md), [tuning doc](docs/tuning/flash-next-20261027.md).

## Docs

- [docs/README.md](docs/README.md): docs index.
- [run-book](gpustack/run-book-v1.md): zero-to-serving, paste blocks, troubleshooting.
- [tuning](docs/tuning/flash-next-20261027.md): 63.30 baseline, negative results, hardware floors.
- [backend card](gpustack/backend-version-card-v1.md): registration form, field by field.
- [compat matrix](gpustack/compat-matrix-v1.md): capabilities and reading boundaries.
- [release notes](gpustack/release-notes-v2.3.0-v1.md): GPUStack 2.3.0.
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

## GPUStack 2.3.0

Tested offline docs, Anthropic `/v1/messages`, model YAML export/import, deployment revisions / rollback; PD disaggregation, shared KV, vGPU / MIG and multi-replica routing are not applicable here. Upgrade / rollback and untested items: [release notes](gpustack/release-notes-v2.3.0-v1.md) and run-book §0.1.

[中文](README.md)
