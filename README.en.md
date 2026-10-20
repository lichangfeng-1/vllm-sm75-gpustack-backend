# vLLM-SM75 x GPUStack custom backend (v0.1.7-beta2 line)

Turns SM75 vLLM into a GPUStack custom backend so Turing cards (T4 / T10) can serve large models
inside GPUStack. This repo covers that line only; the upstream standalone Ultra / workbench
deployment is out of scope here (see "Relationship to upstream" at the end).

---

## 1 What this is, and what we add

Official vLLM has no acceleration path on Turing (Marlin and FP8 W8A8 are unsupported; 7.5 is merely
the minimum compute capability it will run on). Upstream VLLM-SM75 did the SM75 adaptation. We add
two layers on top:

- **pcieipc**: 8-GPU PCIe direct allreduce (backports FlashInfer 0.7.0's PCIe-IPC into the bundled 0.6.18);
- **incple**: AutoRound / INC quantization whitelist patch.

A group of environment variables switches on capabilities the base already has (QSA, FlashQLA, the
custom scheduler) - that is *enabling*, not our implementation.

## 2 Runtime environment (tested lower bound, no upper-bound guarantee)

| Item | Version we tested |
|---|---|
| OS | Ubuntu 24.04.5 LTS (kernel 6.8.0-146-generic) |
| NVIDIA driver | 580.178.04 (CUDA 13.0) |
| Docker | 29.8.0 (storage `overlayfs`, cgroup v2) |
| containerd | 2.3.5 |
| NVIDIA Container Toolkit | 1.20.0 |
| GPUStack | quay.io/gpustack/gpustack:v2.2.3 |
| Hardware | 8x Tesla T10 16 GiB + 251 GiB host RAM |

Below this we do not guarantee (and have not tested); above it we do not guarantee either. Test your own combination.

## 3 Quick start

Full steps and troubleshooting: **`gpustack/run-book-v1.md`**. Minimal sequence:

```bash
# 0) Pull the two base images first (otherwise the first build stalls on a ~42 GiB base-image pull)
docker pull vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688
docker pull quay.io/gpustack/gpustack:v2.2.3

# 1) Clone this repo
git clone https://github.com/lichangfeng-1/vllm-sm75-gpustack-backend.git
cd vllm-sm75-gpustack-backend

# 2) One command: upstream vLLM base -> custom backend (single build, fastest profile by default)
#    The repo merges the upstream docker/Dockerfile with our five layers into
#    gpustack/Dockerfile.allinone (reproducible generator), so one docker build goes from the
#    vllm/vllm-openai base to the final image -- no intermediate base tag.
#    The last layer pins transformers/tokenizers to 5.15.1/0.22.2 (measured 56.83-57.04 tok/s).
#    Output tag: vllm-sm75-gpustack:v0.1.7-beta2-v6.
#    Rollback to base-image versions (5.17.0/0.23.2): --no-tf-pin (output tag -v4-nopin).
#    Legacy two-stage build (troubleshooting only): --two-stage.
bash gpustack/preflight-sources-v1.sh
bash gpustack/build-backend-v1.sh --two-stage

# 3) Start GPUStack (server/worker commands: run-book sections 3 and 5)

# 4) Register the backend version (default = fastest): 0.1.7-sm75.1-fast-custom,
#    image = vllm-sm75-gpustack:v0.1.7-beta2-v6,
#    entrypoint = python3 /opt/vllm-sm75/panel-sidecar.py vllm serve
#    (set-panel-autostart-v1.sh also writes the panel env and SM75_EXTENSIONS)
#    Rollback route: 0.1.7-sm75.1-custom + vllm-sm75-gpustack:v0.1.7-beta2-v4

# 5) Deploy the model: paste the blocks
#    gpustack/args-paste-videooff-v1.txt (24 lines, A route: images on, video off)
#    + gpustack/env-paste-v1.txt (25 lines)
#    add gpustack/env-paste-panel-v1.txt (5 lines, fixed port 1615) for the console and test suites
bash gpustack/lint-paste-v1.sh --args gpustack/args-paste-videooff-v1.txt   # LINT_PASTE_OK env=25 args=24
```

**The chain end must target `runtime-pin-layer`** (default via build-backend-v1.sh): it is a superset
(four layers + console payload + BenchLocal verifier runtime + vendored test assets + the runtime pin),
so one image serves as model backend, console host and verifier host. `--target panel-layer` builds the
unpinned v4 for rollback; `--no-tf-pin` does the same through the script.

## 4 Optional web console (same image, same model row)

The image already carries the console (node v22.23.2 + `/opt/sm75-workbench`). To start it together
with the model, add these env vars to the model row:

```
SM75_PANEL_AUTOSTART=1
SM75_CONSOLE_HOST=0.0.0.0
SM75_CONSOLE_ROOT=/var/lib/gpustack/panel
SM75_CONSOLE_PORT=1615
SM75_EXTENSIONS=panel,harness,benchlocal,speedtest,sql
```

- Console listens on **1615** for both routes (the old-engine thin image defaults to 1615; pinning it here keeps the v4 rollback on the same port). Login token at `$SM75_CONSOLE_ROOT/key`; state lives under the worker data dir, so it survives restarts.
- Without `SM75_PANEL_AUTOSTART` the wrapper is a transparent passthrough.
- Standalone console, the three BenchLocal verifier containers and the radar steps: run-book section 11.

> Known limit: the **usage / workbench / plugins** native pages and the BenchLocal **harness** depend on
> the DSH workbench (`/opt/harness`, 585 MB), which today only runs in the Ultra workbench form. An
> "extension switch" that starts it with the model is in progress; until then use the standalone
> container path in run-book section 11.

## 5 Docs map

| Question | Where |
|---|---|
| Step-by-step from zero to inference | `gpustack/run-book-v1.md` |
| What every parameter does and how to tune it | run-book **section 12** |
| Just copy-paste | run-book **section 13** + the two `.txt` blocks |
| Backend registration form, field by field | `gpustack/backend-version-card-v1.md` |
| Capability mapping and why these numbers are not an acceptance line | `gpustack/compat-matrix-v1.md` |
| One-command console autostart | `gpustack/set-panel-autostart-v1.sh` (dry-run by default, `--apply` to write) |

## 6 Speed reference (panel form, not a GPUStack acceptance line)

Same 8xT10, same ruler (Prefill & Decode benchmark v3, concurrency 1, nine tiers 512 to 131072):

| Configuration | Avg output | Avg prefill |
|---|---|---|
| VLLM-SM75 v0.1.7-beta2 (baseline, SM75 adaptation only) | 43.31 tok/s | 2742.54 |
| Same weights + full optimization set, FP8PLE | 57.73 tok/s | 2749.96 |
| Full optimization set, W4A16-AutoRound | 57.04 tok/s | 2762.51 |

+33.3% on the output side, prefill essentially unchanged. **These were measured in the panel form**,
not in GPUStack form; for the GPUStack-form nine-tier numbers see run-book section 10a (52.81, with a
different configuration window). How to read the tables: `gpustack/compat-matrix-v1.md`, last section.

## 7 Repo layout

```
gpustack/        everything this line delivers: run-book, paste blocks, Dockerfile.v1, gates, scripts
docker/          upstream build assets (base Dockerfile, helpers) - protected by the freeze gate
ultra/source/    console + BenchLocal sources (layer 4 copies from here)
vllm/            upstream SM75 adaptation and our attention overlay
```

## 8 Relationship to upstream, license

- Base: `vllm/vllm-openai:v0.30.0-cu129` (tag + digest pinned).
- New deliverables live in `gpustack/`; upstream files are protected by the 551-file freeze gate
  (`gpustack/gate-tree-unchanged-v1.sh`); exceptions are recorded in `NOTICE`.
- License: Apache-2.0 (see `LICENSE`); third-party notices in `NOTICE` and `gpustack/backend-version-card-v1.md`.

[中文](README.md)
