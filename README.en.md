# vLLM-SM75 v0.1.7 beta2


## 0. Using this repo as a GPUStack backend

This repo wires the SM75 build of vLLM into **GPUStack** as a custom backend, so older Turing
cards (T4 / T10) can serve large models from inside GPUStack. Upstream vLLM has no accelerated
path for Turing: Marlin and FP8 W8A8 are unsupported, and 7.5 is only the lowest compute capability
it will run on. VLLM-SM75 does the SM75 adaptation. On top of it we ship two code layers - PCIe
direct all-reduce across the eight cards (pcieipc) and the AutoRound quantization whitelist
(incple) - and we turn on the base image's QSA, FlashQLA and custom scheduler through environment
variables. Section 0.5 measures what that whole set is worth.

What you need: eight 16 GiB cards and 251 GiB of host RAM. The checkpoint we run weighs
168.7 GiB, more than the total VRAM, and part of it stays in host memory - too little RAM gets you
an OOM kill during load, which looks like a broken image.

Follow `gpustack/run-book-v1.md` start to finish: build the image, start server and worker,
register the backend version, deploy, then run the four acceptance gates. The two blocks you paste
into the form are `gpustack/args-paste-v1.txt` (24 lines) and `gpustack/env-paste-v1.txt`
(25 lines; line 25, the VRAM claim, is mandatory - without it the model queues forever and says nothing).
After pasting, run `bash gpustack/lint-paste-v1.sh`; the last line must read `LINT_PASTE_OK env=25 args=24`.

Why each parameter is set that way, and how to re-derive it for other hardware:
`gpustack/backend-version-card-v1.md` and `gpustack/compat-matrix-v1.md`.

**Optional web console, baked into the same image (no second container)**: the backend image built in
step 2 already carries the console (node v22.23.2 + `/opt/sm75-workbench`). To start the console
together with the model, point the backend version entrypoint at the wrapper and add three env vars
to the model row:

- backend version `0.1.7-sm75.1-custom` `entrypoint`: `python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`
- model env: `SM75_PANEL_AUTOSTART=1`, `SM75_CONSOLE_HOST=0.0.0.0`, `SM75_CONSOLE_ROOT=/var/lib/gpustack/panel`
- the console listens on **1616**; the login token is `$SM75_CONSOLE_ROOT/key`, the log is `panel.log`, and state lives under the worker data dir, so it survives restarts.
- without `SM75_PANEL_AUTOSTART` the wrapper is a transparent passthrough; engine behaviour is unchanged.

The standalone console command, the three BenchLocal verifier containers and the radar-chart steps
are in run-book section 11.

[中文](README.md) · [Release notes](docs/releases/v0.1.7-beta2.md) · [Model profiles](docs/configuration-v0.1.7.md) · [BenchLocal](docs/benchlocal.md)

## 0.5 Speed

One 8xT10 machine, one setup: the harness is the Prefill & Decode benchmark v3 (chao mod), run
on the image's own Ultra/panel form, concurrency 1, nine prompt lengths from 512 to 131072,
every cell 1/1 success:

| Configuration | Mean output | Mean prefill |
|---|---|---|
| VLLM-SM75 v0.1.7-beta2 (baseline, already SM75-adapted) | 43.31 tok/s | 2742.54 |
| same weights, with the set above applied - FP8PLE | 57.73 tok/s | 2749.96 |
| with the set above applied - W4A16-AutoRound | 57.04 tok/s | 2762.51 |

Output is +33.3%, prefill barely moves: the gain is on decode. How to read these three runs,
which numbers must not be compared, and why none of them is a GPUStack acceptance line -
see the last section of `gpustack/compat-matrix-v1.md`.



## 1. Overview

vLLM 0.30.0 adapted for NVIDIA Turing / SM75, including Tesla T4 and T10. Supports Qwen3.8 27B FP8 / AWQ / NVFP4 and Flash Next W4A16, with GDN, FP8 KV, MTP, DFlash2, DSpark, Firefly, CPU KV, power management and monitoring.

Use one image: **`vllm-sm75:v0.1.7-ultra-beta2`**. Ultra includes the console, chat workspace and model tests. Set `SM75_EDITION=standard` to run only the inference API using the same model profiles and engine.

QQ community: **878924874**.

## 2. Updates

- DSpark k5 profiles for 27B FP8 / NVFP4, draft KV grouping and graph capture support.
- Prefix cache configuration, CPU KV shared-memory checks and LMCache configuration hints.
- Nine native BenchLocal suites, all execution dependencies in the Ultra image, generation settings, per-case results, radar charts and portable reports.
- Model/profile launch improvements, profile deletion, API address/key copying, monitoring and UI fixes.

## 3. Standard mode and Ultra

Both use the same inference engine, model profiles, KV settings and power controls. Standard mode exposes port 8000; Ultra adds the management UI on port 1615, Harness workspace, Watcher usage insights and integrated model tests.

## 4. Performance reference

Existing measurements and conditions are available for [27B](docs/validation/v0.1.6-full-images.md) and [Flash Next](docs/flash-next-performance.md). The overview table in the [Chinese README](README.md#4-性能参考) identifies the measured versions.

## 5. Build and start

Requires Linux x86_64, Docker / BuildKit, Git, and an NVIDIA driver plus NVIDIA Container Toolkit on the runtime host. Allow access to the pinned upstream package/download services and about 200 GiB of build storage.

```bash
git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
EDITION=ultra MAX_JOBS=1 bash docker/build.sh

export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
export VLLM_API_KEY='replace-with-your-own-long-random-key'
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1  # actual host path
EDITION=ultra bash docker/run.sh
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

The full build produces one Ultra image containing all nine BenchLocal suites. The startup script reuses that image for resource-limited CPU execution containers; no additional verifier images are needed.

Open `http://<server>:1615`, register a model and select its profile. For 27B FP8 with DSpark, use `SM75_PROFILE=qwen38-27b-fp8-dspark5` and set `MODEL` / `DRAFT_MODEL`. CPU KV 16 GiB uses `CPU_KV_GIB=16` and `SHM_SIZE=32g`.

Keep existing `/data`, cache, `/dsh/home` and `/dsh/workspace` mounts when upgrading. See [build details](docker/BUILD.md), [model profiles](docs/configuration-v0.1.7.md) and [BenchLocal setup](docs/benchlocal.md).
