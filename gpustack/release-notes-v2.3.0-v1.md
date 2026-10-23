# Release notes · GPUStack 2.3.0 × SM75（2026-10-23）

范围：把本仓的 SM75 自定义 vLLM 后端（镜像仍为 `vllm-sm75-gpustack:v0.1.7-beta2-v7`）接到 GPUStack **v2.3.0 正式版**，并在 8×Tesla T10 上完成兼容性与新特性实测。

## 1 版本与来源

| 项 | 值 |
|---|---|
| GPUStack | `quay.io/gpustack/gpustack:v2.3.0`，镜像内 `v2.3.0 (63a03ba)`，digest `sha256:3d2734fa…` |
| 官方 tag | `v2.3.0`，commit `63a03ba137842153092749fbda1b670c96b10edf` |
| SM75 后端 | `vllm-sm75-gpustack:v0.1.7-beta2-v7`（不变） |
| 后端版本名 | `0.1.7-sm75.1-fast-custom`（不变） |
| 入口覆盖 | `python3 /opt/vllm-sm75/panel-sidecar.py vllm serve`（不变） |

## 2 升级要点

1. 停 server/worker，备份 `${GS_DATA_DIR}/postgresql/data`（宿主路径）与配置；旧容器改名保留并 `restart=no`。
2. 先升 server（同 8180、同数据目录），等 `/healthz` 200 和 `Database migration completed`；本机 Alembic `a7e2c9d4f681`。
3. 后升 worker；**必须保留** `--runtime=nvidia --privileged --security-opt label=disable`，否则 `gpu_devices=[]`，调度器报 “does not support backend version”。
4. 模型实例端口变化后，更新 DSH/客户端 `baseURL`；本机 40052→40000。
5. 回退：停 v2.3.0 → 恢复 `postgresql/data` 备份 → 起旧容器。**不能直接把旧容器接到已迁移的库上。**

## 3 本机实测

| 项 | 读数 |
|---|---|
| worker | READY、v2.3.0、8 张 T10 均 `cuda` |
| 模型 | 原定义迁移；41 个 env、TP8/262144/FP8 KV 等参数不变 |
| 实例 | RUNNING、端口 40000、GPU 0–7；八卡 14,454 MiB |
| 管理面 | 8180 `/healthz` 200、`/` 200、`/docs` 200 |
| 推理 | `/health` 200、`/v1/models` 200、chat 200、Anthropic `/v1/messages` 200 |
| 面板 | 1615 `/` 200 |
| 速度冒烟 | 8K/16K/27.5K 实际 prompt：58.75/58.46/65.13 tok/s（SSE、output 1024、并发 1；不是历史九档同尺） |

## 4 新特性实测

- 已测：离线文档 `/docs`、Anthropic 原生 `/v1/messages`、模型 YAML 导出/导入（dry-run + 副本导入，replicas=0）、revision/回滚（spec 级）、定时伸缩（API 接受并持久化，未执行真实窗口）。
- 智能 Benchmark：预拉 `gpustack/benchmark-runner:v0.0.9` 后完成 Custom 3 请求 smoke：3/3 成功、output 107.53 tok/s、TTFT 937 ms、TPOT 28.70 ms、请求时延 3.67 s；单点 `recommended_rate=1.0`、`coverage_applicable=false`，不代表多并发负载曲线。内置 `Max Throughput`/`Meet Latency SLO` 自动寻优未跑。
- 不适用：PD 分离（运行时常驻约 96 GB ×2 套需 16 卡；权重文件 168.8 GiB）、共享 KV（单副本）、vGPU/MIG（T10 无 MIG）、多副本负载路由。

## 5 已知边界

- 本仓文档中的源码行号仍按上游 **v2.2.3** 取证；v2.3.0 只复核了 run-book §0.1 的兼容面，未逐行复核旧行号。
- `description` 等元数据不进入部署 spec，因此不会触发 revision，也不随 spec 回滚。
- 定时伸缩的真实时间窗口未在本机执行：副本模型运行时常驻约 96 GB（权重文件 168.8 GiB），激活窗口会与生产模型争 8 卡。
- 智能 Benchmark 的 `benchmark-runner:v0.0.9` 是 Docker Hub 镜像；离线部署需提前导入。
