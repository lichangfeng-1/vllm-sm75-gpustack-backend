# run-book v1 —— 从零到模型可推理（Linux 宿主）

适用范围：8×Tesla T10（sm_75，16 GiB/卡）＋ GPUStack v2.2.3 ＋ 本仓自建 vLLM 后端镜像。
本文件只给"照做能跑"的最小命令序列；每条参数为什么这么设、换硬件怎么改，见 `gpustack/backend-version-card-v1.md` 与 `gpustack/compat-matrix-v1.md`。

**标注约定**：`【待实跑复核】`＝该命令由源码行号与既有实跑记录推出，尚未在本机复跑；其余步骤为实跑验证过的形制。

---

## 0 前置（先核这三格，缺任何一格都会失败）

| 项 | 要求 | 为什么 |
| --- | --- | --- |
| 主机内存 | **约 251 GiB** | 参数块带 `--engram-config={"cpu_offload":true}`，一部分权重常驻主机内存。64 GB 的机器会在加载阶段被 OOM killer 杀掉，且现象像"镜像坏了" |
| 磁盘 | `/data` 可用 ≥ 250 G | 权重 17 枚 `*.safetensors` 合计 `181,196,007,016 B`（≈168.8 GiB），另需镜像与缓存余量 |
| 网卡 | 记下宿主内网 IP，下面统一用 `HOST_IP` | worker 与浏览器都要能到达它。`export HOST_IP=192.0.2.10` 换成你的地址 |

驱动版本下限**本仓未验**（面板线撞过 NVAPI 与驱动版本不匹配，属另一条形制），此处不写成要求。

## 1 拉仓

```bash
git clone 你的仓地址 vllm-sm75-gpustack-backend
cd vllm-sm75-gpustack-backend
```

`gpustack/.gitattributes` 已给本目录下的 `*.txt`／`*.patch` 写了 `text eol=lf`（仓根那份属上游冻结面，不动它），正常 `git clone` 即可。
若你的 git 全局设过 `core.autocrlf=true`，用这条覆盖一次更稳：`git -c core.autocrlf=false clone …`。

## 2 探活并构建后端镜像

```bash
bash gpustack/preflight-sources-v1.sh          # 只读 HEAD 探测，红就别往下走
bash gpustack/build-gpustack-v1.sh             # 补丁门五道，不执行 docker build
bash gpustack/build-gpustack-v1.sh --build     # 绿后组装并执行 docker build
docker images vllm-sm75-gpustack               # 期望 tag v0.1.7-beta2-v1
```

构建期会拉 `vllm/vllm-openai:v0.30.0-cu129` 作基座（十 GB 量级，首次很慢，建议后台跑并跟看日志）。
两枚构建期下载件（`flashinfer_cubin` wheel 与 `nvidia-pstated` 二进制）都钉在上游 Release ＋ sha256，本仓不托管、不重新分发（后者是 GPL-3.0-only）。

## 3 起 GPUStack server

```bash
docker create --name sm75-gpustack-server --hostname sm75-gpustack-server \
  --network bridge --restart no --env TZ=Asia/Shanghai \
  -p ${HOST_IP}:8180:80 \
  --mount type=bind,source=/data/gpustack-data,target=/var/lib/gpustack \
  quay.io/gpustack/gpustack:v2.2.3
docker start sm75-gpustack-server
```

**两个坑，都是实跑撞过的**：

- **不要加 `--system-default-container-registry`**。加了它，后端注册表里不含 registry 段的镜像名会被拼成 `quay.io/vllm-sm75-gpustack:…` 去远端拉 ⇒ `Failed to pull image`，而本地明明有这枚镜像；表现是"参数块一行都没被执行"。拼接规则 `gpustack/utils/config.py:87-108`。已经设过就清掉：`PUT /v2/config {"system_default_container_registry":""}`（该字段在运行时白名单 `gpustack/utils/config.py:9-12`，可热改）。
- **不要用 `--network host`**。GPUStack 内嵌的 Higress 网关会自己开一串监听口，`--host-ip` 管不到它们 ⇒ 暴露面比设计大。这里用 bridge ＋ 显式 `-p 宿主IP:8180:80`，只放这一个口。

## 4 取初始口令并登录

```bash
docker exec sm75-gpustack-server cat /var/lib/gpustack/initial_admin_password
```

浏览器开 `http://${HOST_IP}:8180` 登录并改口令。**口令别贴进截图、群或任何公开渠道。**

## 5 起 worker

同机部署**不需要 token**：worker 取 token 的路径是 `gpustack/config/config.py:312-313` 的 `read_registration_token(self.data_dir)`，读的是共享 data 目录里的文件，而本形制下 server 与 worker 挂的是同一份 `/data/gpustack-data`。

```bash
docker create --name sm75-gpustack-worker --hostname sm75-gpustack-worker \
  --restart no --env TZ=Asia/Shanghai \
  --privileged --network host --runtime nvidia \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  --mount type=bind,source=/data/gpustack-data,target=/data/gpustack-data \
  quay.io/gpustack/gpustack:v2.2.3 \
  --server-url http://${HOST_IP}:8180 \
  --advertise-address ${HOST_IP} \
  --worker-name sm75-gpustack-worker
docker start sm75-gpustack-worker
```

- **`--advertise-address` 不能省**。多网卡机器上 worker 会通告错地址，表现是"worker 显示 ready，但模型起不来或连不上"，很难查。
- 跨机部署才需要 token：用 UI 的 Workers → 添加 worker 给出的整条命令，`--token` 会出现在命令行上，**同机 `ps` 可见**，跑完记得收紧该 token。
- **`--privileged` ＋ `docker.sock`＝容器内进程等价宿主 root**。这是 GPUStack 自己起模型容器的必要条件，无解，只能限定：只在可信机器上跑、别把 8180 发到不可信网络、API key 别写进 URL（会进访问日志）。
- worker 走 host 网 ⇒ 模型实例端口（如 40014）**直接开在宿主上且引擎侧不带鉴权**。"仅内网可达"是硬约束，不是建议。
- 这里**不设 `--shm-size`**：模型实例容器的 `/dev/shm` 由模型环境变量 `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` 决定（见第 7 步），设 worker 的没用。

## 6 注册后端版本（一次即可）

UI 后端页新建：后端 **vLLM**｜版本 `0.1.7-sm75.1-custom`｜自定义 CUDA 可见性｜镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v1`。

注意两处大小写：表单里框架标签显示 `CUDA`，提交值是小写 `cuda`；列表页显示的是落库值。这是两个字段，不是矛盾。

## 7 部署模型

来源 **ModelScope**｜模型 ID `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`｜后端版本 `0.1.7-sm75.1-custom`｜副本 1。

```bash
# 先自检两块粘贴块（仓根执行）
bash gpustack/lint-paste-v1.sh
# 期望末行逐字：LINT_PASTE_OK env=25 args=24
```

- `后端参数` 整块粘 `gpustack/args-paste-v1.txt`（24 行）。
- `环境变量` 整块粘 `gpustack/env-paste-v1.txt`（**25 行**）。**第 25 行 `GPUSTACK_MODEL_VRAM_CLAIM=120779177984` 是硬性项**：没有它，平台按启发式估出 204.5 GiB 需求，而本机八卡可分配只有 117.76 GiB ⇒ 模型永久排队、界面不报错也不给原因。换机器按 `claim ＝ 实测逐卡驻留 MiB × 卡数 × 1048576` 重算。
- 表单里会多出几条不是你粘的 env（`SAFETENSORS_FAST_GPU`、`RUNAI_STREAMER_*`、`VLLM_CACHE_ROOT`）——那是平台自己塞的默认值，别当成粘坏了。
- 要开图输入：另加 `--limit-mm-per-prompt={"image":4}`（**紧贴等号**，JSON 里带空格会被拆成两个 token）＋删掉 `--language-model-only` ＋**重跑显存标定回填 claim**。开多模态后逐卡实测 15,372 MiB，已超 `0.92 × 16384 ＝ 15,073` 的可分配口径 ⇒ 诚实的 claim 过不了准入，本机这一档存在"准入靠低报、运行靠物理余量"的不体面形态，余量只剩约 1,012 MiB/卡。

## 8 验收（四道，缺一不算过）

```bash
docker ps --format '{{.Names}} {{.Status}}' | grep run-            # 1 实例容器在跑
curl -q -sS -H "Authorization: Bearer $GS_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-w4a16-autoround","messages":[{"role":"user","content":"你好"}],"max_tokens":32}' \
  http://${HOST_IP}:8180/v1/chat/completions                       # 2 正文非空
nvidia-smi --query-gpu=memory.used --format=csv,noheader           # 3 逐卡 ≤ 15500 MiB
docker logs "$(docker ps --format '{{.Names}}' | grep run- | head -1)" 2>&1 \
  | grep -ciE "out of memory|CUDA error|Traceback"                 # 4 须为 0
```

OpenAI 兼容口与 UI **同一个口**（`gpustack/config/config.py:922-927` 的 `get_gateway_port` 对非 worker 角色返回 `self.port`），本机就是 8180，**不是 8080**。
八卡在不在位只看 `GET /v2/gpu-devices`（worker 对象上的 `gpu_devices` 是 null，别拿它判断）。
不许靠继续调低 claim 去骗调度器——claim 低于真实占用就是假绿。

## 9 可选：独立的测速与雷达图容器

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
bash docker/benchlocal.sh start
bash docker/benchlocal.sh status     # 逐枚看三枚：sm75-bench-bugfind / -cli / -hermes
bash docker/benchlocal.sh connect    # 面板容器不在 bench 那张网上时用它补
```

**一条不匹配要先解决**：`docker/benchlocal.sh` 默认认 **Ultra 面板镜像**（`vllm-sm75:v…-ultra-beta…`）并有一道 label 门 `io.sm75.benchlocal.unified=1`。本 GPUStack 线的测速台另有 `gpustack/Dockerfile.bench-v1`（产物 tag `vllm-sm75-gpustack-bench:v1`）。走这条线时要么 `IMAGE=… bash docker/benchlocal.sh start` 覆盖并满足 label 门，要么先另建 Ultra 镜像。**该覆盖路径本仓未实跑** `【待实跑复核】`。
