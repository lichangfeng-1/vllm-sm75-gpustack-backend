# run-book v1 —— 从零到模型可推理（Linux 宿主）

适用范围：8×Tesla T10（sm_75，16 GiB/卡）＋ GPUStack v2.2.3 ＋ 我们自己构建的 vLLM 后端镜像。
本文件只给"照做能跑"的最小命令序列；每条参数为什么这么设、换硬件怎么改，见 `gpustack/backend-version-card-v1.md` 与 `gpustack/compat-matrix-v1.md`。

**标注约定**：`【待实跑复核】`＝该命令由源码行号与既有实跑记录推出，尚未在本机复跑；其余步骤为实跑验证过的形制。

---

## 0 前置（先核这三格，缺任何一格都会失败）

| 项 | 要求 | 为什么 |
| --- | --- | --- |
| 主机内存 | **约 251 GiB** | 参数块带 `--engram-config={"cpu_offload":true}`，一部分权重常驻主机内存。64 GB 的机器会在加载阶段被 OOM killer 杀掉，且现象像"镜像坏了" |
| 磁盘 | 数据目录所在盘可用 ≥ 250 G | 权重 17 枚 `*.safetensors` 合计 `181,196,007,016 B`（≈168.8 GiB），另需镜像与缓存余量 |
| 网卡 | 记下宿主内网 IP，下面统一用 `HOST_IP` | worker 与浏览器都要能到达它。`export HOST_IP=192.0.2.10` 换成你的地址 |
| 数据目录 | 下面统一用 `GS_DATA_DIR`（示例值 `/data/gpustack-data`），`export GS_DATA_DIR=/你的路径`；并且要在起容器**之前**先建好、容器要能写 | `--mount type=bind` 的源目录不存在时，失败发生在 `docker create` 而不是 `docker start`，原文 `invalid mount config for type "bind": bind source path does not exist`。另注意 GPUStack 会以 root 写这枚目录，事后普通账号删不掉 |

驱动版本下限**我们没验过**（面板线撞过 NVAPI 与驱动版本不匹配，属另一条形制），此处不写成要求。

## 1 拉仓

```bash
git clone 你的仓地址 vllm-sm75-gpustack-backend
cd vllm-sm75-gpustack-backend
```

**本文件的命令一律在包根目录里执行**（上面那条 `cd` 之后就不要离开）。仓内脚本要么用相对包根的路径，要么用自身位置定位（`SCRIPT_DIR`），所以从哪个绝对位置克隆都不影响；包内不存在指向某台具体机器的路径。属于你这台机器的两样东西统一用变量给：`HOST_IP` 与 `GS_DATA_DIR`。

`gpustack/.gitattributes` 已给本目录下的 `*.txt`／`*.patch` 写了 `text eol=lf`（仓根那份属上游冻结面，不动它），正常 `git clone` 即可。
若你的 git 全局设过 `core.autocrlf=true`，用这条覆盖一次更稳：`git -c core.autocrlf=false clone …`。

## 2 探活并构建后端镜像

镜像分两段构建。**两段都跑完，第 6 节注册时才有的可填**：第一段产出基座层 `vllm-sm75:v0.1.7-beta2`，第二段在它上面追加我们的三层，产出后端镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v1`。

```bash
bash gpustack/preflight-sources-v1.sh          # 只读 HEAD 探测，红就别往下走
bash gpustack/build-gpustack-v1.sh             # 补丁门五道，不执行 docker build
bash gpustack/build-gpustack-v1.sh --build     # 第一段，产出 vllm-sm75:v0.1.7-beta2
docker images vllm-sm75                        # 期望这一枚在（十 GB 量级，首次很慢，建议后台跑并跟看日志）

docker build --file gpustack/Dockerfile.v1 --target timezone-layer \
  --build-arg BASE_TAG=vllm-sm75:v0.1.7-beta2 \
  -t vllm-sm75-gpustack:v0.1.7-beta2-v1 .      # 第二段：层1 pcieipc ＋ 层2 incple ＋ 层3 时区

docker images vllm-sm75-gpustack               # 期望 tag v0.1.7-beta2-v1
```

四条会踩的：

- 第一段的产物名是 `vllm-sm75:v…`，**它不是后端镜像**。拿它去注册，实例起不来且现象像镜像坏了。
- 第二段取 `--target timezone-layer`（`gpustack/Dockerfile.v1:145`，文件末段＝最终层）。取 `pcieipc-layer` 只得到层 1，缺 incple 与 tzdata。
- `BASE_TAG` 必须是**没被本线加装过**的那层。层 1 的装机脚本是追加型的，见到自己的 marker 就退 3（`gpustack/layers/pcieipc/install-pcieipc.sh:35`），拿已含层 1 的 tag 当基座会把构建直接打红。
- 基座那枚 tag 与 digest 钉在 `docker/Dockerfile:3` 与 `docker/build.sh:21`，现值 `vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb…`。换基座用第一段的 `BASE_IMAGE=` 环境变量覆盖（脚本原样转成 `--build-arg`，不必改仓内文件）。**要注意换的到底是什么**：上游那几行是按 `--index …/cu129` 重装 `torch==2.13.0+cu129` 并断言 `torch.version.cuda == "12.9"` 的（`docker/Dockerfile:15-17`），所以只换基座时镜像里的 torch 与 triton 不变，变的是它下面那层 CUDA 工具链；要让 torch 也跟着换档，得连这几行一起动，而 `docker/Dockerfile` 在上游冻结清单里。

两枚构建期下载件（`flashinfer_cubin` wheel 与 `nvidia-pstated` 二进制）都钉在上游 Release ＋ sha256，我们不托管、不重新分发（后者是 GPL-3.0-only）。

## 3 起 GPUStack server

```bash
docker create --name sm75-gpustack-server --hostname sm75-gpustack-server \
  --network bridge --restart no --env TZ=Asia/Shanghai \
  -p ${HOST_IP}:8180:80 \
  --mount type=bind,source=${GS_DATA_DIR},target=/var/lib/gpustack \
  quay.io/gpustack/gpustack:v2.2.3
docker start sm75-gpustack-server
```

**两个坑，都是实跑撞过的**：

先说一句指路约定：本节与后文凡指向 Python／mjs 源码并带行号的——无论写成 `gpustack/xxx.py:行号` 还是只写 `xxx.py:行号`——指的都是**上游 GPUStack v2.2.3 的源码路径与行号**，这个仓里没有这些文件。我们自己写的件集中在 `gpustack/` 目录与仓根文档：多数带版本后缀（如 `run-book-v1.md`），也有不带的（如 `layers/incple/patch_ple_inc.py`、`layers/pcieipc/install-pcieipc.sh`），所以按路径区分，别按后缀猜。行号按上游 `v2.2.3` 正式版源码计，换版本请先复核。

- **不要加 `--system-default-container-registry`**。加了它，后端注册表里不含 registry 段的镜像名会被拼成 `quay.io/vllm-sm75-gpustack:…` 去远端拉，报 `Failed to pull image`，而本地明明有这枚镜像；表现是"参数块一行都没被执行"。拼接规则 `gpustack/utils/config.py:87-108`。已经设过就清掉：`PUT /v2/config {"system_default_container_registry":""}`（该字段在运行时白名单 `gpustack/utils/config.py:9-12`，可热改；这条 PUT 要带鉴权，无 token 直接 401）。

  这条警告已于 2026-10-15 在同一台机上跑出**双向证据**：带该 flag 起 server 时，worker 日志逐字 `image resolved to: quay.io/vllm-sm75-gpustack:v0.1.7-beta2-v1`，接 `manifest unknown`，实例转 `ERROR`；换成不带该 flag 重建 server 后，同一个镜像名由本地直接接住，实例转 `RUNNING`。

  还有一条容易读错，别拿它当"没事"的依据：**数据库里那一列是空的，CLI flag 照样生效**。`clusters.system_default_container_registry` 实测为 `NULL`，而 worker 仍去拉 `quay.io/…`；落点在 `gpustack/routes/clusters.py:794-795`（`cluster.system_default_container_registry or cfg.system_default_container_registry`）。所以判这格只能读运行时行为，读库会漏。
- **不要用 `--network host`**。GPUStack 内嵌的 Higress 网关会自己开一串监听口，`--host-ip` 管不到它们，暴露面因此比设计大。这里用 bridge ＋ 显式 `-p 宿主IP:8180:80`，只放这一个口。

## 4 取初始口令并登录

```bash
docker exec sm75-gpustack-server cat /var/lib/gpustack/initial_admin_password
```

浏览器开 `http://${HOST_IP}:8180` 登录并改口令。**口令别贴进截图、群或任何公开渠道。**

这格只适用于**全新首启**：`initial_admin_password` 是一次性的，改过口令之后该文件就从数据目录里消失（10-15 实测：老部署的目录里已无此件，对照一枚全新空目录起时它在、长度 12）。老部署取不到它，只能走界面重置。另外管理登录口是 `POST /auth/login`，**不带 `/v2` 前缀**（`/v2/auth/login` 实测 404）。

## 5 起 worker

同机部署**不需要 token**：worker 取 token 的路径是 `gpustack/config/config.py:312-313` 的 `read_registration_token(self.data_dir)`，读的是**它自己 data_dir 里**那一份。不传 `--data-dir` 时 data_dir 是 OS 默认值，也就是容器内的 `/var/lib/gpustack`（`config.py:82`）⇒ **worker 必须把 `${GS_DATA_DIR}` 挂到 `/var/lib/gpustack`**（与 server 同一个目标路径），两边看到的才是同一份目录。本文件此前写作 target=${GS_DATA_DIR}，那样 worker 在自己那格空目录里找不到 token，只能退回 --token。现写法是 2026-10-18 在同一台机上实跑复验过的形制：该 worker 建的模型实例 `resolved_path=/var/lib/gpustack/cache/model_scope/…`，169 GiB 权重正常加载。

```bash
docker create --name sm75-gpustack-worker --hostname sm75-gpustack-worker \
  --restart no --env TZ=Asia/Shanghai \
  --privileged --network host --runtime nvidia \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  --mount type=bind,source=${GS_DATA_DIR},target=/var/lib/gpustack \
  quay.io/gpustack/gpustack:v2.2.3 \
  --server-url http://${HOST_IP}:8180 \
  --advertise-address ${HOST_IP} \
  --worker-name sm75-gpustack-worker
docker start sm75-gpustack-worker
```

- **`--advertise-address` 不能省**。多网卡机器上 worker 会通告错地址，表现是"worker 显示 ready，但模型起不来或连不上"，很难查。
- 跨机部署才需要 token：用 UI 的 Workers → 添加 worker 给出的整条命令，`--token` 会出现在命令行上，**同机 `ps` 可见**，跑完记得收紧该 token。
- **`--privileged` ＋ `docker.sock`＝容器内进程等价宿主 root**。这是 GPUStack 自己起模型容器的必要条件，无解，只能限定：只在可信机器上跑、别把 8180 发到不可信网络、API key 别写进 URL（会进访问日志）。
- worker 走 host 网，所以模型实例端口（如 40014）**直接开在宿主上且引擎侧不带鉴权**。"仅内网可达"是硬约束，不是建议。
- 这里**不设 `--shm-size`**：模型实例容器的 `/dev/shm` 由模型环境变量 `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` 决定（见第 7 步），设 worker 的没用。

### 5a 跨机部署时，worker 的启动参数可以问 server 要（不用点界面）

```bash
curl -s -H "X-API-Key: ${GS_KEY}" http://${HOST_IP}:8180/v2/clusters/1/registration-token
```

返回五格 `token / server_url / image / env / args`。本机 2026-10-18 现读：`server_url` ＝ `http://172.17.100.10:8180`，`image` ＝ `quay.io/gpustack/gpustack:v2.2.3`，`env` 里一条 `GPUSTACK_TOKEN`（58 字符），`args` 是空数组。这一格的结构体注释逐字写着它是"The arguments of docker run command to register a worker"（`gpustack/schemas/clusters.py:696-706`），所以 `-e GPUSTACK_TOKEN=…` 与命令行 `--token …` 两种写法都算照用途用。

两条注意：

- **这条 token 是写凭据**，拿到它的人能往这个集群里注册 worker（`gpustack/routes/clusters.py:733-734` 的注释就是这么定性的，接口因此按"可写"权限把关）。它会出现在 `ps` 和 shell 历史里，跨机跑完就轮换。
- **别用 `GET /v2/clusters/{id}/manifests` 拿启动命令**。那条只给 Kubernetes 集群渲染 DaemonSet，非 K8s 集群直接报错（判据 `clusters.py:761-764`）。本机 `provider=Docker`，实测退 **422**，原文 `Cannot get manifests for cluster sm75-stage2(id: 1) with provider ClusterProvider.Docker`。

## 6 注册后端版本（一次即可）

UI 后端页新建：后端 **vLLM**｜版本 `0.1.7-sm75.1-custom`｜自定义 CUDA 可见性｜镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v1`。

注意两处大小写：表单里框架标签显示 `CUDA`，提交值是小写 `cuda`；列表页显示的是落库值。这是两个字段，不是矛盾。

### 6a 同一件事的 API 形态

```bash
curl -s -H "X-API-Key: ${GS_KEY}" http://${HOST_IP}:8180/v2/inference-backends
```

`version_configs` 是**以版本名为键的字典**。本机现读 vLLM 那一行（`id=1`、`backend_source=built_in`、`built_in_version_configs` 30 格、`version_configs` 只有我们这一格）：

```json
{"0.1.7-sm75.1-custom": {"image_name": "vllm-sm75-gpustack:v0.1.7-beta2-v1", "run_command": null, "entrypoint": null, "built_in_frameworks": null, "custom_framework": "cuda", "env": null}}
```

写入用 `PUT /v2/inference-backends/{id}`（路由 `routes/inference_backend.py:1138`）。四条规矩，踩了就是 400：

- **版本名必须以 `-custom` 结尾**。凡是"用户自定义版本"（`built_in_frameworks` 为空、`custom_framework` 有值）都强制校验这个后缀（`inference_backend.py:1583-1608`）——上面那个版本号里的 `-custom` 不是装饰，删掉就被拒。
- **内置后端不许设 `default_version`**（`inference_backend.py:1172-1175`）。vLLM 属内置后端，它的默认版本由平台自己管。
- `backend_name` 不可改（`inference_backend.py:1166-1169`）。
- **PUT 按整行提交**：`version_configs` 里少写一格，等于请求删除那一格，会先过一道删除校验（`inference_backend.py:1178` 的 `_validate_version_removal`）。所以正确做法是**先 GET 拿整行、只在那本字典上加改、再整行 PUT**，别手写一份最小 body。

## 7 部署模型

来源 **ModelScope**｜模型 ID `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`｜后端版本 `0.1.7-sm75.1-custom`｜副本 1。

```bash
# 先自检两块粘贴块（仓根执行）
bash gpustack/lint-paste-v1.sh
# 期望末行逐字：LINT_PASTE_OK env=25 args=24
```

- `后端参数` 整块粘 `gpustack/args-paste-v1.txt`（24 行）。
- `环境变量` 整块粘 `gpustack/env-paste-v1.txt`（**25 行**）。**第 25 行 `GPUSTACK_MODEL_VRAM_CLAIM=120779177984` 是硬性项**：没有它，平台按启发式估出 204.5 GiB 需求，而本机八卡可分配只有 117.76 GiB，所以模型永久排队、界面不报错也不给原因。换机器按 `claim ＝ 实测逐卡驻留 MiB × 卡数 × 1048576` 重算。
- 表单里会多出几条不是你粘的 env（`SAFETENSORS_FAST_GPU`、`RUNAI_STREAMER_*`、`VLLM_CACHE_ROOT`）——那是平台自己塞的默认值，别当成粘坏了。
- 要开图输入：另加 `--limit-mm-per-prompt={"image":4}`（**紧贴等号**，JSON 里带空格会被拆成两个 token）＋删掉 `--language-model-only` ＋**重跑显存标定回填 claim**。开多模态后逐卡实测 15,372 MiB，已超 `0.92 × 16384 ＝ 15,073` 的可分配口径，所以诚实的 claim 过不了准入，本机这一档存在"准入靠低报、运行靠物理余量"的不体面形态，余量只剩约 1,012 MiB/卡。
- 上面那句里只有 `--limit-mm-per-prompt` 是"要多张图"才需要的。真正卡住图输入的是 `--language-model-only`：删掉它之后不必显式设限额也能进图——这枚档在平台侧决议成 `Qwen4ExpForConditionalGeneration`（带视觉塔），默认限额 1 张。10-15 实测一张工作台截图进模型、正文准确描述了画面，硬指标 `usage.multimodal_tokens.image = 2040`、prompt 2117 token、`mm_cache_queries_total = 1.0`，引擎自报 `Multi-modal warmup completed`。**该次取证走的是引擎直连口（本机 40028），不是 8180 网关**，网关那条腿当时因无可用 API key 仍是 401，别把这条读成"网关侧图像已验"。

## 8 验收（四道，缺一不算过）

```bash
docker ps --format '{{.Names}} {{.Status}}' | grep run-            # 1 实例容器在跑
curl -q -sS -H "Authorization: Bearer $GS_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-w4a16-autoround","messages":[{"role":"user","content":"你好"}],"max_tokens":256}' \
  http://${HOST_IP}:8180/v1/chat/completions                       # 2 正文非空
nvidia-smi --query-gpu=memory.used --format=csv,noheader           # 3 逐卡 ≤ 15500 MiB
INST="$(docker ps --format '{{.Names}}' | grep run- | head -1)"
docker logs "$INST" 2>&1 | grep -ciE "out of memory|CUDA error|Traceback"   # 4 须为 0
```

OpenAI 兼容口与 UI **同一个口**（`gpustack/config/config.py:922-927` 的 `get_gateway_port` 对非 worker 角色返回 `self.port`），本机就是 8180，**不是 8080**。
八卡在不在位只看 `GET /v2/gpu-devices`（worker 对象上的 `gpu_devices` 是 null，别拿它判断）。
不许靠继续调低 claim 去骗调度器——claim 低于真实占用就是假绿。

道 2 有两个坑，都是 10-15 撞过的：

1. **`$GS_KEY` 必须在建的时候就要到手。** 平台侧的键是"建时给一次"的形态：`api_keys` 表里 `access_key` 只有 16 个字符（是标识位），secret 只存散列，事后从库里捞不出来。凡是以后还要走网关做验收的部署，**建 key 当场就落盘**，否则这一道整条腿死掉，而它红起来只有一句 401 `Invalid authentication credentials`。
2. **`max_tokens` 要给够。** 这档带 `--reasoning-parser=qwen3`，思考链与正文分家；预算开小了会被思链吃满，出现 HTTP 200、`finish_reason=length`、`completion_tokens` 恰等于上限而正文 0 字（10-15 实测那发 reasoning_tokens 48、正文长度 0）。只看 HTTP 状态码判不了这一道。

## 9 可选：独立的测速与雷达图容器

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
bash docker/benchlocal.sh start
bash docker/benchlocal.sh status     # 逐枚看三枚：sm75-bench-bugfind / -cli / -hermes
bash docker/benchlocal.sh connect    # 面板容器不在 bench 那张网上时用它补
```

**一条不匹配要先解决**：`docker/benchlocal.sh` 默认认 **Ultra 面板镜像**（`vllm-sm75:v…-ultra-beta…`）并有一道 label 门 `io.sm75.benchlocal.unified=1`。本 GPUStack 线的测速台另有 `gpustack/Dockerfile.bench-v1`（产物 tag `vllm-sm75-gpustack-bench:v1`）。走这条线时要么 `IMAGE=… bash docker/benchlocal.sh start` 覆盖并满足 label 门，要么先另建 Ultra 镜像。

这条覆盖路径**已实跑通**（2026-10-18，本机 8×T10）：

- label 门满足：本机那枚 bench 镜像（ID `992df1afbdc2`，现读 tag `vllm-sm75-gpustack-bench:v0.1.7-beta2-v1`）的 `io.sm75.benchlocal.unified` 值为 `1`。
- 执行形态：`ULTRA_DATA_ROOT=<你的绝对目录> BENCHLOCAL_ENV_FILE=$ULTRA_DATA_ROOT/benchlocal.env IMAGE=vllm-sm75-gpustack-bench:v0.1.7-beta2-v1 bash docker/benchlocal.sh start`，三枚容器（bugfind／cli／hermes）起来后 `status` 逐枚 running，容器内 `GET 127.0.0.1:4200/health` 逐枚回 `{"ok":true,"busy":false,"managed":true}`。
- **一个会绊人的点**：`IMAGE` 不给时脚本按 `vllm-sm75:v${VERSION/-beta/-ultra-beta}` 拼默认值（本仓 VERSION 下得到 `vllm-sm75:v0.1.7-ultra-beta2`），而实际发版的面板镜像 tag 带后缀，这枚裸 tag 在盘上通常**不存在**，脚本会在 label 门上退 2 并报 `Build the beta2 Ultra image first`。所以走这条路必须显式给 `IMAGE=`。
- 三枚容器的名字是固定的：若盘上已有同名的别线容器，脚本会先 `stop` 再 `rm` 重建（`docker/benchlocal.sh:46-47`）。跑前先 `docker inspect` 留证、跑完按原镜像重建并 `stop` 回原状态，别线在用时先跟使用者对齐。

### 9a 这三枚不是测速台，而且不接受你手填的 base_url

2026-10-18 读镜像内实物（`vllm-sm75-gpustack-bench:v0.1.7-beta2-v1`）确认，三条都有行号：

- 它们是**评测校验器**，只放行五个口：`/run-scenario`、`/verify-answer`、`/verify-canonical`、`/verify-oneshot`、`/verify-multiround`（`/opt/sm75-workbench/benchlocal-verifiers/gateway.mjs:12`），每个口都要带 `Authorization: Bearer $SM75_BENCH_VERIFIER_KEY`（同文件 `:6-7`，键短于 32 字符直接拒绝启动）。
- **请求体里一旦带 `model`，它的 `inferenceBaseUrl` 必须逐字等于 `<SM75_BENCH_PROXY_ORIGIN>/benchlocal-inference/v1`**，否则 400 `Invalid managed model endpoint`（`:78-81`）。那个 origin 默认是 `http://sm75-bench-controller:1615`，也就是**控制端（面板／桌面客户端）自己开的那一跳代理**。⇒ "把 base_url 直接填成 GPUStack 网关 `http://<宿主>:8180/v1`" 这条路走不通；要走，得让控制端在 1615 上做反代。
- **速度测试与雷达图不在这三枚里**，在镜像内的面板工作集：无头入口 `/opt/sm75-workbench/console/benchmark-runner.py`，雷达图前端是 `public/benchlocal-native-ui.js`（108–136 行那段 Radar）与 `public/index.html:751`。

### 9b 单独手动起一枚（与脚本等价）

```bash
NETWORK=sm75-benchlocal-native
KEY_FILE=/srv/vllm-sm75/ultra/benchlocal.env   # 首启由脚本生成的那份，内容一行 SM75_BENCH_VERIFIER_KEY=<64 位十六进制>
docker network inspect "$NETWORK" >/dev/null 2>&1 || \
  docker network create --internal --driver bridge --label io.sm75.benchlocal.role=network "$NETWORK"
docker run -d --name sm75-bench-cli \
  --network "$NETWORK" --runtime runc --restart unless-stopped \
  --init --read-only --pids-limit 512 --memory 4g --cpus 2 \
  --cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g,mode=1777 \
  --tmpfs /home/verifier:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=700 \
  --tmpfs /workspace:rw,exec,nosuid,nodev,size=512m,uid=10001,gid=10001,mode=700 \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
  --env SM75_BENCHLOCAL_ROLE=cli-40 --env-file "$KEY_FILE" \
  --env SM75_BENCH_PROXY_ORIGIN=http://sm75-bench-controller:1615 \
  --label io.sm75.benchlocal.role=verifier --label io.sm75.benchlocal.unified=1 \
  --entrypoint python3 vllm-sm75-gpustack-bench:v0.1.7-beta2-v1 \
  /opt/vllm-sm75/benchlocal-entrypoint.py
```

逐条来自 `docker/benchlocal.sh:50-62`，两处有意不抄：脚本还打了两枚只给 Unraid 界面看的 label（`:60`），手动版用不上；`--env-file` 那本键文件要满足 `^SM75_BENCH_VERIFIER_KEY=[a-fA-F0-9]{64}$`（`:34`），没有就先 `bash docker/benchlocal.sh start` 让脚本生成一份再复用。

三枚之间只差两处——`--name` 与 `SM75_BENCHLOCAL_ROLE`（角色对应关系写在 `benchlocal.sh:37`：`bugfind:bugfind-15 / cli:cli-40 / hermes:hermesagent-20`）。角色决定内层端口：`bugfind-15`→4010、`cli-40`→4040、`hermesagent-20`→4010，外层统一 4200（`docker/helpers/benchlocal-entrypoint.py:5`、`gateway.mjs:11`）。

三枚的前置门，都是"不起就直接退 2"那种：镜像内要有 `/opt/sm75-workbench/benchlocal-verifiers/runtime-lock.json`（`benchlocal-entrypoint.py:10`）；容器内一看见 `/dev/nvidia0` 就拒绝启动（`:11`）——**这三枚是 CPU 校验器，不许挂卡**；那张网是 `--internal`，容器没有对外网络，所以只有挂在同一张网上的控制端能驱动它们（`benchlocal.sh:35`，补接线用 `benchlocal.sh connect`）。

## 10 性能预期（以及仓根那三张表的适用面，务必先读）

仓根 `README.md` 的性能节挂着三张测速表：基线档 **43.31**、FP8PLE **57.73**、W4A16-AutoRound **57.04** tokens/s（平均总输出）。

**这三张不是本 run-book 这条路的读数。** 它们的备注逐字是 `G292Z20 8*T10 SM-75 …`，**没有出现 GPUStack 字样**，
是在镜像自带的 Ultra／面板形态上、用"大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）"跑出来的。
**照本文件起完 GPUStack 之后，不要拿 57.73 当验收线**，否则你会以为自己配错了。

要在 GPUStack 形态上量同一把尺的数，用**镜像自带的测速核心**，不是第 9a 节那三枚校验容器（它们的入口只认控制端代理那一跳）。

第一步，把尺子装到一个宿主目录里（核心是第三方 `gengchaogit/llm_speedtest` 的一个固定 commit，脚本按 commit 取字节并把 sha256 写进 `manifest.json`，我们不另存副本）：

```bash
docker run --rm -v "$PWD/bench-seed:/seed" --entrypoint python3 \
  vllm-sm75-gpustack-bench:v0.1.7-beta2-v1 \
  /opt/sm75-workbench/console/install-benchmark.py /seed
```

钉的是 commit `eb19940368f0d811ef7b14f77f61d1c7e8676c7a`、产物 sha256 前缀 `18b7b1d39900a8cb`，**与仓根三张表同一份字节**（`install-benchmark.py` 全文只做三件事：取 `llm_test_backend.py`、装 `slowapi==0.1.9` 到 `deps/`、写 `manifest.json`）。这一步要能访问 `raw.githubusercontent.com`；本机走的是仓内配的国内镜像优先路由，官方直连只兜底。

第二步，无头复跑（每完成一格增量落 `<cfg>.results.json`，文件 mtime 就是取证时刻）：

```bash
docker run --rm --network host \
  -v "$PWD/bench-seed:/speedtest:ro" -v "$PWD/cfg.json:/work/cfg.json:ro" \
  -v "$KEY_FILE:/run/secrets/gs_apikey:ro" -v "$PWD/out:/work/out" \
  --entrypoint python3 vllm-sm75-gpustack-bench:v0.1.7-beta2-v1 \
  /opt/sm75-workbench/console/benchmark-runner.py /speedtest /work/cfg.json
```

`cfg.json` 的字段与本机结案那批用的值（`url` 换成你的网关口，`key_file` 指到第 4 步落盘的那份键文件，**键值不进命令行也不进日志**）：

```json
{
  "tag": "gpustack-form-v1",
  "url": "http://172.17.100.10:8180/v1/chat/completions",
  "model": "qwen3.8-flash-next-w4a16-autoround",
  "key_file": "/run/secrets/gs_apikey",
  "lengths": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
  "output": 1024,
  "concurrency": 1,
  "timeout": 30000,
  "temperature": 1.0,
  "top_p": 0.95,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "out_prefix": "/work/gpustack-form-v1"
}
```

`lengths` 就是 README 那张表的九档（步长 128、倍数 2 展开即 512→131072），`output` **必须是 1024**——这把尺子的历史数字全在这一格上，填 16 就不是同一把尺了。`out_prefix` 决定结果落盘位置：加 `.json` 后即为产物文件，挂出来的宿主目录里就是取证件。

**一处未逐字核对**：本节下面那组读数用的驱动另外复刻了测速台自己 `/ws/test` 的调用序（预热、两次网络时延采样、每档估提示词长度并算**动态超时**）。`benchmark-runner.py` 是否也走动态超时这一步没核对过；131072 那档最容易受它影响，复跑对不上数时先查这一处。

另一条更省事的路：只认第 8 步那四道防假绿断言（实例 `running` ＋ 一次真实推理非空 ＋ 逐卡 ≤15500 MiB ＋ 零 OOM），速度另开窗口测。

**跨形态比较的三条硬规矩**（都是踩过才写下的）：
- 只跟**同一把尺的同一字段**比——面板/GPUStack 各自那个"平均总输出"字段，不要拿手算的九格算术均值去对它的均值。
- 长度集合必须**逐格对齐**（九档就是九档，少一格直减不成立），且单格差落在 **±4.6%** 以内不得当结论。
- **首格（512）常是冷启动离群点**，跨批比要么固定剔第一格，要么先补一档不计数的预热。

### 10a GPUStack 形态的九档读数（2026-10-18 20:10 北京结案）

被测对象＝GPUStack 平台拉起的实例（镜像 `vllm-sm75-gpustack:v0.1.7-beta2-v1`），经 8180 网关；尺子＝上一节那份 pinned 核心，与仓根三张表同一份字节；参数逐格同上；九档 512→131072；两批配对。

| 口径 | 平均总输出 tokens/s | 平均总预填充 tokens/s |
|---|---|---|
| 面板字段（含首格，末格双计那式） | **52.81** | **2488.03** |
| 九格算术均值 | 52.85 | — |
| 九格中位 | 52.73 | — |
| 剔首格算术 | 52.89 | 2738.77 |

逐格输出（512→131072）：52.53／52.11／52.64／53.28／53.55／52.95／52.73／53.41／52.46。

三条结论，**推导链在报告里，别只引这三句**：

1. 比同权重的面板参照 57.04 低 **7.4%**，逐格 −4.26%～−9.22%、八格同向、多格超 ±4.6% 噪声门 ⇒ 是真差，不是批次噪声。
2. **这一刀不在网关**。同尺改指引擎直连口 40015（同实例、同档，另两批）得 52.57，与走网关的 52.81 差 **+0.46%**（噪声带内），逐格 −2.71%～+3.61% 全在门内 ⇒ 差在被平台拉起的那个**引擎档**（镜像层／参数档），不在 Higress 路由与鉴权那一跳。**归因一项没做**，报告第八节列了五条候选（含 10-18 并档那笔、args 24 条与 0a7 档的差集）。
3. **512 首格罚是本形态多出来的一笔**。四批全离群（预填充 586～700 对次格 2391～2657；首 token 延迟 837～889 ms 对 413～444 ms），但与省电档无关（起跑 8～10 秒八卡已进 P0/1590，每 10 秒采样器留痕）、与网关无关（直连路同样离群），而面板参照的 512 格不离群 ⇒ **机制未查**。跨形态比数一律**同时报含首格与剔首格两套**。

**口径边界**：这组数属 GPUStack 形态，**不替换也不回填**仓根那三张面板形态表（43.31／57.73／57.04）；两形态相减得到的 7.4% 不得记到任何单一机制名下。10-14 那枚 55.34 的读数仍然作废在案（疑因当时开着视频输入，未归因），它不是本节的对照——本节四批都是纯文本档。

## 11 同一条构建线上的三枚产物（面板形态怎么起）

先把一件事说清楚：**第 2 步建出来的后端镜像里起不了面板**。2026-10-18 逐字节查过成品 `vllm-sm75-gpustack:v0.1.7-beta2-v1`——里面有 `/opt/vllm-sm75/`（`console` 目录只剩 `lmcache_probe.py` 一支探针脚本）、有 `runtime-entrypoint.py`、有 `profiles/*.json`，**但没有 node 可执行文件、也没有面板服务端 `server.mjs` 那套负载**。面板要跑的是 `runtime-entrypoint.py:69` 那句 `node /opt/sm75-workbench/console/server.mjs`，这两个前提在这枚镜像里都不成立。

这不是漏装，是形态分工。同一份基座（`docker/Dockerfile` 的第一段产物）在仓里派生三枚，各管一件事：

| 产物 | 建法 | 用途 |
|---|---|---|
| `vllm-sm75-gpustack:v0.1.7-beta2-v1` | 第 2 步两条命令 | GPUStack 拉起来跑模型的**后端镜像**，面板负载不进它 |
| `vllm-sm75-gpustack-bench:v0.1.7-beta2-v1` | `gpustack/Dockerfile.bench-v1`（`ARG STAGE1_TAG` 指上面那枚） | 第 9 节三枚校验容器 ＋ 第 10 节那把测速尺子的载体，**带 node**（v22.23.2） |
| `vllm-sm75-console:v0.1.7-beta2-20261013-sharedbase-v1` | 仓内 `docker/Dockerfile.ultra-full` 那条线 | **面板形态**，node ＋ `/opt/sm75-workbench/console` 全在 |

三枚的 `ai.vllm.*` 标签逐字相同（`build.commit=ced6857a…`、`image.tag=vllm/vllm-openai:v0.30.0-cu129`、`sm75.release=vllm-sm75-v0.1.7-beta2`），所以"我们改的东西"只在共用第一段里存在一次，后面三枚都继承它——**该进镜像的补丁确实全进镜像了**，宿主侧没有 hack 脚本。

面板那条命令，按现役那台机上跑过的容器参数现读还原（`--shm-size` 是 64 GiB 不是默认 10、两枚端口 1615 面板／18001 引擎、`--restart no`）：

```bash
docker run -d --name sm75-panel --restart no --runtime nvidia \
  --shm-size 64g -p 1615:1615 -p 18001:8000 \
  -v "$PANEL_DATA:/console-data" \
  --entrypoint python3 vllm-sm75-console:v0.1.7-beta2-20261013-sharedbase-v1 \
  /opt/vllm-sm75/runtime-entrypoint.py
```

**这一条标 【未实跑复验】**：本机实跑过的是它的上一代 tag（`vllm-sm75:v0.1.7-ultra-beta2-consolefix-log-to20s-pcieipc-incple-benchseed`），那枚容器还挂了 `/dsh`、`/models`、四个缓存目录和一枚 `libnvidia-api.so.1` 私有副本；上面这条只保留了"面板起得来"必需的那几格，其余按你自己的机器加。**撞了错就按报错回来补这一节，别当成教程已验。**

"预置复刻我们测好的配置"这件事在面板形态里的载体是 `/opt/vllm-sm75/profiles/*.json`（镜像内已有 `flash-next-tp8-256k-mtp3.json`、`flash-next-tp8-256k-nomtp.json` 等）。**但有一条实测过的风险要先知道**：面板会**静默重写**档位 argv——本机实测过 53 项被改写成 37 项、丢掉 `--engram-config` 等 10 个 SM75 定制项，非人为触发。所以用面板跑之前，把档位 JSON 与 `gpustack/args-paste-v1.txt` 逐字 diff 一次，改完再看一眼落盘的那份，别看界面显示。

GPUStack 这条线**不吃这个风险**：它的参数来自第 7 步那两块粘贴块，平台原样传给 `vllm` 命令行，没有中间重写者。这也是这一节放在第 11 节而不是放进主流程的原因。
