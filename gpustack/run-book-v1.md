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

同机部署**不需要 token**：worker 取 token 的路径是 `gpustack/config/config.py:312-313` 的 `read_registration_token(self.data_dir)`，读的是共享 data 目录里的文件，而本形制下 server 与 worker 挂的是同一份 `${GS_DATA_DIR}`。

```bash
docker create --name sm75-gpustack-worker --hostname sm75-gpustack-worker \
  --restart no --env TZ=Asia/Shanghai \
  --privileged --network host --runtime nvidia \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  --mount type=bind,source=${GS_DATA_DIR},target=${GS_DATA_DIR} \
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

这条覆盖路径的两个前置条件里，**label 门这一条已实测满足**：10-15 现读本机那枚 bench 镜像（ID `992df1afbdc2`）的 `io.sm75.benchlocal.unified` 值为 `1`。剩下的那一半——`benchlocal.sh` 接受 `IMAGE=` 覆盖之后整条链路真跑得通——**我们还没实跑** `【待实跑复核】`。

## 10 性能预期（以及仓根那三张表的适用面，务必先读）

仓根 `README.md` 的性能节挂着三张测速表：基线档 **43.31**、FP8PLE **57.73**、W4A16-AutoRound **57.04** tokens/s（平均总输出）。

**这三张不是本 run-book 这条路的读数。** 它们的备注逐字是 `G292Z20 8*T10 SM-75 …`，**没有出现 GPUStack 字样**，
是在镜像自带的 Ultra／面板形态上、用"大模型 Prefill & Decode 性能测试工具 v3（chao 魔改版）"跑出来的。
**照本文件起完 GPUStack 之后，不要拿 57.73 当验收线**，否则你会以为自己配错了。

要在 GPUStack 形态上量同一把尺的数，两条路：

1. 用第 9 步那三枚 bench 容器，把它们的 base_url 指向 GPUStack 网关（`http://` ＋ `HOST_IP` ＋ `:8180`，带 API key），
   参数照 README 那张配置表逐格填：并发 1／请求超时 30000 ms／期望输出长度 **1024**／
   最小 512／最大 131072／步长 128／步长倍数 2／Temperature 1／Top P 0.95／Presence 0／Frequency 0。
   **注意第 9 步那条 `IMAGE=` 覆盖路径我们还没实跑**，走不通就先用面板形态量。
2. 或者只认第 8 步那四道防假绿断言（实例 `running` ＋ 一次真实推理非空 ＋ 逐卡 ≤15500 MiB ＋ 零 OOM），
   速度另开窗口测。

**跨形态比较的三条硬规矩**（都是踩过才写下的）：
- 只跟**同一把尺的同一字段**比——面板/GPUStack 各自那个"平均总输出"字段，不要拿手算的九格算术均值去对它的均值。
- 长度集合必须**逐格对齐**（九档就是九档，少一格直减不成立），且单格差落在 **±4.6%** 以内不得当结论。
- **首格（512）常是冷启动离群点**，跨批比要么固定剔第一格，要么先补一档不计数的预热。

本线目前有一枚 GPUStack 形态的读数，**未进对外表**：维护者判断疑因当时开着视频输入所致，**未归因**，
所以这里不给数字，也不许拿它去算"GPUStack 比面板慢百分之几"。
