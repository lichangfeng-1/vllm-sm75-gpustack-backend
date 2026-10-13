# 退出码归属表 · GPUStack 交付线（任务 9 的映射表）

一句话：本表登记本线每一枚退出码归哪个件、判的是什么。仓内凡写"其余码位归属以本表为权威"的地方，权威就是这份文件。

## 〇、本表是什么、不是什么（先读这一节）

- 本表是**人读的权威登记，不是任何脚本的输入**。全仓没有脚本读取本表——`gate-tree-unchanged-v1.sh` 的判据输入是 `gpustack/SHA256SUMS-upstream.txt`（该件 `:32`），它自己的码位用法逐条写在它的头注 `:10-17`。所以删掉本表不会让任何一道门变红；把这条事实写在这里，是为了不让下一位把"本表是权威"误读成"有脚本会校验本表"。
- 本表只收**仓内可核对的声明处**，每行给 `文件:行号`。行号是本仓当前提交态的读数；改动那些件的头部注释后须同步本表。
- 数字相同但承载层不同，不算冲突，见第一节。

## 一、三个承载层

| 层 | 谁在执行 | 非零码怎么被读到 |
|---|---|---|
| 甲＝宿主脚本面 | 在检出机上直接 `bash`/`python` 跑的守卫与生成器 | 调用者看脚本自己的 rc |
| 乙＝镜像内 RUN 面 | `docker build` 期间的 RUN 门禁与装机脚本 | docker 只报"构建失败"，件自己的码写在构建日志文案里 |
| 丙＝收敛面 | `build-gpustack-v1.sh` 作总入口 | 上游漂移收 9、补丁不匹配收 8、docker 非零收 20 |

## 二、甲层（宿主脚本面）

| 码 | 归属件 | 判的是什么 | 声明处 |
|---|---|---|---|
| 0 | 全部甲层件 | 通过 | 各件头注首行 |
| 2 | 全部甲层件 | 用法/前置类红的统一码位：参数形态非法、缺必要工具、依赖件缺失或不可读、落点守卫判死、行尾含 CR、末行缺换行 | `gate-tree-unchanged-v1.sh:12`、`build-gpustack-v1.sh:25`、`lint-paste-v1.sh:32`、`bridge-sha-v1.sh:16`、`preflight-sources-v1.sh:15`、`verify-gpustack-form-v1.sh:77`、`gen-args-paste-v1.py:41` |
| 3 | `preflight-sources-v1.sh` | 至少一个分组"全组红"（逐组取或，组内有一条绿就不红） | `preflight-sources-v1.sh:18` |
| 4 | `gate-tree-unchanged-v1.sh` | 基线表格式非法（非法行／重复路径／末行缺换行导致行数与条目不符） | `gate-tree-unchanged-v1.sh:13` |
| 5 | `gate-tree-unchanged-v1.sh` | 基线表已存在且未加 `--force`（防静默重定基线） | `gate-tree-unchanged-v1.sh:14` |
| 6 | `gate-tree-unchanged-v1.sh` | 作用域为空／文件不可读／必要工具缺失 | `gate-tree-unchanged-v1.sh:15` |
| 7 | `gate-tree-unchanged-v1.sh` | 上游树漂移——该护栏唯一的红码（篡改／缺失／上游作用域内新增，含行尾变化） | `gate-tree-unchanged-v1.sh:11` |
| 8 | `build-gpustack-v1.sh` | 补丁不匹配（dry-run 返回 1＝hunk 不匹配，或特征计数不符） | `build-gpustack-v1.sh:27` |
| 9 | `build-gpustack-v1.sh` | 上游文件被改（把护栏自己的 7 映射过来） | `build-gpustack-v1.sh:28` |
| 10 | `gate-tree-unchanged-v1.sh` | 基线表不存在 | `gate-tree-unchanged-v1.sh:16` |
| 13 | `bridge-sha-v1.sh` | 指纹桥差集非空：`DIFF`＝两边字节不同，`MISSING_A`/`MISSING_B`＝一边缺件，两类行可区分 | `bridge-sha-v1.sh:18` |
| 14 | `lint-paste-v1.sh`／`gen-args-paste-v1.py` | args 块里出现 GPUStack 会尾补的三条之一（`--host`／`--port`／`--served-model-name`） | `lint-paste-v1.sh:33`、`gen-args-paste-v1.py:42` |
| 15 | `lint-paste-v1.sh` | env 块键重复 | `lint-paste-v1.sh:34` |
| 16 | `lint-paste-v1.sh`／`gen-args-paste-v1.py` | 内容断言红的共用码位：行形态非法、占位符残留、行数≠24、flag 缺失/重复/集合不符、`--max-model-len` 取值不符、与内置常量表逐字或顺序不符（L11-L13 与 A1-A7 全归此码，靠点名文案区分） | `lint-paste-v1.sh:35-39`、`gen-args-paste-v1.py:43-45` |
| 19 | `gate-tree-unchanged-v1.sh` | 上游作用域内存在编译产物（`__pycache__`／`*.pyc`／`*.pyo`），须先清理 | `gate-tree-unchanged-v1.sh:17` |
| 21-28 | `verify-gpustack-form-v1.sh` | 运行期判据：21 健康探针非 200 或未就绪／22 chat content 空／23 模型清单未回显对外名／24 启动日志出现字面 `${`／25 出现 `--speculative-config`／26 出现 ray 执行器注入／27 `--neg-pcieipc` 下日志仍出现 `FLASHINFER_PCIE_IPC`／28 拼装三段顺序失衡 | `verify-gpustack-form-v1.sh:82-90` |

## 三、乙层（镜像内 RUN 面与装机件）

| 码 | 归属件 | 判的是什么 | 声明处 |
|---|---|---|---|
| 0 | 全部乙层件 | 通过（含装机件的幂等再跑：改前改后 sha 一致） | `install-pcieipc.sh:33`、`install-incple-v1.sh:29`、`lint-count-v1.sh:17` |
| 1 | `Dockerfile.v1` 层 1 成对门 | `PCIEIPC_DEFER` 与 `PCIEIPC_IMPORT_GATE_LABEL` 不成对（只允许 `0:hard`／`1:deferred`）——**未登记缺口，见第五节 G1** | `Dockerfile.v1:59` |
| 2 | 装机与门禁件 | 定位/前置类红：site-packages 探测失败、前置物料缺失、证据目录不可写、落点守卫判死、`..` 段；`gate-check-v3.py` 的 `GATE_FAIL` 也一律走 2（`sys.exit(2)`），故 `gate-evidence-v1.sh` 的前置红与 gate-check 红同码，靠文案与 JSON 里的 `gate_check.rc` 区分 | `install-pcieipc.sh:34`、`install-incple-v1.sh:30`、`lint-count-v1.sh:18`、`gate-evidence-v1.sh:34-37`、`gate-check-v3.py:40-41` |
| 3-9 | `install-pcieipc.sh` | 幂等门逐条：3 `comm/__init__.py` 已含 marker／4 `comm/` 下已存在／5 `data/` 下 csrc 或 cuh 已存在／6 `jit/comm.py` 已含函数／7 trace 模板已含函数／8 manifest 解析·落盘·sha 对拍异常／9 追加块缺失或行数增量不符 | `install-pcieipc.sh:35-41` |
| 11a | `install-pcieipc.sh` | 运行期 import 门禁失败（字母后缀把同数字的 11b 分开） | `install-pcieipc.sh:42` |
| 11（＝11b） | `patch_ple_inc.py`，由 `install-incple-v1.sh` 原样透传 | 锚点缺失或命中数≠1（整行精确匹配，缩进参与比对；红即零写盘） | `patch_ple_inc.py:41`、`install-incple-v1.sh:32-33` |
| 12 | `install-incple-v1.sh` 与 `patch_ple_inc.py` 同码位 | incple 断言红：INCConfig 计数不符、终态 `py_compile` 失败、临时件回读契约失败；patch 侧为"临时件验绿后原子落位"，此红面下目标件必然未动——install 第 4 步再对"patch 非零而目标件 sha 变了"做原子性自证，同样红 12 | `install-incple-v1.sh:34-36`、`patch_ple_inc.py:42-43` |
| 17 | `lint-count-v1.sh` | 件数门红的共用码位：存在性断言红、件数不符、清单条目与盘上不符、行格式非法、sha 不符、重复键（简报只给 17 一个码，靠点名文案区分） | `lint-count-v1.sh:19-21` |
| 18 | `Dockerfile.v1` 层 3（时区） | 本层唯一红码：tzdata 未装成、`TZ` 未落进容器环境、`date +%z` 非 `+0800`、`/etc/localtime` 与上海 tzfile 不一致、`/etc/timezone` 文本不符，以及 date/printenv/cmp/cat 工具缺失（工具缺失单列文案仍归 18） | `Dockerfile.v1:181`、`:189`、`:199-210` |
| 12 | `Dockerfile.v1` 层 2 幂等复跑门 | 第二次运行未报 `IDEMPOTENT_NO_CHANGE`（改前改后 sha 不一致）——与 incple 断言同码位、不同判据 | `Dockerfile.v1:118` |
| 21 | `Dockerfile.bench-v1` harness 段 | `NPM_REGISTRY` 在本 stage 内取到空值（跨 stage ARG 未重声明，或 `--build-arg` 传了空值） | `Dockerfile.bench-v1:137`、声明见 `:102` |
| 22 | `Dockerfile.bench-v1` harness 段 | npm 实际生效源≠请求值（`npm config set` 静默失败） | `Dockerfile.bench-v1:143` |
| 23 | `Dockerfile.bench-v1` harness 段 | 生效源不含 npmmirror＝国内源默认值未落地；改走官方源须显式改这一行的字面量并在构建留档 | `Dockerfile.bench-v1:146`、`:103-104` |
| 24 | `Dockerfile.bench-v1` bench 段 | 三条 LABEL 的源变量有空值（证据行会静默为空），末行文案带 `missing=` 计数 | `Dockerfile.bench-v1:171`、声明见 `:107` |

## 四、丙层（总入口收敛）

`build-gpustack-v1.sh --build` 把 docker 的非零统一收进 **20＝构建失败**，原始 rc 写进文案；透传会把 docker 的 1/2/7/8/9/125 读成"用法/上游/补丁"这些保留语义（`build-gpustack-v1.sh:29-31`）。因此乙层件的红码在构建日志里以文案形式可见，作为 rc 只能在件被单独调用时读到。

## 五、未登记与重叠面（如实列出，本表不代为裁决）

- **G1 未登记码位 1**：`Dockerfile.v1:59` 的成对门用 `exit 1`，而该文件 `:22-23` 列举的层内码（17、11a、11b、12、18、2、gate-check 自有 2）里没有 1。两种处置都成立——把 1 登记进本表，或改判到已登记的 2——本表只登记现状，改不改由交付线裁。
- **G2 同数字跨层复用**：21/22/23/24 在甲层是 `verify-gpustack-form-v1.sh` 的运行期判据，在乙层是 `Dockerfile.bench-v1` 的构建期 RUN 断言，语义完全不同。判读规则：先看红码出现在哪一侧（宿主 rc 还是构建日志），再查对应层的表。
- **G3 同数字跨件复用**：2 被所有件共用（口径一致＝用法/前置），3-9 与甲层的 3/4/5/6/7/10 分属不同件；因分属甲/乙两层，不构成同层冲突。
- **G4 预留面与现状重叠**：`build-gpustack-v1.sh:31` 写"任务 9 的映射表须按 21 起算套件子门"，而 21-28 已被任务 8 的运行期判据占用（甲层）。该预留与实际占用重叠，本表如实登记，不声称已消解。

## 六、维护规则

- 新增码位：先在本表加一行（含声明处 `文件:行号`），再在件里使用该码；两边同一提交落地，避免"脚本用了、表里没有"。
- 各件头注只列举**自己**用到的码位并指向本表，不复制全表快照——滚动全表每加一个任务就会膨胀，别处放快照必然腐化（`gate-tree-unchanged-v1.sh:19-21` 同口径）。
- 判据文本以脚本为准，本表不复制判据实现；两边不一致时逐条对脚本与本表，改本表。
- 本表被指的地方只有一处：`gate-tree-unchanged-v1.sh:20`。那一行是注释级指向，不构成运行时依赖（见第〇节）。
