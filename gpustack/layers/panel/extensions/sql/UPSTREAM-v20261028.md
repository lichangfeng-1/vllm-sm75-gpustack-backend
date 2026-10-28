# SQL 判分扩展 sm75-finalanswer-v1（2026-10-28）

- 上游仓：`wangxian001/SQL_LLM_benchmark`
- 上游 pin：`2aa49062f411928acd02d90ac8dddffe1b77a00e`（V1.2.4，2026-08-04，上游无更新）
- 本地补丁：`sm75-finalanswer-v1`，只改判分器“选哪一发查询”，不改 DuckDB 版本、不改行/列/首行判定与容差
- 记录日期：2026-10-28

## 1 文件对照

| 文件 | 上游 sha256 | 补丁后 sha256 | 字节 |
|---|---|---|---|
| `sql_benchmark.html` | `f8209c2dd5421070be58ffac7b6bf4b0cd6c713f51ed4fec7f3ef4bf4abcf3fc` | `28acd6a809dd4b8ccc45c427dfdbadefc7a8d047375b7812e2933e313de57fc7` | 194,452 → 199,850 |
| `sql_benchmark-v20261028.html` | — | 同上（与补丁后逐字节相同） | 199,850 |
| `manifest-v20261028.json` | — | 记录补丁后 sha、`scoring=sm75-finalanswer-v1`、`original_ui=1` | — |
| `UPSTREAM-v20261028.md` | — | 本文件 | — |

上游原件 `sql_benchmark.html` 保持原位、零改动，作为回退件。

## 2 补丁内容（6 处）

1. 文件头/标题版本改为 `V1.2.4-sm75.1（20261028）`，与上游 V1.2.4 可区分。
2. 系统提示词加一句：调用 `results_ok` 时用可选 `sql` 参数带上“最终答案那一发”的原文（不要求它必须是最后一发）。
3. `results_ok` 工具 schema 增加可选 `sql`（`required` 仍为空），不新增工具、不增加往返。
4. `verifyResults` 增加 `checkedFirstRowColumns`：期望 `first_row` 的列在实际结果里一个都不存在时，`firstRowMatch` 强制 false（修假阳性；判分结论不变，`columnNamesMatch` 本来就 Fail）。
5. 新增 `cleanColumnKey` / `normalizeSqlForMatch` / `resultShapeMatches` / `selectAnswerQuery`，放在 `formatSchema` 与 `executeSql` 之间，确保 `ultra/source/console/sql-benchmark-runner.mjs` 的抽取锚点仍可用。
6. `runQuestionBenchmark` 记录每个成功执行的 `{step, sql, result}`；`results_ok` 时选一发回填 `generatedSql`/`queryResult`，并返回 `answerSelection`（规则/步号/候选数）；结果面板显示“判分所用查询”。

## 3 识别规则与口径

每题只选一发，优先级：

1. **explicit**：`results_ok.sql` 按空白归一化后在本轮已成功执行的查询里从后往前命中；
2. **shape**：最后一个“行数/列数/列名集合”与题目期望一致的候选；
3. **last-executed**：一个签名都不匹配时退回上游“末发”语义（答错仍 Fail）。

判定完全复用上游 `verifyResults`/`isCorrect`（行数、列数、列名、首行数值、`5×10^-(小数位+1)` 容差、DuckDB 1.4.4 均不变）。口径名 `sm75-finalanswer-v1`，与上游“末发”口径不可直接混用；对外同时给两个数。

## 4 验证（2026-10-28，本机只读）

- 合成 trace 单测 18 条断言全过（只答一发/答对后自查/答错后探查/期望列缺失/显式标记/标记未命中/部分列缺失/同形探查限界；一条消息多 SQL 块由集成测试 E 覆盖）；上游负控显示旧 `firstRowMatch=true` 假阳性。
- agent 循环集成测试 13 条断言全过（假 `callLLM`/`executeSql` 驱动真实 `runQuestionBenchmark`，覆盖 explicit 与 shape 两条路径）。
- 2026-10-07 归档 4 轮 100 题次复判（本机 duckdb 1.4.4，同一批 CSV）：
  - 页面 87/100 → 新口径 96/100（Q19 无 `ORDER BY` 随机项按页面原判计）；同一次执行对照为上游 86 → 新口径 95。
  - 历史 10 次误杀修复 9 次；R4 Q21 因第 8 步把已答对的口径改错，按“最终答案错”保持 Fail；3 次真错不变；规则性退化 0。
  - 识别命中：shape 99、last-executed 1、explicit 0（旧 trace 无标记）；选中步早于末发 9 次，与修复数一致。

复现命令与逐项断言见本仓 `gpustack/tests/sql_judge/README.md`。

## 5 部署映射（本阶段未执行）

面板入口固定在 `public/app.js` 的 `/bench-app/sql/sql_benchmark.html`，所以“加一个带版本后缀的新文件”不会自动生效；切换必须替换**规范名**入口。

```bash
# 在交付仓根目录、有 Docker 与父镜像的宿主机上
PARENT=vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15 IMAGE=vllm-sm75-gpustack:v0.1.7-beta2-v13-opt15-sqljudge-v1 bash gpustack/build-sql-finalanswer-v1.sh
```

薄层只覆盖 `/opt/sm75-workbench/extensions/sql/` 下的入口 HTML、版本化副本、manifest 与本文档，不动推理层。Dockerfile 的 COPY 源全部用带版本后缀的文件，直接以扩展目录为上下文构建也不会误装上游入口；脚本内置五道验装（入口 sha、版本化副本 sha、manifest 标记、HTML 标记、firstRowMatch 补丁）。

**只换镜像不会切换已有面板卷**：运行时读持久卷 `SM75_CONSOLE_ROOT`（默认 `/var/lib/gpustack/panel`），`panel-sidecar.py:125-140` 与 `server.mjs` 都只在 `extensions/sql` 不存在/无 `manifest.json` 时才从镜像 seed。切换必须另开窗口刷新卷：

1. 备份面板卷里的 `extensions/sql` 目录（`docker cp <面板容器>:/var/lib/gpustack/panel/extensions/sql <宿主备份目录>`）。
2. 停面板，删除或改名卷内 `extensions/sql`（保留备份，不直接销毁）。
3. 用新镜像起面板；sidecar 会把镜像内的补丁版重新 seed 到卷。
4. 验装：卷内 `sql_benchmark.html` 的 sha256＝`28acd6a8…57fc7`；页面版本标签为 `V1.2.4-sm75.1（20261028）`；跑一题后结果面板出现“判分所用查询”。

本阶段不改面板卷、不重启、不换模型行；以上四步属于后续面板窗口。

## 6 回退

- 回退必须同时处理**镜像与卷**：只切回父镜像 tag 不会恢复旧页面，因为卷里已有 `manifest.json`，sidecar 不会覆盖。
- 正确回退：停面板 → 用步骤 1 的备份恢复卷内 `extensions/sql` → 用父镜像（`…-v13-opt15`）起面板。
- 全量重建 `Dockerfile.allinone` 时，仓内 `sql_benchmark.html` 仍是上游原件，默认构建不带补丁（新文件不静默生效）；卷若被刷新过，仍需按上一条恢复。
- 版本化副本与本文档可长期保留，供审计/对照。

## 7 已知限界

- 无 `ORDER BY` 的题首行随机，判分本身有 ±1 题波动（Q19 类），与识别规则无关。
- 上游 agent 循环只处理一条 assistant 消息里的第一个 tool call；文内其它 SQL 块不执行、不参与判分。
- 同形探查（行列名与期望全同、数值不同）会被当成最后一发答案形状候选（已知限界，测试 ⑨ 固定该行为）。
- 若模型在答案形状查询之后又执行了异形查询并确认，本规则无法区分“探查”与“改答”，仍取答案形状那一发；`results_ok.sql` 显式标记可消除该歧义。
- `SEED-MANIFEST.json` 是上游种子身份记录（上游文件哈希），不随本地补丁更新；运行时只读 `manifest.json` 的 `original_ui`。
- `shape` 规则使用题目的期望行列签名（不含数值）做候选筛选；数值判定完全独立，不用期望值放水。
- 显式标记只在模型主动带 `sql` 时生效；不带时走 shape，带错时记日志并回退 shape。
