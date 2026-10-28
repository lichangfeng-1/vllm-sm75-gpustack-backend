# SQL 判分修复验证包（sm75-finalanswer-v1）

本目录是 `gpustack/layers/panel/extensions/sql/sql_benchmark-v20261028.html` 的验证件，全部只读、无网络依赖（复判需要本地 DuckDB 1.4.4）。

## 1 合成 trace 单测（Node，无第三方依赖）

```bash
node gpustack/tests/sql_judge/test-selector-v1.mjs gpustack/layers/panel/extensions/sql/sql_benchmark-v20261028.html
node gpustack/tests/sql_judge/test-agentloop-v1.mjs gpustack/layers/panel/extensions/sql/sql_benchmark-v20261028.html
node gpustack/tests/sql_judge/control-upstream-v1.mjs gpustack/layers/panel/extensions/sql/sql_benchmark.html
```

- `test-selector-v1.mjs`：18 条断言，覆盖只答一发、答对后自查、答错后探查、期望列缺失、显式标记、标记未命中、部分列缺失、同形探查限界。
- `test-agentloop-v1.mjs`：13 条断言，用假 `callLLM`/`executeSql` 驱动真实 `runQuestionBenchmark`；其中 E 覆盖“一条消息多 SQL 块”（上游只执行首个 tool call，文内未执行 SQL 不参与判分）。
- `control-upstream-v1.mjs`：负控，证明上游原件在“期望列全缺失”时 `firstRowMatch=true`（假阳性），补丁后为 false。

## 2 归档 trace 复判（Python + DuckDB 1.4.4）

前置：页面按钮“导出详细测试日志到本地”得到的 CSV（兼容 `trace_round_*.csv`、`round_*.csv`、`test_trace_*.csv`；`Round` 列可选，缺省从文件名推）；扩展目录的 `tables/*.csv`。

```bash
# 1) 从页面抽题目与期望值
node gpustack/tests/sql_judge/extract_questions_v1.mjs \
  gpustack/layers/panel/extensions/sql/sql_benchmark-v20261028.html questions.json

# 2) trace CSV -> 逐题候选 SQL
python gpustack/tests/sql_judge/parse_traces_v2.py <trace_dir> candidates.json

# 3) 修复前后对照（upstream=末发，fixed=最终答案识别）
python gpustack/tests/sql_judge/rejudge_v5.py questions.json candidates.json <tables_dir> rejudge.json
```

`parse_traces_v2.py` 会同时解析 `results_ok` 携带的可选 `sql`（显式最终答案标记）写入 `final_sql`；`rejudge_v5.py` 优先用它，未命中再走形状识别。系统 Python 需装 `duckdb==1.4.4`（`pip install duckdb==1.4.4`），脚本会断言版本。脚本先做自检：官方 SQL 复算必须与页面判定一致；无 `ORDER BY` 且行列齐的随机项（双向）只告警，其它不一致 `exit 2`。

## 3 2026-10-07 归档数据上的读数

- 页面 87/100 → 新口径 96/100（Q19 无 `ORDER BY` 随机项按页面原判计）；同一次执行对照：上游 86 → 新口径 95。
- 历史 10 次误杀修复 9 次；R4 Q21 因最终答案被改错保持 Fail；3 次真错不变；规则性退化 0。

## 4 口径

`sm75-finalanswer-v1` 与上游“末发”口径不可直接混用；对外同时给两个数。判定语义（行/列/首行、容差、DuckDB 1.4.4）不变，只改“选哪一发”。

已知限界：同形探查会被当成最后一发答案形状候选；答案形状查询之后又执行异形查询并确认时，无法区分“探查”与“改答”，仍取答案形状那发（显式标记可消除）。
