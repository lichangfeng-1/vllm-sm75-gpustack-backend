# SQL_LLM_benchmark 本地补丁说明（2026-10-29）

- 上游：`wangxian001/SQL_LLM_benchmark`，commit `2aa49062f411928acd02d90ac8dddffe1b77a00e`
- 上游原始页面：`f8209c2dd5421070be58ffac7b6bf4b0cd6c713f51ed4fec7f3ef4bf4abcf3fc`
- 本版页面：`sql_benchmark-v20261029.html`，sha256 `6f8c836fe3702eef7909463f7051de53666617080db5c98c9dfc200eaea829da`
- 判分口径：`sm75-finalanswer-v1`（2026-10-28 的“取最终答案查询”修复继续保留）

## 2026-10-29 增量

### 1. 系统提示词补充财年/财季格式约束

在 `getSystemPrompt()` 中增加：

> Preserve the original formatting of fiscal period labels: when a result column contains fiscal year or fiscal quarter values, return them exactly as stored in the source column (for example, FY2019 or FY2019 Q4); do not strip prefixes, convert them to integers, or reformat them.

**原因：** Q10 的题干只要求返回 `prior_year` / `current_year`，没有说明年份格式；基准参考 SQL 返回源列原样字符串 `FY2019` / `FY2020`，模型返回整数 `2019` / `2020` 被判 Fail。同类题还有 Q1、Q5、Q16、Q20、Q22、Q25。把约束放在系统提示词里，是一次性覆盖同类题，避免逐题改题干造成口径不一致。

### 2. 工具结果单行大字段截断

`formatResultSummary()` 增加：

- 单个字段值最多 500 字符，超出加 `… (N chars total)`
- 整条工具结果最多 4000 字符，超出加 `… (truncated)`

**原因：** Q22 第 14 步模型写了 `LIST(d."Fiscal Quarter")` 但漏了 `DISTINCT`，首行 `qs` 字段达到约 36 万字符；工具结果虽然只取首行，首行本身把 prompt 推到 245,761 tokens，叠加 `max_tokens=16384` 后为 262,145，超过 262,144 上限 1 token，API 返回 400。截断后单条工具结果不再可能撑爆上下文。

## 部署映射

1. 面板卷 `extensions/sql/` 的规范名 `sql_benchmark.html` ← 本版文件。
2. `manifest.json` 的 `files.sql_benchmark.html` 必须同步为新 sha（`6f8c836f…`）；旧 `manifest-v20261028.json` 保留回退。
3. 版本文件 `sql_benchmark-v20261029.html` 与旧 `sql_benchmark-v20261028.html` 并存。
4. 热更后浏览器需要刷新；已经加载的页面不会自动换文件。

## 验证

- `grep -c "Preserve the original formatting of fiscal period labels"` = 1
- `grep -c "clip(firstRow[col], 500)"` = 1
- `grep -c "summary.length > 4000"` = 1
- 页面内 `sm75-finalanswer` 判分逻辑仍在
- 18002 直接取 `/sql_benchmark.html` 的 sha 等于本版 sha

## 回退

- 页面：恢复 `sql_benchmark.html.before-promptfix-20261029`（sha `28acd6a8…`）。
- 清单：恢复 `manifest.json.before-promptfix-20261029`（sha `964e80ba…`）。
- 回退后同样需要刷新页面。

## 可比性说明

本版同时改动了系统提示词与工具结果截断，**分数不再与上游原始基准逐题严格可比**。对外引用时应写明“SM75 本地口径（finalanswer + 财年格式约束 + 工具结果截断）”，不能直接与上游或其他机器的原始基准分数横比。
