# parse_traces_v2.py — 把 trace CSV 拆成「每题按步候选 SQL + results_ok 携带的最终答案 SQL」
# 用法: python parse_traces_v2.py <trace目录> <candidates.json 输出路径>
# v2: ① 兼容 trace_round_*.csv / round_*.csv / test_trace_*.csv 命名
#     ② 解析 results_ok 的 args.sql（sm75-finalanswer-v1 显式标记），写入 final_sql
#     ③ Round 列可选（run_server.py 导出的 CSV 无 Round），缺省从文件名推
import csv
import glob
import json
import os
import re
import sys

TRACE_DIR, OUT = sys.argv[1], sys.argv[2]


def col(headers, prefix):
    for h in headers:
        if h.startswith(prefix):
            return h
    raise SystemExit("FAIL: 找不到列 " + prefix)


def opt_col(headers, prefix):
    for h in headers:
        if h.startswith(prefix):
            return h
    return None


def round_from_name(path):
    m = re.search(r"round[_-]?(\d+)", os.path.basename(path), re.I)
    return "Round " + m.group(1) if m else "Round 1"


def parse_args(raw):
    """trace 里的 args 是 JSON 字符串字面量；返回 dict 或 None"""
    try:
        inner = json.loads(raw)
        if isinstance(inner, str):
            inner = json.loads(inner)
        return inner if isinstance(inner, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_trace(text):
    """返回 ({idx,sql,exec,parse_error}..., final_sql)"""
    lines = text.split("\n")
    steps = []
    final_sql = None
    for i, line in enumerate(lines):
        m = re.match(r"^\[工具调用\] run_sql_query with args: (.*)$", line)
        if m:
            args = parse_args(m.group(1))
            sql = args.get("sql") if args else None
            exec_state = "unknown"
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].startswith("[执行结果]"):
                    exec_state = "ok"
                    break
                if lines[j].startswith("[执行报错]"):
                    exec_state = "err"
                    break
            steps.append({"idx": len(steps) + 1, "sql": sql, "exec": exec_state,
                          "parse_error": sql is None})
        m2 = re.match(r"^\[工具调用\] results_ok with args: (.*)$", line)
        if m2:
            args = parse_args(m2.group(1))
            if args and isinstance(args.get("sql"), str) and args["sql"].strip():
                final_sql = args["sql"]
    return steps, final_sql


paths = set()
for pattern in ("trace_round_*.csv", "round_*.csv", "test_trace_*.csv"):
    paths.update(glob.glob(os.path.join(TRACE_DIR, pattern)))

runs = []
for path in sorted(paths):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        cRound = opt_col(headers, "Round")
        cQid = col(headers, "Question ID")
        cStatus = col(headers, "Status")
        cGen = col(headers, "Generated SQL")
        cTrace = col(headers, "Full Dialogue Trace")
        cExpRows = col(headers, "Expected Row Count")
        cActRows = col(headers, "Actual Row Count")
        cExpCols = col(headers, "Expected Column Count")
        cActCols = col(headers, "Actual Column Count")
        for row in reader:
            steps, final_sql = parse_trace(row[cTrace] or "")
            runs.append({
                "file": os.path.basename(path),
                "round": row[cRound].strip() if cRound else round_from_name(path),
                "qid": row[cQid].strip(),
                "status": row[cStatus].strip(),
                "official_sql": row[cGen],
                "n_steps": len(steps),
                "n_parsed": sum(1 for s in steps if s["sql"]),
                "exp_rows": row[cExpRows],
                "act_rows": row[cActRows],
                "exp_cols": row[cExpCols],
                "act_cols": row[cActCols],
                "steps": steps,
                "final_sql": final_sql,
            })

json.dump(runs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
bad = [r for r in runs if r["n_steps"] != r["n_parsed"]]
print("OK 题次=%d  步数解析不齐的题次=%d  带显式标记的题次=%d" % (
    len(runs), len(bad), sum(1 for r in runs if r["final_sql"])))
for r in bad[:10]:
    print("  需人工看: %s %s steps=%s parsed=%s" % (r["round"], r["qid"], r["n_steps"], r["n_parsed"]))
