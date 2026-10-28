# rejudge_v5.py — 10-07 归档 trace 修复前后对照复判（只读）
# 用法: python rejudge_v5.py questions.json candidates.json <tables_dir> <out.json>
# 口径：
#   upstream = 最后一发成功执行的 SQL（页面原语义）
#   fixed    = sm75-finalanswer-v1：最后一个「行数/列数/列名集合」与期望一致的候选；无则回退末发
# 判分规则与 sql_benchmark.html 的 verifyResults/isCorrect 一致（含 checkedFirstRowColumns 修复）。
import json
import os
import re
import sys

import duckdb

assert duckdb.__version__ == "1.4.4", "需要 duckdb 1.4.4，当前 " + duckdb.__version__

questions_path, candidates_path, tables_dir, out_path = sys.argv[1:5]

Q = {str(q["id"]): q for q in json.load(open(questions_path, encoding="utf-8"))}
R = json.load(open(candidates_path, encoding="utf-8"))

con = duckdb.connect()
for t in ("Customer", "Date", "Product", "Reseller", "Sales", "Sales_Order", "Sales_Territory"):
    con.execute('CREATE OR REPLACE TABLE "%s" AS SELECT * FROM read_csv_auto(?)' % t,
                [tables_dir + "/" + t + ".csv"])

clean = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
HAS_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.I)
def norm(x):
    text = re.sub(r"\s+", " ", str(x or ""))
    return re.sub(r";\s*$", "", text).strip()


def js_str(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def score(qid, sql):
    """返回 dict：shape（行/列/列名全合）、pass（上游四检全过）、明细"""
    q = Q[qid]
    head = re.sub(r"^\s*(--[^\n]*\n|\s)+", "", sql or "").lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        return {"shape": False, "pass": False, "error": "非 SELECT"}
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except Exception as e:
        return {"shape": False, "pass": False, "error": str(e)[:160]}
    amap = {clean(c): c for c in cols}
    emap = {clean(c) for c in q["columns"]}
    missing = [c for c in q["columns"] if clean(c) not in amap]
    extra = [c for c in cols if clean(c) not in emap]
    row_ok = len(rows) == q["row_count"]
    colcnt_ok = len(cols) == len(q["columns"])
    names_ok = not missing and not extra
    shape = row_ok and colcnt_ok and names_ok
    diffs = []
    checked = 0
    if rows:
        idx = {c: i for i, c in enumerate(cols)}
        for c, ev in q["first_row"].items():
            key = clean(c)
            if key not in amap:
                continue
            checked += 1
            av = rows[0][idx[amap[key]]]
            es = js_str(ev)
            dec = len(es.split(".")[1]) if "." in es else 0
            tol = 5 * 10 ** -(dec + 1)
            try:
                a, e = float(av), float(ev)
                eps = 2.220446049250313e-16 * max(abs(a), abs(e))
                num_ok = abs(a - e) <= tol + eps
            except (TypeError, ValueError):
                num_ok = False
            if not (num_ok or str(av) == es):
                diffs.append({"col": c, "exp": es, "act": str(av)})
    first_ok = len(diffs) == 0 and len(rows) > 0 and checked > 0
    return {
        "shape": shape, "pass": shape and first_ok,
        "rows": len(rows), "cols": len(cols), "missing": missing, "extra": extra,
        "row_ok": row_ok, "colcnt_ok": colcnt_ok, "names_ok": names_ok,
        "first_ok": first_ok, "diffs": diffs, "checked_first_row_cols": checked,
    }


hard, warn, runs = [], [], []
for n, r in enumerate(R, 1):
    qid = r["qid"].replace("Q", "").strip()
    if qid not in Q:
        hard.append((r["round"], r["qid"], "题目ID不在 questions.json"))
        continue
    if r["status"] not in ("pass", "fail"):
        hard.append((r["round"], r["qid"], "页面状态非 pass/fail: " + str(r["status"])))
        continue

    # 每题每条 SQL 只执行一次：官方自检与候选判分复用同一结果（避免无 ORDER BY 的二次随机）
    cands = []
    for s in r["steps"]:
        if not s["sql"] or s["exec"] != "ok":
            continue
        det = score(qid, s["sql"])
        cands.append({"step": s["idx"], "sql": s["sql"], "result": det})
    if not cands:
        hard.append((r["round"], r["qid"], "无可执行候选"))
        continue

    official = None
    if r["official_sql"]:
        for c in cands:
            if norm(c["sql"]) == norm(r["official_sql"]):
                official = c["result"]
                break
        if official is None:
            official = score(qid, r["official_sql"])
    if official is None:
        hard.append((r["round"], r["qid"], "缺少 official_sql"))
        continue
    if official["pass"] != (r["status"] == "pass"):
        rec = (r["round"], r["qid"], "官方复算 %s vs 页面 %s" % (official["pass"], r["status"]))
        if official["row_ok"] and official["names_ok"] and not HAS_ORDER_BY.search(r["official_sql"]):
            warn.append(rec)
        else:
            hard.append(rec)
            continue

    last = cands[-1]
    final_sql = r.get("final_sql")
    explicit = None
    if final_sql:
        wanted = norm(final_sql)
        for c in reversed(cands):
            if norm(c["sql"]) == wanted:
                explicit = c
                break
    shaped = [c for c in cands if c["result"]["shape"]]
    if explicit is not None:
        selected, rule = explicit, "explicit"
    elif shaped:
        selected, rule = shaped[-1], "shape"
    else:
        selected, rule = last, "last-executed"
    explicit_unmatched = bool(final_sql) and explicit is None
    noise = (not HAS_ORDER_BY.search(selected["sql"] or "")
             and selected["result"]["row_ok"] and selected["result"]["names_ok"])
    last_no_order_by = (not HAS_ORDER_BY.search(last["sql"] or "")
                        and last["result"]["row_ok"] and last["result"]["names_ok"])
    runs.append({
        "round": r["round"], "qid": r["qid"], "page_status": r["status"],
        "answer_steps": [c["step"] for c in cands if c["result"]["pass"]],
        "n_steps": r["n_steps"], "n_candidates": len(cands),
        "upstream": {"step": last["step"], "pass": bool(last["result"]["pass"])},
        "fixed": {"step": selected["step"], "pass": bool(selected["result"]["pass"]),
                  "rule": rule, "shape_steps": [c["step"] for c in shaped],
                  "explicit_unmatched": explicit_unmatched},
        "no_order_by": noise,
        "last_no_order_by": last_no_order_by,
    })
    if n % 25 == 0:
        print("进度 %d/%d" % (n, len(R)), flush=True)

if hard:
    print("\n硬自检未通过 %d 条，结果不可信，已终止：" % len(hard))
    for b in hard[:15]:
        print("  ", b[0], b[1], b[2])
    sys.exit(2)

json.dump({"runs": runs, "warn": warn}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n自检：官方 SQL 复算与页面判定一致（随机顺序告警 %d 条）" % len(warn))
for w in warn:
    print("  告警 %s %s：%s（该 SQL 无 ORDER BY，首行随机）" % w)

rounds = sorted({o["round"] for o in runs}, key=lambda s: int(re.sub(r"\D", "", s)))
print("\n%-9s %-10s %-12s %-12s %-10s" % ("轮次", "页面", "上游复算", "新口径", "修回/退化"))
page_tot = up_tot = fx_tot = 0
for rd in rounds:
    sub = [o for o in runs if o["round"] == rd]
    page = sum(1 for o in sub if o["page_status"] == "pass")
    up = sum(1 for o in sub if o["upstream"]["pass"])
    fx = sum(1 for o in sub if o["fixed"]["pass"])
    fixed_n = sum(1 for o in sub if o["page_status"] == "fail" and o["fixed"]["pass"])
    regress = sum(1 for o in sub if o["page_status"] == "pass" and not o["fixed"]["pass"] and not o["no_order_by"])
    page_tot += page; up_tot += up; fx_tot += fx
    print("%-9s %d/%-7d %d/%-9d %d/%-9d +%d/-%d" % (
        rd, page, len(sub), up, len(sub), fx, len(sub), fixed_n, regress))

tot = len(runs)
print("%-9s %d/%-7d %d/%-9d %d/%-9d" % ("合计", page_tot, tot, up_tot, tot, fx_tot, tot))
print("页面 %.1f%%｜上游复算 %.1f%%｜新口径 %.1f%%" % (100.0 * page_tot / tot, 100.0 * up_tot / tot, 100.0 * fx_tot / tot))
noise_n = sum(1 for o in runs if o["page_status"] == "pass" and not o["fixed"]["pass"] and o["no_order_by"])
print("若把无 ORDER BY 的随机项按页面原判计入，新口径 %d/%d (%.1f%%)" % (
    fx_tot + noise_n, tot, 100.0 * (fx_tot + noise_n) / tot))
from collections import Counter
print("识别规则：" + "、".join("%s=%d" % (k, v) for k, v in sorted(Counter(o["fixed"]["rule"] for o in runs).items())))

fixed_up = [o for o in runs if o["page_status"] == "fail" and o["fixed"]["pass"]]
regress = [o for o in runs if o["page_status"] == "pass" and not o["fixed"]["pass"] and not o["no_order_by"]]
noise = [o for o in runs if o["page_status"] == "pass" and not o["fixed"]["pass"] and o["no_order_by"]]
stay_fail = [o for o in runs if not o["fixed"]["pass"] and o not in noise]
print("\n修回 Pass %d 条：" % len(fixed_up))
for o in fixed_up:
    print("  %s %s 选中第 %s 步（%s；形状命中 %s），原答对步 %s" % (
        o["round"], o["qid"], o["fixed"]["step"], o["fixed"]["rule"], o["fixed"]["shape_steps"], o["answer_steps"]))
print("\n退化 Fail %d 条（已排除无 ORDER BY 随机项）：" % len(regress))
for o in regress:
    print("  %s %s selected=%s" % (o["round"], o["qid"], o["fixed"]["step"]))
print("\n无 ORDER BY 随机项 %d 条（页面原判 Pass，本次复算首行随机）：" % len(noise))
for o in noise:
    print("  %s %s selected=%s 答对步 %s" % (o["round"], o["qid"], o["fixed"]["step"], o["answer_steps"]))
print("\n仍 Fail %d 条：" % len(stay_fail))
for o in stay_fail:
    kind = "曾答对但最终答案错" if o["answer_steps"] else "无任何候选通过"
    print("  %s %s 选中第 %s 步（%s；答对步 %s）=> %s" % (
        o["round"], o["qid"], o["fixed"]["step"], o["fixed"]["rule"], o["answer_steps"], kind))

bad = [o for o in runs if o["upstream"]["pass"] != (o["page_status"] == "pass")]
hard_bad = [o for o in bad if not (o["last_no_order_by"] and ((o["page_status"] == "pass") != o["upstream"]["pass"]))]
print("\n上游复算与页面不一致：%d（其中无 ORDER BY 随机 %d）" % (len(bad), len(bad) - len(hard_bad)))
if hard_bad:
    for o in hard_bad:
        print("  硬不一致 %s %s page=%s upstream=%s" % (o["round"], o["qid"], o["page_status"], o["upstream"]["pass"]))
    sys.exit(3)
