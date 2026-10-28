// extract_questions_v1.mjs — 从 sql_benchmark.html 抽出 questionsList（题目+期望值）为 JSON
// 用法: node extract_questions_v1.mjs <html路径> <输出json路径>
import fs from "node:fs";
import vm from "node:vm";

const [, , htmlPath, outPath] = process.argv;
const html = fs.readFileSync(htmlPath, "utf8");
const m = html.match(/const questionsList = (\[[\s\S]*?\]);\n/);
if (!m) {
  console.error("FAIL: 未匹配到 questionsList");
  process.exit(1);
}
const list = vm.runInNewContext("(" + m[1] + ")", Object.create(null), { timeout: 10000 });
if (!Array.isArray(list) || list.length !== 25) {
  console.error("FAIL: 题目数异常 " + (list && list.length));
  process.exit(1);
}
for (const q of list) {
  if (!q.id || !Array.isArray(q.columns) || !q.first_row || typeof q.row_count !== "number") {
    console.error("FAIL: 题目字段缺失 id=" + q.id);
    process.exit(1);
  }
}
fs.writeFileSync(outPath, JSON.stringify(list));
console.log("OK 已写出 " + list.length + " 题 -> " + outPath);
