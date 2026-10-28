// test-selector-v1.mjs — 合成 trace 单测：直接抽取新版 HTML 的判分函数（与 sql-benchmark-runner.mjs 同锚点）
// 用法: node test-selector-v1.mjs <sql_benchmark-v20261028.html>
import fs from 'node:fs';
import vm from 'node:vm';

const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
const between = (a, b) => {
  const i = html.indexOf(a), j = html.indexOf(b, i);
  if (i < 0 || j < 0) throw Error('锚点缺失: ' + a);
  return html.slice(i, j);
};
const questions = html.match(/const questionsList = (.*);/)[0];
const context = vm.createContext({ console });
vm.runInContext(questions + '\n' + between('function formatSchema(schema)', 'async function executeSql(sql)') + '\nglobalThis.core={questionsList,verifyResults,isCorrect,selectAnswerQuery,resultShapeMatches};', context, { timeout: 1000 });
const { questionsList, verifyResults, isCorrect, selectAnswerQuery, resultShapeMatches } = context.core;

let n = 0;
const ok = (cond, msg) => { if (!cond) throw Error('FAIL: ' + msg); n++; console.log('  ok  ' + msg); };

const q = questionsList.find(item => item.id === 2);
const expected = q;
const str = v => String(v);
const answerRow = Object.fromEntries(q.columns.map(c => [c, str(q.first_row[c])]));
const answerResult = { columns: q.columns.slice(), rows: [answerRow, Object.fromEntries(q.columns.map(c => [c, '0']))], numRows: q.row_count };
const wrongAnswerResult = { ...answerResult, rows: [{ ...answerRow, total_revenue: '1' }, answerResult.rows[1]] };
const probeResult = {
  columns: ['Order Quantity', 'Unit Price', 'Extended Amount', 'Product Standard Cost', 'Total Product Cost', 'Sales Amount', 'Unit Price Discount Pct'],
  rows: Array.from({ length: 10 }, (_, i) => ({ 'Order Quantity': String(i + 1), 'Unit Price': '1', 'Extended Amount': '1', 'Product Standard Cost': '1', 'Total Product Cost': '1', 'Sales Amount': '1', 'Unit Price Discount Pct': '0' })),
  numRows: 10,
};
const probeResult2 = { ...probeResult, numRows: 5, rows: probeResult.rows.slice(0, 5) };
const exec = (step, sql, result) => ({ step, sql, result });

console.log('Q2 expected: rows=%d cols=%d', q.row_count, q.columns.length);

// ① 只答一发
{
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT answer', answerResult)], null);
  ok(sel.rule === 'shape' && sel.query.step === 1, '① 只答一发：选中第 1 步（shape）');
  ok(isCorrect(verifyResults(expected, sel.query.result)), '① 只答一发：判 Pass');
}

// ② 答对后自查/探查
{
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT answer', answerResult), exec(2, 'SELECT probe', probeResult), exec(3, 'SELECT probe2', probeResult2)], null);
  ok(sel.rule === 'shape' && sel.query.step === 1, '② 答对后探查：仍选第 1 步');
  ok(isCorrect(verifyResults(expected, sel.query.result)), '② 答对后探查：判 Pass');
}

// ③ 答错后探查
{
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT wrong', wrongAnswerResult), exec(2, 'SELECT probe', probeResult)], null);
  ok(sel.query.step === 1, '③ 答错后探查：选第 1 步（形状唯一的答案候选）');
  ok(!isCorrect(verifyResults(expected, sel.query.result)), '③ 答错后探查：判 Fail');
}

// ⑤ 期望列缺失：firstRowMatch 不得假阳性
{
  const report = verifyResults(expected, probeResult);
  ok(report.firstRowMatch === false, '⑤ 期望列全缺失：firstRowMatch=false');
  ok(report.checkedFirstRowColumns === 0, '⑤ 期望列全缺失：checkedFirstRowColumns=0');
  ok(report.missingColumns.length === q.columns.length, '⑤ 期望列全缺失：missingColumns 全列');
  ok(!isCorrect(report), '⑤ 期望列全缺失：判 Fail');
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT probe', probeResult)], null);
  ok(sel.rule === 'last-executed' && sel.query.step === 1, '⑤ 无答案形状候选：回退末发');
  ok(!isCorrect(verifyResults(expected, sel.query.result)), '⑤ 回退末发：判 Fail');
}

// ⑥ 显式标记优先（空白归一化，大小写不动）
{
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT answer', answerResult), exec(2, 'SELECT probe', probeResult)], '  SELECT   answer ; ');
  ok(sel.rule === 'explicit' && sel.query.step === 1, '⑥ 显式标记：空白归一化后命中第 1 步');
}

// ⑦ 标记未命中（含大小写不同）：回退签名识别并标记
{
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT answer', answerResult), exec(2, 'SELECT probe', probeResult)], 'select answer');
  ok(sel.rule === 'shape' && sel.query.step === 1 && sel.explicitUnmatched === true, '⑦ 标记大小写不同：不误配，回退 shape 且 explicitUnmatched=true');
}

// ⑧ 首行有部分期望列时仍按上游语义比对（不回归）
{
  const partial = { columns: q.columns.filter(c => c !== 'gross_margin_pct'), rows: [{ ...answerRow }], numRows: 2 };
  const report = verifyResults(expected, partial);
  ok(report.checkedFirstRowColumns === q.columns.length - 1, '⑧ 部分列缺失：仍比对存在的期望列');
  ok(report.firstRowMatch === true && report.columnNamesMatch === false && !isCorrect(report), '⑧ 部分列缺失：首行吻合但整体 Fail（列名不符）');
}

// ⑨ 已知限界：同形探查（行列名全同、数值不同）会被当成最后一个答案形状候选
{
  const sameShapeProbe = { ...answerResult, rows: [{ ...answerRow, total_revenue: '1' }, answerResult.rows[1]] };
  const sel = selectAnswerQuery(expected, [exec(1, 'SELECT answer', answerResult), exec(2, 'SELECT same_shape_probe', sameShapeProbe)], null);
  ok(sel.query.step === 2, '⑨ 同形探查：规则取最后一发形状匹配（已知限界）');
  ok(!isCorrect(verifyResults(expected, sel.query.result)), '⑨ 同形探查：判 Fail（不洗白）');
}

console.log('\nALL PASS (%d assertions)', n);
