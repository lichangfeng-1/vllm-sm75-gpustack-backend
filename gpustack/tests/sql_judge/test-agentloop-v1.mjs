// test-agentloop-v1.mjs — 集成测试：用假 callLLM/executeSql 驱动真实 runQuestionBenchmark
// 用法: node test-agentloop-v1.mjs <sql_benchmark-v20261028.html>
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(process.argv[2], 'utf8');
const between = (a, b) => {
  const i = html.indexOf(a), j = html.indexOf(b, i);
  if (i < 0 || j < 0) throw Error('锚点缺失: ' + a);
  return html.slice(i, j);
};
const questions = html.match(/const questionsList = (.*);/)[0];

let fakeExecute = null;
const ctx = vm.createContext({
  console,
  dbSchema: {},
  setTimeout,
  clearTimeout,
  AbortController,
  executeSql: async sql => fakeExecute(sql),
});
vm.runInContext(
  questions + '\n' +
  between('function formatSchema(schema)', 'async function executeSql(sql)') + '\n' +
  between('function extractThinking(content)', 'async function saveTraceToServer(') +
  '\nglobalThis.core={questionsList,verifyResults,isCorrect,selectAnswerQuery,runQuestionBenchmark};',
  ctx, { timeout: 1000 });
const core = ctx.core;

let seq = 0;
function installFakeLLM(script) {
  ctx.callLLM = async (endpoint, apiKey, model, systemPrompt, messages, maxTokens, temperature, cacheSalt, signal, onCallAttempt) => {
    if (onCallAttempt) onCallAttempt();
    const next = script.shift();
    if (!next) throw Error('脚本用尽');
    return {
      message: { content: next.content || '', reasoning_content: null, tool_calls: [{ id: 'call_' + (++seq), type: 'function', function: { name: next.name, arguments: JSON.stringify(next.args || {}) } }] },
      usage: { inputTokens: 1, outputTokens: 1 },
      cacheSaltApplied: false,
    };
  };
}

const q = core.questionsList.find(item => item.id === 2);
const answerRow = Object.fromEntries(q.columns.map(c => [c, String(q.first_row[c])]));
const answerResult = { columns: q.columns.slice(), rows: [answerRow, Object.fromEntries(q.columns.map(c => [c, '0']))], numRows: q.row_count };
const wrongAnswerResult = { ...answerResult, rows: [{ ...answerRow, total_revenue: '1' }, answerResult.rows[1]] };
const probeResult = {
  columns: ['Order Quantity', 'Unit Price', 'Extended Amount', 'Product Standard Cost', 'Total Product Cost', 'Sales Amount', 'Unit Price Discount Pct'],
  rows: Array.from({ length: 10 }, (_, i) => ({ 'Order Quantity': String(i + 1), 'Unit Price': '1', 'Extended Amount': '1', 'Product Standard Cost': '1', 'Total Product Cost': '1', 'Sales Amount': '1', 'Unit Price Discount Pct': '0' })),
  numRows: 10,
};
const results = { ANSWER_SQL: answerResult, WRONG_SQL: wrongAnswerResult, PROBE_SQL: probeResult };
fakeExecute = sql => { const r = results[sql]; if (!r) throw Error('未知 SQL: ' + sql); return r; };

let n = 0;
const ok = (cond, msg) => { if (!cond) throw Error('FAIL: ' + msg); n++; console.log('  ok  ' + msg); };

async function runCase(name, script) {
  installFakeLLM(script.slice());
  const logs = [];
  const res = await core.runQuestionBenchmark(q, 'http://fake', '', 'fake-model', 1024, 0.1, null, 600000, 2, new AbortController().signal, (msg) => logs.push(msg), () => {}, () => {});
  console.log('\n[' + name + ']');
  return { res, logs };
}

// A 答对后探查 + results_ok 显式 sql
{
  const { res, logs } = await runCase('A 显式标记', [
    { name: 'run_sql_query', args: { sql: 'ANSWER_SQL' } },
    { name: 'run_sql_query', args: { sql: 'PROBE_SQL' } },
    { name: 'results_ok', args: { sql: 'ANSWER_SQL' } },
  ]);
  ok(res.generatedSql === 'ANSWER_SQL', 'A 回填答案 SQL（不是末发探查）');
  ok(res.queryResult === answerResult, 'A 回填答案结果');
  ok(res.answerSelection && res.answerSelection.rule === 'explicit' && res.answerSelection.step === 1, 'A 识别规则=explicit，步号=1');
  ok(core.isCorrect(core.verifyResults(q, res.queryResult)), 'A 判 Pass');
  ok(logs.some(x => x.includes('判分取第 1 步')), 'A 日志记录判分步号');
}

// B 无标记：签名回退
{
  const { res } = await runCase('B 签名回退', [
    { name: 'run_sql_query', args: { sql: 'ANSWER_SQL' } },
    { name: 'run_sql_query', args: { sql: 'PROBE_SQL' } },
    { name: 'results_ok', args: {} },
  ]);
  ok(res.generatedSql === 'ANSWER_SQL' && res.answerSelection.rule === 'shape' && res.answerSelection.step === 1, 'B 无标记时按签名选第 1 步');
  ok(core.isCorrect(core.verifyResults(q, res.queryResult)), 'B 判 Pass');
}

// C 答错后探查：仍然 Fail
{
  const { res } = await runCase('C 答错后探查', [
    { name: 'run_sql_query', args: { sql: 'WRONG_SQL' } },
    { name: 'run_sql_query', args: { sql: 'PROBE_SQL' } },
    { name: 'results_ok', args: {} },
  ]);
  ok(res.generatedSql === 'WRONG_SQL' && res.answerSelection.step === 1, 'C 选中答错的答案查询');
  ok(!core.isCorrect(core.verifyResults(q, res.queryResult)), 'C 判 Fail（不放水）');
}

// D 标记未执行过：回退签名并告警
{
  const { res, logs } = await runCase('D 标记未命中', [
    { name: 'run_sql_query', args: { sql: 'ANSWER_SQL' } },
    { name: 'results_ok', args: { sql: 'SELECT never_ran' } },
  ]);
  ok(res.answerSelection.rule === 'shape' && res.answerSelection.explicitUnmatched === true, 'D 标记未命中→shape 且 explicitUnmatched');
  ok(logs.some(x => x.includes('未在已执行查询中命中')), 'D 日志告警标记未命中');
}

// E 一条消息里多个 SQL 块：只执行 tool call，文内未执行 SQL 不参与判分
{
  const { res } = await runCase('E 文内多 SQL 块', [
    { name: 'run_sql_query', args: { sql: 'ANSWER_SQL' }, content: '```sql\nSELECT ANSWER_SQL\n```\n```sql\nSELECT NEVER_RAN\n```' },
    { name: 'results_ok', args: {} },
  ]);
  ok(res.generatedSql === 'ANSWER_SQL' && res.answerSelection.step === 1, 'E 文内未执行 SQL 不参与，选中已执行的答案');
  ok(core.isCorrect(core.verifyResults(q, res.queryResult)), 'E 判 Pass');
}

console.log('\nALL PASS (%d assertions)', n);
