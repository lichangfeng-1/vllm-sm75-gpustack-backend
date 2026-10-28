// control-upstream-v1.mjs — 负控：上游原件在同一「期望列全缺失」用例上必须给出 firstRowMatch=true
import fs from 'node:fs';
import vm from 'node:vm';
const html = fs.readFileSync(process.argv[2], 'utf8');
const between = (a, b) => { const i = html.indexOf(a), j = html.indexOf(b, i); if (i < 0 || j < 0) throw Error('锚点缺失'); return html.slice(i, j); };
const ctx = vm.createContext({ console });
vm.runInContext(html.match(/const questionsList = (.*);/)[0] + '\n' + between('function formatSchema(schema)', 'async function executeSql(sql)') + '\nglobalThis.core={questionsList,verifyResults,isCorrect};', ctx, { timeout: 1000 });
const { questionsList, verifyResults, isCorrect } = ctx.core;
const q = questionsList.find(item => item.id === 2);
const probe = {
  columns: ['Order Quantity', 'Unit Price', 'Extended Amount', 'Product Standard Cost', 'Total Product Cost', 'Sales Amount', 'Unit Price Discount Pct'],
  rows: Array.from({ length: 10 }, () => ({ 'Order Quantity': '1', 'Unit Price': '1', 'Extended Amount': '1', 'Product Standard Cost': '1', 'Total Product Cost': '1', 'Sales Amount': '1', 'Unit Price Discount Pct': '0' })),
  numRows: 10,
};
const report = verifyResults(q, probe);
if (report.firstRowMatch !== true) { console.log('负控失效：上游 firstRowMatch=' + report.firstRowMatch); process.exit(1); }
if (report.columnNamesMatch !== false) { console.log('负控失效：上游 columnNamesMatch=' + report.columnNamesMatch); process.exit(1); }
if (isCorrect(report)) { console.log('负控失效：上游整体判 Pass'); process.exit(1); }
console.log('负控成立：上游 firstRowMatch=true（假阳性），columnNamesMatch=false，整体 Fail');
