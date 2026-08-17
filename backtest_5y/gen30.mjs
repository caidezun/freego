/* 用页面内置的中文策略编译器把 30 条策略翻译成 Python 回测代码，并输出解析结果供人工核对 */
import fs from 'fs';
import path from 'path';
import { Strategy, Codegen } from './appmods.mjs';
import { STRATEGIES } from './strategies30.mjs';

const OUT = process.env.OUT || '/workspace/backtest_5y';
const DIR = path.join(OUT, 'strategies');
fs.mkdirSync(DIR, { recursive: true });

const CTX = {
  start: process.env.START || '2021-08-17',
  end: process.env.END || '2026-08-17',
  cash: 10000000, posPct: 5, maxPos: 20, fee: 0.5, slip: 0.5
};

const index = [];
for (const st of STRATEGIES) {
  const spec = Strategy.compile(st.text);
  const py = Codegen.gen(spec, CTX);
  const base = `${st.id}_${st.name.replace(/[\/\\:*?"<>|\s]/g, '')}`;
  fs.writeFileSync(path.join(DIR, base + '.py'), py);
  const rec = {
    id: st.id, name: st.name, cat: st.cat, src: st.src, text: st.text,
    buy_op: spec.buy.op, sell_op: spec.sell.op,
    buy: spec.buy.conds.map(Strategy.describe),
    sell: spec.sell.conds.map(Strategy.describe),
    risk: spec.risk, warns: spec.warns, py_file: 'strategies/' + base + '.py', py_lines: py.split('\n').length
  };
  fs.writeFileSync(path.join(DIR, base + '.spec.json'), JSON.stringify({ ...rec, spec }, null, 2));
  index.push(rec);
  console.log(`${st.id} ${st.name}  (${py.split('\n').length} 行)`);
  console.log(`   买入[${spec.buy.op === 'or' ? '任一' : '全部'}] ${rec.buy.join(' ｜ ') || '（无）'}`);
  console.log(`   卖出[${spec.sell.op === 'or' ? '任一' : '全部'}] ${rec.sell.join(' ｜ ') || '（无）'}`
    + `   止盈${spec.risk.takeProfit * 100 || '-'}% 止损${spec.risk.stopLoss * 100 || '-'}% 期限${spec.risk.maxHold || '-'}日`);
  if (spec.warns.length) console.log(`   ⚠ ${spec.warns.join('；')}`);
}
fs.writeFileSync(path.join(OUT, 'strategies_index.json'), JSON.stringify({ ctx: CTX, list: index }, null, 2));
console.log(`\n已生成 ${index.length} 份策略代码到 ${DIR}`);
