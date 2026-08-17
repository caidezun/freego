/* 校验页面内置 JS 引擎与它生成的 Python 代码：同一批标的、同一区间，30 个策略逐笔比对 */
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { Strategy, JsEngine, loadCsv, i2d, d2i } from './appmods.mjs';
import { STRATEGIES } from './strategies30.mjs';

const CSVDIR = '/home/ubuntu/ashare5y/csv';
const WORK = '/tmp/harness/jspy';
fs.mkdirSync(WORK, { recursive: true });
const all = fs.readdirSync(CSVDIR).filter(f => f.endsWith('.csv'));
const step = Math.floor(all.length / 24);
const pick = Array.from({ length: 24 }, (_, i) => all[i * step]);

const dataset = pick.map(f => {
  const [code, name] = f.replace('.csv', '').split('_');
  return loadCsv(path.join(CSVDIR, f), code, name);
});
/* 关键：喂给 Python 的 CSV 必须由「页面内存里的序列」导出，而不是直接拷原始 CSV。
   页面把价格存成 Float32Array 以压缩体积，若 Python 读的是原始十进制字符串，
   两边的 MA 会在 1e-7 量级上不同，遇到 MA 完全相等的临界情形就会让严格比较翻转。
   页面自身的 JS 引擎与 Pyodide 消费的都是同一份 float32 数据，因此不存在这个问题。 */
const DD = path.join(WORK, 'data');
fs.rmSync(DD, { recursive: true, force: true });
fs.mkdirSync(DD, { recursive: true });
for (const s of dataset) {
  const out = ['date,open,high,low,close,volume,amount,amp,chg,turnover'];
  for (let i = 0; i < s.n; i++) {
    out.push([i2d(s.d[i]), s.o[i], s.h[i], s.l[i], s.c[i], s.v[i], s.a[i], s.amp[i], s.chg[i], s.tr[i]]
      .map(v => typeof v === 'number' ? String(v) : v).join(','));
  }
  fs.writeFileSync(path.join(DD, `${s.code}_${s.name}.csv`), out.join('\n') + '\n');
}
console.log(`子集 ${dataset.length} 只，共 ${dataset.reduce((s, x) => s + x.n, 0)} 根K线`);

const ctx = { start: '2021-08-17', end: '2026-08-17', cash: 10000000, posPct: 5, maxPos: 20, fee: 0.5, slip: 0.5 };
fs.writeFileSync(path.join(WORK, 'driver.py'), `import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("s", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ds = m.load_dataset_from_csv("${WORK}/data")
r = m.run(ds, "${ctx.start}", "${ctx.end}", {"init_cash": ${ctx.cash}, "pos_pct": ${ctx.posPct},
    "max_hold": ${ctx.maxPos}, "fee_permil": ${ctx.fee}, "slip_permil": ${ctx.slip}})
print(json.dumps(r["trades"], ensure_ascii=False))
`);

/* 逐笔比对：代码/日期/股数/卖出原因必须完全相同；价格与收益率允许展示层舍入误差
   （JS 的 toFixed 逢半进位、Python 的 round 用银行家进位，会在最后一位差 1） */
const sameTrade = (x, y) => x && y && x.code === y.code && x.buy_date === y.buy_date &&
  x.sell_date === y.sell_date && x.shares === y.shares && x.reason === y.reason &&
  Math.abs(x.buy_price - y.buy_price) <= 0.0011 && Math.abs(x.sell_price - y.sell_price) <= 0.0011 &&
  Math.abs(x.ret - y.ret) <= 2e-5;
const show = t => t ? `${t.code} ${t.buy_date}@${t.buy_price} → ${t.sell_date}@${t.sell_price} ${t.shares}股 ${t.reason} ${t.ret}` : '—';
let bad = 0;
for (const st of STRATEGIES) {
  const spec = Strategy.compile(st.text);
  const py = path.join('/workspace/backtest_5y/strategies',
    `${st.id}_${st.name.replace(/[\/\\:*?"<>|\s]/g, '')}.py`);
  const jsR = JsEngine.run(dataset.map(s => ({ ...s })), spec, ctx);
  let pyT;
  try {
    pyT = JSON.parse(execFileSync('python3', [path.join(WORK, 'driver.py'), py],
      { encoding: 'utf8', maxBuffer: 1 << 28 }));
  } catch (e) { console.log(`✗ ${st.id} ${st.name} Python 执行失败: ${(e.stderr || '').slice(0, 300)}`); bad++; continue; }
  const a = jsR.trades, b = pyT;
  let i = -1;
  for (let k = 0; k < Math.max(a.length, b.length); k++) if (!sameTrade(a[k], b[k])) { i = k; break; }
  const same = a.length === b.length && i < 0;
  if (!same) {
    bad++;
    console.log(`✗ ${st.id} ${st.name}  JS ${a.length} 笔 / PY ${b.length} 笔`);
    console.log(`   首个差异 #${i}\n     JS ${show(a[i])}\n     PY ${show(b[i])}`);
  } else {
    console.log(`✓ ${st.id} ${st.name}  ${a.length} 笔一致`);
  }
}
console.log(bad ? `\n${30 - bad}/30 一致，${bad} 个不一致` : '\n页面 JS 引擎与生成的 Python 代码：30 个策略全部逐笔一致');
process.exit(bad ? 1 : 0);
