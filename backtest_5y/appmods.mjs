/* 从单文件应用里抽出核心模块，供 Node 侧直接调用（与页面执行的是同一份代码） */
import fs from 'fs';

const html = fs.readFileSync('/workspace/stock-quant-backtest.html', 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const exportLine = '\nmodule.exports={Strategy,Codegen,JsEngine,Ind,packRows,enrich,d2i,i2d,stats,Pinyin,mergeRows,unpackRows};';
const stubDoc = {
  addEventListener() { }, querySelector: () => null,
  createElement: () => ({ style: {}, remove() { } }), head: { appendChild() { } }
};
const mod = { exports: {} };
new Function('module', 'document', 'window', 'navigator', 'performance', 'indexedDB', 'ResizeObserver',
  code + exportLine)(mod, stubDoc, { addEventListener() { }, devicePixelRatio: 1 }, {}, performance, {},
    class { observe() { } });

export const { Strategy, Codegen, JsEngine, Ind, packRows, enrich, d2i, i2d, stats, Pinyin } = mod.exports;

/* 读取下载好的 CSV 为引擎可用的序列对象 */
export function loadCsv(file, code2, name) {
  const txt = fs.readFileSync(file, 'utf8').trim().split('\n');
  const rows = [];
  for (let i = 1; i < txt.length; i++) {
    const p = txt[i].split(',');
    if (p.length < 10) continue;
    // CSV: date,open,high,low,close,volume,amount,amp,chg,turnover
    // packRows 需要 [dateInt,o,c,h,l,vol,amt,amp,chg,turnover]
    rows.push([d2i(p[0]), +p[1], +p[4], +p[2], +p[3], +p[5], +p[6], +p[7], +p[8], +p[9]]);
  }
  return packRows(code2, name, code2[0] === '6' ? 1 : 0, rows);
}
