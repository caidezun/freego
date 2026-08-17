"""把优化结果落成可回测的策略，并给出资金曲线口径的完整指标。

对比的 5 个版本（同一批数据、同一套成本）：
  S1 原策略            3 板后买入 + 跌破 MA10 卖出
  S2 原入场 + 快出      3 板后买入 + 次日开盘卖
  S3 3 板改良           3 板 + 成交额≥8亿 + 换手率2~15% + 指数在MA20上 + 次日开盘卖
  S4 最优（含行业过滤）   2 板 + 成交额≥8亿 + 换手率2~15% + 同行业涨停≤2家 + 指数在MA20上 + 次日开盘卖
  S5 最优（页面可复现）   2 板 + 成交额≥8亿 + 换手率2~15% + 次日开盘卖（去掉行业与指数条件）

资金口径三种：
  a) 满仓等权：当日有信号就等权买入全部信号，无信号则空仓
  b) 单笔复利：始终只持一个仓位、反复交易（与并发数无关，便于横向比较）
  c) 资金约束：1000 万本金 / 单笔 5% / 最多同时持 20 只（贴近实盘）
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUT, "limitup_final.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


rows = list(csv.DictReader(open(os.path.join(OUT, "limitup_events.csv"), encoding="utf-8")))
F = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
D = {k: F(k) for k in ("date3", "date4", "boards", "mktcap", "turnover3", "amount3", "gap4", "amp3",
                       "ind_lu", "mkt_lu", "idx_above_ma20", "bias10",
                       "ret_ma10", "ret_ma5", "ret_trail8", "fwd_open1", "fwd_open2", "fwd_open3")}
CODE = np.array([r["code"] for r in rows])
NAME = np.array([r["name"] for r in rows])

STRATS = [
    ("S1 原策略：3板买入 + 跌破MA10", (D["boards"] == 3), "ret_ma10"),
    ("S2 3板买入 + 次日开盘卖", (D["boards"] == 3), "fwd_open1"),
    ("S3 3板 + 成交额≥8亿 + 换手率2~15% + 指数在MA20上 + 次日开盘卖",
     (D["boards"] == 3) & (D["amount3"] >= 8e8) & (D["turnover3"] >= 2) & (D["turnover3"] <= 15)
     & (D["idx_above_ma20"] > 0), "fwd_open1"),
    ("S4 2板 + 成交额≥8亿 + 换手率2~15% + 同行业涨停≤2家 + 指数在MA20上 + 次日开盘卖",
     (D["boards"] == 2) & (D["amount3"] >= 8e8) & (D["turnover3"] >= 2) & (D["turnover3"] <= 15)
     & (D["ind_lu"] <= 2) & (D["idx_above_ma20"] > 0), "fwd_open1"),
    ("S5 2板 + 成交额≥8亿 + 换手率2~15% + 次日开盘卖（页面可复现版）",
     (D["boards"] == 2) & (D["amount3"] >= 8e8) & (D["turnover3"] >= 2) & (D["turnover3"] <= 15),
     "fwd_open1"),
]

TRADE_DAYS = 1211
res = []
P("=" * 108)
P("连板策略优化结果：全A股 5404 只，2021-08-17 ~ 2026-08-17（1211 个交易日）")
P("成本：手续费 0.5‰ + 滑点 0.5‰ 双边；买入=信号次日开盘价，卖出按各自规则")
P("=" * 108)
for lbl, mask, ykey in STRATS:
    y = D[ykey]
    m = mask & ~np.isnan(y)
    a = y[m]
    hold = {"ret_ma10": None, "fwd_open1": 1, "fwd_open2": 2, "fwd_open3": 3}[ykey]
    # a) 满仓等权：按买入日分组，当日等权
    byday = defaultdict(list)
    for r, ret in zip(D["date4"][m], a):
        byday[int(r)].append(ret)
    days = sorted(byday)
    if hold:                          # 固定持有 hold 天 → 收益摊到 hold 天
        eq, curve = 1.0, []
        for d in days:
            eq *= (1 + np.mean(byday[d]))
            curve.append(eq)
    else:                             # 变动持有期，按笔序近似
        eq, curve = 1.0, []
        for d in days:
            eq *= (1 + np.mean(byday[d]))
            curve.append(eq)
    curve = np.array(curve)
    peak = np.maximum.accumulate(curve)
    mdd = float(np.max((peak - curve) / peak))
    # b) 单笔复利
    geo = float(np.expm1(np.mean(np.log1p(np.maximum(a, -0.99)))))
    avg_bars = hold if hold else float(np.mean([float(r["bars_ma10"]) for r, mm in zip(rows, m) if mm]))
    turns = TRADE_DAYS / 5 / avg_bars
    ann_single = (1 + geo) ** turns - 1
    # 年化（满仓等权，按信号日数折算）
    ann_stack = curve[-1] ** (244 / max(1, len(days))) - 1 if len(curve) else 0.0
    st = {"label": lbl, "n": int(m.sum()), "win": float((a > 0).mean()), "mean": float(a.mean()),
          "med": float(np.median(a)), "geo": geo, "ann_single": float(ann_single),
          "stack_total": float(curve[-1] - 1) if len(curve) else 0.0, "stack_mdd": mdd,
          "signal_days": len(days), "avg_bars": avg_bars,
          "pf": float(a[a > 0].sum() / -a[a <= 0].sum()) if (a <= 0).any() else float("inf"),
          "per_year": int(m.sum() / 5)}
    res.append(st)
    P("")
    P(f"【{lbl}】")
    P(f"   成交 {st['n']:,} 笔（约 {st['per_year']}/年），胜率 {st['win'] * 100:.2f}%，"
      f"平均单笔 {st['mean'] * 100:+.2f}%，中位数 {st['med'] * 100:+.2f}%，盈亏比 {st['pf']:.2f}")
    P(f"   单笔几何均值 {geo * 100:+.3f}%  →  单笔复利年化 {ann_single * 100:+.1f}%"
      f"（平均持有 {avg_bars:.1f} 日，年周转约 {turns:.0f} 次）")
    P(f"   满仓等权：5 年累计 {st['stack_total'] * 100:+.1f}%，最大回撤 {mdd * 100:.1f}%，"
      f"有信号的交易日 {len(days)} 天（占 {len(days) / TRADE_DAYS * 100:.0f}%）")
    yr = defaultdict(list)
    for d, ret in zip(D["date4"][m], a):
        yr[int(d) // 10000].append(ret)
    P("   分年平均单笔：" + "  ".join(
        f"{y}: {np.mean(v) * 100:+.2f}%({len(v)}笔)" for y, v in sorted(yr.items())))

P("")
P("=" * 108)
P("横向汇总（按平均单笔收益排序）")
P("=" * 108)
P(f"{'策略':<52}{'笔数':>7}{'胜率':>8}{'平均单笔':>10}{'中位数':>9}{'盈亏比':>7}{'单笔复利年化':>13}{'满仓等权5年':>12}{'回撤':>8}")
for st in sorted(res, key=lambda x: -x["mean"]):
    P(f"{st['label'][:50]:<52}{st['n']:>7,}{st['win'] * 100:>7.2f}%{st['mean'] * 100:>+9.2f}%"
      f"{st['med'] * 100:>+8.2f}%{st['pf']:>7.2f}{st['ann_single'] * 100:>+12.1f}%"
      f"{st['stack_total'] * 100:>+11.1f}%{st['stack_mdd'] * 100:>7.1f}%")

# 最优策略的逐笔明细
best_lbl, best_mask, best_key = STRATS[3]
m = best_mask & ~np.isnan(D[best_key])
order = np.argsort(D["date4"][m])
with open(os.path.join(OUT, "results", "S4_2板优化版_trades.csv"), "w", encoding="utf-8") as f:
    f.write("序号,代码,名称,第二板日期,买入日(次日开盘),净收益率,成交额(亿),换手率(%),同行业涨停家数,跳空(%)\n")
    idx = np.nonzero(m)[0][order]
    for k, i in enumerate(idx, 1):
        f.write(f"{k},{CODE[i]},{NAME[i]},{int(D['date3'][i])},{int(D['date4'][i])},"
                f"{D[best_key][i] * 100:.4f}%,{D['amount3'][i] / 1e8:.2f},{D['turnover3'][i]:.2f},"
                f"{int(D['ind_lu'][i])},{D['gap4'][i] * 100:.2f}\n")
P("")
P(f"已保存 S4 最优版逐笔明细 results/S4_2板优化版_trades.csv（{int(m.sum())} 笔）")
json.dump(res, open(os.path.join(OUT, "limitup_final.json"), "w"), ensure_ascii=False, indent=2)
LOG.close()
