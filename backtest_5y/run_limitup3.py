"""专项回测：连续三个涨停 → 第四个交易日以开盘价尝试买入 → 跌破10日线卖出。

三个维度都单独给出结论：
  1) 信号有多少、第四天能不能买到（一字涨停板买不到）
  2) 买到之后的胜率、单笔收益、持有时间、分年表现
  3) 两种成交口径：只在第四天试一次 / 买不到就顺延（最多 5 个交易日）
另外做卖出规则敏感性（跌破 5/10/20 日线）与资金约束组合的实盘化对照。
"""
import json
import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results")
NPZ = os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz")
START = int(os.environ.get("START", "20210817"))
END = int(os.environ.get("END", "20260817"))
NAME = "31_三连板后回踩10日线"
LOG = open(os.path.join(OUT, "limitup3_run.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def pct(x):
    return f"{x * 100:+.2f}%"


spec = json.load(open(os.path.join(OUT, "strategies", NAME + ".spec.json"), encoding="utf-8"))
rules, sp = spec, spec["spec"]
P("=" * 96)
P("策略：连续三个涨停，第四个交易日起以开盘价买入（买不到则视口径处理），跌破10日线卖出")
P(f"中文规则：{rules['text']}")
P(f"解析结果：买入 = {' 且 '.join(rules['buy'])}；卖出 = {' 或 '.join(rules['sell'])}；"
  f"无止盈止损、无持有期上限")
P(f"涨停判定：涨幅达板块上限（主板 9.5%↑ / 创业板科创板 19.5%↑ / 北交所 29.5%↑）且收盘价 = 最高价")
P("=" * 96)

pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
i0, i1 = pnl.col_range(START, END)
P(f"样本：{pnl.N} 只A股、{int(pnl.n_bars.sum())} 根前复权日线，回测区间 {d2s(START)} ~ {d2s(END)}"
  f"（{i1 - i0} 个交易日）")

base = {"init_cash": 10000000, "pos_pct": 5, "max_hold": 20, "fee_permil": 0.5, "slip_permil": 0.5,
        "take_profit": 0.0, "stop_loss": 0.0, "max_bars": 0, "min_bars": 1}

# ── 1) 信号与可成交性 ──
buy_m = E.group_mask(ind, sp["buy"], pnl.cf["close"].shape)
buy_m[pnl.n_bars < 30] = False
o, c, h, l, chg = (pnl.cf[k] for k in ("open", "close", "high", "low", "chg"))
with np.errstate(invalid="ignore"):
    untradable = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
in_win = np.zeros_like(buy_m)
for i in range(pnl.N):
    n = int(pnl.n_bars[i])
    if n:
        d = pnl.dates[i, :n]
        in_win[i, :n] = (d >= START) & (d <= END)
sig = buy_m & in_win
rows, cols = np.nonzero(sig)
P("")
P(f"1) 三连板信号：区间内共 {len(rows):,} 次，涉及 {len(set(rows)):,} 只标的")
nxt_ok = nxt_bad = nxt_none = 0
gap_open = []
for r, k in zip(rows, cols):
    if k + 1 >= int(pnl.n_bars[r]):
        nxt_none += 1
        continue
    if untradable[r, k + 1]:
        nxt_bad += 1
    else:
        nxt_ok += 1
        gap_open.append(o[r, k + 1] / c[r, k] - 1)
P(f"   第四个交易日能以开盘价成交：{nxt_ok:,} 次（{nxt_ok / max(1, len(rows)) * 100:.1f}%）；"
  f"一字涨停板买不到：{nxt_bad:,} 次（{nxt_bad / max(1, len(rows)) * 100:.1f}%）；无后续K线：{nxt_none}")
g = np.array(gap_open)
P(f"   能买到时第四天的开盘跳空：均值 {g.mean() * 100:+.2f}%，中位数 {np.median(g) * 100:+.2f}%，"
  f"跳空为正的占比 {(g > 0).mean() * 100:.1f}%")
byyear = Counter(int(pnl.dates[r, k]) // 10000 for r, k in zip(rows, cols))
P("   分年信号数：" + "  ".join(f"{y}年 {byyear[y]:,}" for y in sorted(byyear)))

# ── 2) 两种成交口径 ──
P("")
P("2) 两种成交口径（信号级等权，每个信号都按等额下单，不受持仓数量限制）")
variants = {}
for label, retry in (("A 只在第四天试一次，买不到作罢", 0), ("B 买不到就顺延，最多等 5 天", 5)):
    p = dict(base, retry_days=retry)
    r = E.signal_backtest(pnl, ind, sp, START, END, p)
    m = E.signal_metrics(r)
    variants[label] = (r, m)
    P(f"   【{label}】")
    P(f"     成交 {m['trades']:,} 笔（放弃 {m['missed_signals']:,} 个信号），覆盖 {m['stocks_traded']:,} 只，"
      f"平均同时在持 {m['avg_positions']:.0f} 个仓位")
    P(f"     胜率 {m['win_rate'] * 100:.2f}%   平均单笔 {pct(m['avg_ret'])}   中位数 {pct(m['median_ret'])}   "
      f"盈亏比 {m['profit_factor']:.2f}")
    P(f"     平均盈利 {pct(m['avg_win'])}   平均亏损 {pct(m['avg_loss'])}   平均持有 {m['avg_bars']:.1f} 个交易日")
    P(f"     最好一笔 {pct(m['best'])}   最差一笔 {pct(m['worst'])}")
    P(f"     等权组合 5 年 {pct(m['total_return'])}（年化 {pct(m['annual_return'])}，最大回撤 "
      f"{m['max_drawdown'] * 100:.2f}%，夏普 {m['sharpe']:.2f}）")
    P(f"     单仓复利年化 {pct(m['annual_single'])}（始终只持一个仓位、反复交易的口径）")

main_r, main_m = variants["A 只在第四天试一次，买不到作罢"]

# 逐笔交易与净值落盘
with open(os.path.join(RES, NAME + "_signal_trades.csv"), "w", encoding="utf-8") as f:
    f.write("序号,代码,名称,买入日,买入价,卖出日,卖出价,净收益率,毛收益率,持有交易日,卖出原因\n")
    for k, t in enumerate(main_r["trades"], 1):
        f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},{d2s(t['sell_date'])},"
                f"{t['sell_price']},{t['ret'] * 100:.4f}%,{t['gross_ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")
with open(os.path.join(RES, NAME + "_signal_equity.csv"), "w", encoding="utf-8") as f:
    f.write("日期,等权净值\n")
    for k, d in enumerate(main_r["dates"]):
        f.write(f"{d2s(d)},{main_r['equity_curve'][k]:.6f}\n")

# ── 3) 分年与分布 ──
P("")
P("3) 口径A 的分年表现与收益分布")
yr = defaultdict(list)
for t in main_r["trades"]:
    yr[t["buy_date"] // 10000].append(t["ret"])
P("   " + f"{'年份':<6}{'笔数':>7}{'胜率':>9}{'平均单笔':>10}{'中位数':>10}{'合计贡献':>11}")
for y in sorted(yr):
    a = np.array(yr[y])
    P("   " + f"{y:<6}{len(a):>7,}{(a > 0).mean() * 100:>8.1f}%{a.mean() * 100:>9.2f}%"
              f"{np.median(a) * 100:>9.2f}%{a.sum() * 100:>10.1f}%")
allr = np.array([t["ret"] for t in main_r["trades"]])
buckets = [(-1, -0.2), (-0.2, -0.1), (-0.1, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.1),
           (0.1, 0.2), (0.2, 0.5), (0.5, 10)]
P("   收益分布：" + "  ".join(
    f"[{int(a * 100)}%,{int(b * 100)}%) {((allr >= a) & (allr < b)).sum()}" for a, b in buckets))
P(f"   卖出原因：{dict(Counter(t['reason'] for t in main_r['trades']))}")
P(f"   持有 1 天即卖出的比例：{(np.array([t['bars'] for t in main_r['trades']]) <= 1).mean() * 100:.1f}%")

# ── 4) 卖出规则敏感性 ──
P("")
P("4) 卖出规则敏感性（其余条件不变，口径A）")
P("   " + f"{'卖出规则':<16}{'笔数':>8}{'胜率':>9}{'平均单笔':>10}{'平均持有':>10}{'等权5年':>11}{'回撤':>9}")
for n in (5, 10, 20):
    sp2 = json.loads(json.dumps(sp))
    sp2["sell"]["conds"] = [{"kind": "cmp", "a": {"ind": "close"}, "op": "<", "b": {"ind": "ma", "n": n}}]
    r2 = E.signal_backtest(pnl, ind, sp2, START, END, dict(base, retry_days=0))
    m2 = E.signal_metrics(r2)
    P("   " + f"{'跌破' + str(n) + '日线':<16}{m2['trades']:>8,}{m2['win_rate'] * 100:>8.2f}%"
              f"{m2['avg_ret'] * 100:>9.2f}%{m2['avg_bars']:>9.1f}日{m2['total_return'] * 100:>10.2f}%"
              f"{m2['max_drawdown'] * 100:>8.1f}%")
for extra, lbl in ((0.08, "止损8%"), (0.15, "止损15%")):
    sp2 = json.loads(json.dumps(sp))
    p2 = dict(base, retry_days=0, stop_loss=extra)
    r2 = E.signal_backtest(pnl, ind, sp2, START, END, p2)
    m2 = E.signal_metrics(r2)
    P("   " + f"{'跌破10日线+' + lbl:<14}{m2['trades']:>8,}{m2['win_rate'] * 100:>8.2f}%"
              f"{m2['avg_ret'] * 100:>9.2f}%{m2['avg_bars']:>9.1f}日{m2['total_return'] * 100:>10.2f}%"
              f"{m2['max_drawdown'] * 100:>8.1f}%")

# ── 4b) 分板块 ──
P("")
P("4b) 分板块（涨停幅度上限不同，波动天差地别）")
codes = np.array([str(x) for x in pnl.codes])
groups = {
    "沪深主板（10%涨停）": np.array([not (c[:2] in ("30", "68")) for c in codes]),
    "创业板+科创板（20%涨停）": np.array([c[:2] in ("30", "68") for c in codes]),
}
P("   " + f"{'标的池':<22}{'笔数':>8}{'胜率':>9}{'平均单笔':>10}{'中位数':>10}{'平均持有':>10}{'等权5年':>11}")
for lbl, mask in groups.items():
    r3 = E.signal_backtest(pnl, ind, sp, START, END, dict(base, retry_days=0), universe=mask)
    m3 = E.signal_metrics(r3)
    P("   " + f"{lbl:<20}{m3['trades']:>8,}{m3['win_rate'] * 100:>8.2f}%{m3['avg_ret'] * 100:>9.2f}%"
              f"{m3['median_ret'] * 100:>9.2f}%{m3['avg_bars']:>9.1f}日{m3['total_return'] * 100:>10.2f}%")

# ── 4c) 买入时价格离 MA10 有多远 ──
ma10 = ind.get("ma10")
dist = []
for t2 in main_r["trades"]:
    i = pnl.codes.index(t2["code"])
    n = int(pnl.n_bars[i])
    d = pnl.dates[i, :n]
    k = int(np.searchsorted(d, t2["buy_date"]))
    if k < n and ma10[i, k] == ma10[i, k]:
        dist.append(pnl.cf["close"][i, k] / ma10[i, k] - 1)
dist = np.array(dist)
P("")
P(f"4c) 买入当日收盘价高于 MA10 的幅度：中位数 {np.median(dist) * 100:.1f}%，"
  f"均值 {dist.mean() * 100:.1f}%，四分位 {np.percentile(dist, 25) * 100:.1f}% ~ "
  f"{np.percentile(dist, 75) * 100:.1f}%")
P(f"    也就是说「跌破10日线」这个出场条件，在建仓那一刻就已经在脚下约 {np.median(dist) * 100:.0f}%，")
P("    等它触发时通常已经把涨幅回吐干净，这是单笔中位数亏两位数的直接原因。")

# ── 5) 资金约束组合 ──
P("")
P("5) 资金约束组合（1000 万本金 / 单笔 5% / 最多同时持 20 只，次日开盘成交）")
for pick, lbl in (("amount", "候选按成交额优先"), ("random", "候选随机挑选")):
    rp = E.backtest(pnl, ind, sp, START, END, dict(base), pick=pick)
    mp = E.metrics(rp)
    P(f"   {lbl}：成交 {mp['trades']:,} 笔，胜率 {mp['win_rate'] * 100:.2f}%，"
      f"5 年总收益 {pct(mp['total_return'])}，年化 {pct(mp['annual_return'])}，"
      f"最大回撤 {mp['max_drawdown'] * 100:.2f}%，夏普 {mp['sharpe']:.2f}")
    if pick == "amount":
        with open(os.path.join(RES, NAME + "_port_trades.csv"), "w", encoding="utf-8") as f:
            f.write("序号,代码,名称,买入日,买入价,卖出日,卖出价,股数,净收益率,盈亏(元),持有交易日,卖出原因\n")
            for k, t in enumerate(rp["trades"], 1):
                f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},"
                        f"{d2s(t['sell_date'])},{t['sell_price']},{t['shares']},{t['ret'] * 100:.4f}%,"
                        f"{t['pnl']:.2f},{t['bars']},{t['reason']}\n")
        with open(os.path.join(RES, NAME + "_port_equity.csv"), "w", encoding="utf-8") as f:
            f.write("日期,权益(元)\n")
            for d, v in rp["equity"]:
                f.write(f"{d2s(d)},{v}\n")
        port = mp

# ── 6) 样例交易 ──
P("")
P("6) 口径A 收益最高与最低的各 5 笔")
ts = sorted(main_r["trades"], key=lambda t: -t["ret"])
for t in ts[:5] + ts[-5:]:
    P(f"   {t['code']} {t['name']:<8} {d2s(t['buy_date'])} 开盘 {t['buy_price']:>8.3f} → "
      f"{d2s(t['sell_date'])} {t['sell_price']:>8.3f}  {pct(t['ret']):>9}  持有 {t['bars']:>3} 日")

bm = json.load(open(os.path.join(OUT, "benchmark.json"), encoding="utf-8"))
P("")
P(f"基准：全市场等权买入持有 5 年 {pct(bm['equal_weight_mean'])}（中位数 {pct(bm['median'])}）")
json.dump({"rules": rules, "signals": int(len(rows)), "buyable": nxt_ok, "unbuyable": nxt_bad,
           "variants": {k: v[1] for k, v in variants.items()}, "portfolio": port,
           "benchmark": bm},
          open(os.path.join(OUT, "limitup3_summary.json"), "w"), ensure_ascii=False, indent=2)
LOG.close()
