"""专项回测：三连板后第四日开盘试买，买入后涨停继续持有、收盘不涨停就卖。

规则原话：连续三个涨停，第四个交易日开始尝试买入，以开盘价买入（如果能买入的话），
买入后第二天收盘不涨停就卖，如果涨停就继续持有，只要收盘不涨停就卖。

中文规则（页面可编译）：当近3个交易日有3个涨停时买入；当收盘不涨停时卖出。
撮合：收盘信号 → 次日开盘成交；T+1；一字板不成交。另给「开板当日收盘卖」作对照。
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
os.makedirs(RES, exist_ok=True)
NPZ = os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz")
START = int(os.environ.get("START", "20210817"))
END = int(os.environ.get("END", "20260817"))
NAME = "32_三连板后持有到开板"
LOG = open(os.path.join(OUT, "limitup32_run.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def pct(x):
    return f"{x * 100:+.2f}%"


spec_path = os.path.join(OUT, "strategies", NAME + ".spec.json")
spec = json.load(open(spec_path, encoding="utf-8"))
rules, sp = spec, spec["spec"]
P("=" * 96)
P("策略：连续三个涨停，第四个交易日起以开盘价买入；买入后涨停继续持有，收盘不涨停则卖")
P(f"中文规则：{rules['text']}")
P(f"解析结果：买入 = {' 且 '.join(rules['buy'])}；卖出 = {' 或 '.join(rules['sell'])}；"
  f"无止盈止损、无持有期上限")
P("涨停判定：涨幅达板块上限（主板 9.5%↑ / 创业板科创板 19.5%↑ / 北交所 29.5%↑）且收盘价 = 最高价")
P("=" * 96)

pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
i0, i1 = pnl.col_range(START, END)
P(f"样本：{pnl.N} 只A股、{int(pnl.n_bars.sum())} 根前复权日线，回测区间 {d2s(START)} ~ {d2s(END)}"
  f"（{i1 - i0} 个交易日）")

base = {"init_cash": 10000000, "pos_pct": 5, "max_hold": 20, "fee_permil": 0.5, "slip_permil": 0.5,
        "take_profit": 0.0, "stop_loss": 0.0, "max_bars": 0, "min_bars": 1}

o, c, h, l, chg = (pnl.cf[k] for k in ("open", "close", "high", "low", "chg"))
pct_lim = E.limit_pct_of(pnl.codes)
with np.errstate(invalid="ignore"):
    lu = (chg >= pct_lim[:, None]) & (c >= h - 1e-6)
    lu = np.where(np.isnan(chg), False, lu)
    untradable = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)

# ── 1) 信号与可成交性（与策略 31 同一买入条件）──
buy_m = E.group_mask(ind, sp["buy"], pnl.cf["close"].shape)
buy_m[pnl.n_bars < 30] = False
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
  f"一字板买不到：{nxt_bad:,} 次（{nxt_bad / max(1, len(rows)) * 100:.1f}%）；无后续K线：{nxt_none}")
g = np.array(gap_open)
P(f"   能买到时次日开盘跳空：均值 {g.mean() * 100:+.2f}%，中位数 {np.median(g) * 100:+.2f}%，"
  f"跳空为正的占比 {(g > 0).mean() * 100:.1f}%")
byyear = Counter(int(pnl.dates[r, k]) // 10000 for r, k in zip(rows, cols))
P("   分年信号数：" + "  ".join(f"{y}年 {byyear[y]:,}" for y in sorted(byyear)))

# 独立三连板事件（streak 第一次等于 3）
streak = np.zeros(lu.shape, dtype=np.int16)
for j in range(lu.shape[1]):
    streak[:, j] = np.where(lu[:, j], (streak[:, j - 1] if j else 0) + 1, 0)
indep = 0
len_dist = Counter()
for i in range(pnl.N):
    n = int(pnl.n_bars[i])
    for k in range(n):
        if streak[i, k] != 3:
            continue
        d = int(pnl.dates[i, k])
        if not (START <= d <= END):
            continue
        indep += 1
        s = 3
        kk = k + 1
        while kk < n and streak[i, kk] == s + 1:
            s += 1
            kk += 1
        len_dist[s] += 1
P(f"   独立三连板事件：{indep:,} 次；连板长度 "
  + "、".join(f"{k}板 {len_dist[k]}" for k in sorted(len_dist)[:12])
  + (f"…最长 {max(len_dist)}" if len_dist else ""))

# ── 2) 引擎口径 ──
P("")
P("2) 信号级等权（每个信号等额下单；卖出=收盘不涨停的次日开盘，T+1）")
variants = {}
for label, retry in (("A 只在信号次日试一次，买不到作罢", 0), ("B 买不到就顺延，最多等 5 天", 5)):
    r = E.signal_backtest(pnl, ind, sp, START, END, dict(base, retry_days=retry))
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
    P(f"     单仓复利年化 {pct(m['annual_single'])}")

main_r, main_m = variants["A 只在信号次日试一次，买不到作罢"]

with open(os.path.join(RES, NAME + "_signal_trades.csv"), "w", encoding="utf-8") as f:
    f.write("序号,代码,名称,买入日,买入价,卖出日,卖出价,净收益率,毛收益率,持有交易日,卖出原因\n")
    for k, t in enumerate(main_r["trades"], 1):
        f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},{d2s(t['sell_date'])},"
                f"{t['sell_price']},{t['ret'] * 100:.4f}%,{t['gross_ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")
with open(os.path.join(RES, NAME + "_signal_equity.csv"), "w", encoding="utf-8") as f:
    f.write("日期,等权净值\n")
    for k, d in enumerate(main_r["dates"]):
        f.write(f"{d2s(d)},{main_r['equity_curve'][k]:.6f}\n")

# ── 3) 分年、分布、持有期 ──
P("")
P("3) 口径A 的分年表现、收益分布、持有期")
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
bars = np.array([t["bars"] for t in main_r["trades"]])
P(f"   持有天数：均值 {bars.mean():.2f}，中位数 {np.median(bars):.0f}，"
  f"1日 {(bars <= 1).mean() * 100:.1f}%  2日 {(bars == 2).mean() * 100:.1f}%  "
  f">=5日 {(bars >= 5).mean() * 100:.1f}%  最长 {int(bars.max())} 日")

# 买入后还能吃到几根涨停
code_i = {c: i for i, c in enumerate(pnl.codes)}
extra_lu = []
buy_day_lu = 0
for t in main_r["trades"]:
    i = code_i[t["code"]]
    n = int(pnl.n_bars[i])
    ds = pnl.dates[i, :n]
    e = int(np.searchsorted(ds, t["buy_date"]))
    xi = int(np.searchsorted(ds, t["sell_date"]))
    if e < n and lu[i, e]:
        buy_day_lu += 1
    cnt = 0
    for b in range(e, min(xi, n)):
        if lu[i, b]:
            cnt += 1
        else:
            break
    extra_lu.append(cnt)
extra_lu = np.array(extra_lu)
P(f"   买入当日收盘仍涨停（第4板及以后）：{buy_day_lu:,} / {len(main_r['trades']):,} "
  f"（{buy_day_lu / max(1, len(main_r['trades'])) * 100:.1f}%）")
P("   持有期间连续涨停根数（含买入当日）："
  + "  ".join(f"{k}根 {(extra_lu == k).sum()}" for k in range(0, 8))
  + f"  ≥8根 {(extra_lu >= 8).sum()}")

# ── 4) 开板当日收盘卖 vs 次日开盘卖 ──
P("")
P("4) 出场价对照：开板当日收盘卖（更贴近「收盘不涨停就卖」字面） vs 次日开盘卖（引擎默认）")
FEE = SLIP = 0.5 / 1000
ret_close, ret_open, hold_n, extra_n = [], [], [], []
# 买入信号与引擎相同（近3日涨停次数==3），区别只是出场价用开板收盘还是次日开盘
for i in range(pnl.N):
    n = int(pnl.n_bars[i])
    if n < 30:
        continue
    j = 0
    while j < n - 2:
        d = int(pnl.dates[i, j])
        if not (START <= d <= END) or not buy_m[i, j]:
            j += 1
            continue
        e = j + 1
        if e >= n or untradable[i, e]:
            j += 1
            continue
        if not (START <= int(pnl.dates[i, e]) <= END):
            j += 1
            continue
        entry = o[i, e] * (1 + SLIP)
        k = e + 1  # T+1 起检查
        while k < n - 1 and lu[i, k]:
            k += 1
        if k >= n:
            j = n
            continue
        n_lu = int(lu[i, e:k].sum()) if k > e else 0
        extra_n.append(n_lu)
        hold_n.append(k - e)
        px_c = c[i, k] * (1 - SLIP)
        ret_close.append(px_c / entry - 1 - 2 * FEE)
        if k + 1 < n and not untradable[i, k + 1]:
            px_o = o[i, k + 1] * (1 - SLIP)
            ret_open.append(px_o / entry - 1 - 2 * FEE)
        else:
            ret_open.append(c[i, k] / entry - 1 - 2 * FEE)
        j = k  # 不重叠
ret_close, ret_open = np.array(ret_close), np.array(ret_open)
hold_n, extra_n = np.array(hold_n), np.array(extra_n)
for lbl, a in (("开板当日收盘卖", ret_close), ("开板次日开盘卖", ret_open)):
    P(f"   {lbl:<12} {len(a):>6,} 笔  胜率 {(a > 0).mean() * 100:5.1f}%  "
      f"平均 {a.mean() * 100:+6.2f}%  中位数 {np.median(a) * 100:+6.2f}%  "
      f"盈亏比 {a[a > 0].sum() / -a[a <= 0].sum() if (a <= 0).any() else float('inf'):.2f}")
P(f"   事件口径平均持有 {hold_n.mean():.2f} 日，买入后还能吃到的涨停根数均值 {extra_n.mean():.2f}")

# ── 5) 对照：同一买入、跌破 MA10 卖（策略 31）──
P("")
P("5) 同一买入条件、不同卖出：开板就走 vs 跌破 MA10")
sp31 = json.load(open(os.path.join(OUT, "strategies", "31_三连板后回踩10日线.spec.json"),
                      encoding="utf-8"))["spec"]
r31 = E.signal_backtest(pnl, ind, sp31, START, END, dict(base, retry_days=0))
m31 = E.signal_metrics(r31)
P(f"   开板就走（本策略）  {main_m['trades']:>6,} 笔  胜率 {main_m['win_rate'] * 100:5.2f}%  "
  f"平均 {main_m['avg_ret'] * 100:+6.2f}%  中位数 {main_m['median_ret'] * 100:+6.2f}%  "
  f"持有 {main_m['avg_bars']:.1f} 日")
P(f"   跌破MA10（策略31） {m31['trades']:>6,} 笔  胜率 {m31['win_rate'] * 100:5.2f}%  "
  f"平均 {m31['avg_ret'] * 100:+6.2f}%  中位数 {m31['median_ret'] * 100:+6.2f}%  "
  f"持有 {m31['avg_bars']:.1f} 日")

# ── 6) 分板块 ──
P("")
P("6) 分板块")
codes = np.array([str(x) for x in pnl.codes])
groups = {
    "沪深主板（10%涨停）": np.array([not (c[:2] in ("30", "68")) for c in codes]),
    "创业板+科创板（20%涨停）": np.array([c[:2] in ("30", "68") for c in codes]),
}
P("   " + f"{'标的池':<22}{'笔数':>8}{'胜率':>9}{'平均单笔':>10}{'中位数':>10}{'平均持有':>10}")
for lbl, mask in groups.items():
    r3 = E.signal_backtest(pnl, ind, sp, START, END, dict(base, retry_days=0), universe=mask)
    m3 = E.signal_metrics(r3)
    P("   " + f"{lbl:<20}{m3['trades']:>8,}{m3['win_rate'] * 100:>8.2f}%{m3['avg_ret'] * 100:>9.2f}%"
              f"{m3['median_ret'] * 100:>9.2f}%{m3['avg_bars']:>9.1f}日")

# ── 7) 资金约束 ──
P("")
P("7) 资金约束组合（1000 万本金 / 单笔 5% / 最多同时持 20 只）")
port = None
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

# ── 8) 样例 ──
P("")
P("8) 口径A 收益最高与最低的各 5 笔")
ts = sorted(main_r["trades"], key=lambda t: -t["ret"])
for t in ts[:5] + ts[-5:]:
    P(f"   {t['code']} {t['name']:<8} {d2s(t['buy_date'])} 开盘 {t['buy_price']:>8.3f} → "
      f"{d2s(t['sell_date'])} {t['sell_price']:>8.3f}  {pct(t['ret']):>9}  持有 {t['bars']:>3} 日")

bm = json.load(open(os.path.join(OUT, "benchmark.json"), encoding="utf-8"))
P("")
P(f"基准：全市场等权买入持有 5 年 {pct(bm['equal_weight_mean'])}（中位数 {pct(bm['median'])}）")
json.dump({
    "rules": rules,
    "signals": int(len(rows)), "buyable": nxt_ok, "unbuyable": nxt_bad, "independent": indep,
    "variants": {k: v[1] for k, v in variants.items()},
    "close_exit": {"n": int(len(ret_close)), "win": float((ret_close > 0).mean()),
                   "mean": float(ret_close.mean()), "median": float(np.median(ret_close))},
    "open_exit_event": {"n": int(len(ret_open)), "win": float((ret_open > 0).mean()),
                        "mean": float(ret_open.mean()), "median": float(np.median(ret_open))},
    "vs31": m31, "portfolio": port, "benchmark": bm,
    "buy_day_still_lu": int(buy_day_lu),
    "extra_lu_mean": float(extra_lu.mean()) if len(extra_lu) else 0,
}, open(os.path.join(OUT, "limitup32_summary.json"), "w"), ensure_ascii=False, indent=2)
LOG.close()
print("已写入", os.path.join(OUT, "limitup32_run.log"))
