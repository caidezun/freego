"""专项回测：横盘超过 20 日 → 放量上涨 → 价>MA5>MA10>MA20 → 第一个涨停后第二个交易日开盘买 → 跌破 MA10 卖。

「第一个涨停后的第二个交易日」：涨停日为 T，T+1 为其后第一个交易日，买入价 = T+2 开盘。
引擎口径是「收盘信号 → 次日开盘成交」，因此把买入掩码打在 T+1 收盘，撮合端就是 T+2 开盘。

横盘、放量发生在涨停当日之前/当日，当前中文编译器表达不了（会错买到 T+1），故不进策略库。
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
NAME = "34_横盘放量首板T2"
LOG = open(os.path.join(OUT, "sideways_lu_run.log"), "w", encoding="utf-8")
EMPTY = {"buy": {"op": "and", "conds": []}, "sell": {"op": "and", "conds": []}}


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def pct(x):
    return f"{x * 100:+.2f}%"


def lag(x, n=1):
    if x.dtype == bool:
        y = np.zeros(x.shape, dtype=bool)
    else:
        y = np.full(x.shape, np.nan, dtype=x.dtype)
    y[:, n:] = x[:, :-n]
    return y


def window_mask(pnl):
    m = np.zeros((pnl.N, pnl.L), dtype=bool)
    for i in range(pnl.N):
        n = int(pnl.n_bars[i])
        if n:
            d = pnl.dates[i, :n]
            m[i, :n] = (d >= START) & (d <= END)
    return m


def params0(**kw):
    p = {"init_cash": 10_000_000, "pos_pct": 5, "max_hold": 20,
         "fee_permil": 0.5, "slip_permil": 0.5,
         "take_profit": 0.0, "stop_loss": 0.0, "max_bars": 0, "min_bars": 1}
    p.update(kw)
    return p


def dump_row(m):
    return {
        "trades": int(m["trades"]), "win": float(m["win_rate"]),
        "avg": float(m["avg_ret"]), "med": float(m["median_ret"]),
        "pf": float(m["profit_factor"]) if np.isfinite(m["profit_factor"]) else None,
        "avg_win": float(m["avg_win"]), "avg_loss": float(m["avg_loss"]),
        "bars": float(m["avg_bars"]), "total": float(m["total_return"]),
        "ann": float(m["annual_return"]), "mdd": float(m["max_drawdown"]),
        "sharpe": float(m["sharpe"]), "stocks": int(m["stocks_traded"]),
        "missed": int(m["missed_signals"]), "best": float(m["best"]),
        "worst": float(m["worst"]), "avg_pos": float(m["avg_positions"]),
    }


def print_m(label, m):
    P(f"   【{label}】")
    P(f"     成交 {m['trades']:,} 笔（放弃 {m['missed_signals']:,} 个信号），覆盖 {m['stocks_traded']:,} 只，"
      f"平均同时在持 {m['avg_positions']:.1f} 个仓位")
    P(f"     胜率 {m['win_rate'] * 100:.2f}%   平均单笔 {pct(m['avg_ret'])}   中位数 {pct(m['median_ret'])}   "
      f"盈亏比 {m['profit_factor']:.2f}")
    P(f"     平均盈利 {pct(m['avg_win'])}   平均亏损 {pct(m['avg_loss'])}   平均持有 {m['avg_bars']:.1f} 个交易日")
    P(f"     最好 {pct(m['best'])}   最差 {pct(m['worst'])}")
    P(f"     等权组合 5 年 {pct(m['total_return'])}（年化 {pct(m['annual_return'])}，最大回撤 "
      f"{m['max_drawdown'] * 100:.2f}%，夏普 {m['sharpe']:.2f}）")


def t2_signal(setup):
    """涨停日 T 满足 setup → 在 T+1 收盘发信号 → 引擎于 T+2 开盘买入。"""
    return lag(setup, 1)


P("=" * 100)
P("横盘>20日 + 放量上涨 + 价>MA5>MA10>MA20 + 首板后第二个交易日开盘买 + 跌破MA10卖")
P("样本窗口 2021-08-17 ~ 2026-08-17；成本 手续费 0.5‰ + 滑点 0.5‰ 双边")
P("=" * 100)

pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
i0, i1 = pnl.col_range(START, END)
N, L = pnl.N, pnl.L
o, h, l, c, v, chg = (pnl.cf[k] for k in ("open", "high", "low", "close", "volume", "chg"))
P(f"样本：{N} 只、{int(pnl.n_bars.sum())} 根日线，区间 {d2s(START)} ~ {d2s(END)}（{i1 - i0} 个交易日）")

in_win = window_mask(pnl)
ma5, ma10, ma20 = ind.get("ma5"), ind.get("ma10"), ind.get("ma20")
vma5 = ind.get("vma5")
pct_lim = E.limit_pct_of(pnl.codes)
with np.errstate(invalid="ignore"):
    lu = (chg >= pct_lim[:, None]) & (c >= h - 1e-6)
    lu = np.where(np.isnan(chg), False, lu)
    untradable = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
    ma_align = np.isfinite(c) & np.isfinite(ma5) & np.isfinite(ma10) & np.isfinite(ma20) \
        & (c > ma5) & (ma5 > ma10) & (ma10 > ma20)
    # 涨停前 20 日箱体（用 T-1 的 20 日高低点，不含涨停当日）
    hh20 = E.roll_max(h, 20)
    ll20 = E.roll_min(l, 20)
    box20 = (hh20 - ll20) / np.maximum(ll20, 1e-12)
    hhc20 = E.roll_max(c, 20)
    llc20 = E.roll_min(c, 20)
    cr20 = (hhc20 - llc20) / np.maximum((hhc20 + llc20) / 2.0, 1e-12)
    hh30 = E.roll_max(h, 30)
    ll30 = E.roll_min(l, 30)
    box30 = (hh30 - ll30) / np.maximum(ll30, 1e-12)
    vol_vs_prev = v / lag(vma5, 1)          # 相对涨停前 5 日均量（不含当日）
    vol_vs_inc = v / vma5                   # 含当日的 5 日均量（引擎默认「放量」）

lu_prev = lag(lu, 1)
first_lu = lu & ~lu_prev                    # 连板的第一根
lu20 = ind.get("lu20")
no_lu20 = lag(np.isfinite(lu20) & (lu20 == 0), 1)   # T-20..T-1 无涨停
consol15 = lag(np.isfinite(box20) & (box20 < 0.15), 1)
consol20 = lag(np.isfinite(box20) & (box20 < 0.20), 1)
consol10 = lag(np.isfinite(box20) & (box20 < 0.10), 1)
consol25 = lag(np.isfinite(box20) & (box20 < 0.25), 1)
consol_c12 = lag(np.isfinite(cr20) & (cr20 < 0.12), 1)
consol30 = lag(np.isfinite(box30) & (box30 < 0.20), 1)
vol2 = np.isfinite(vol_vs_prev) & (vol_vs_prev > 2.0)
vol15 = np.isfinite(vol_vs_prev) & (vol_vs_prev > 1.5)
vol3 = np.isfinite(vol_vs_prev) & (vol_vs_prev > 3.0)
vol2_inc = np.isfinite(vol_vs_inc) & (vol_vs_inc > 2.0)
up = np.isfinite(chg) & (chg > 0)

sell_ma10 = np.isfinite(c) & np.isfinite(ma10) & (c < ma10)

# 主口径：首板 + 前20日横盘(高低振幅<15%) + 前20日无涨停 + 当日相对前5日均量>2 + 多头排列
setup_full = first_lu & consol15 & no_lu20 & vol2 & ma_align
buy_full = t2_signal(setup_full)

P("")
P("1) 信号拆解（涨停日 T 上计数，买入发生在 T+2 开盘）")
n_lu = int((lu & in_win).sum())
n_first = int((first_lu & in_win).sum())
n_ma = int((first_lu & ma_align & in_win).sum())
n_vol = int((first_lu & vol2 & in_win).sum())
n_box = int((first_lu & consol15 & in_win).sum())
n_nol = int((first_lu & no_lu20 & in_win).sum())
n_full = int((setup_full & in_win).sum())
P(f"   区间内涨停收盘 {n_lu:,} 次；其中首板（昨日非涨停）{n_first:,}")
P(f"   首板且多头排列 C>MA5>MA10>MA20：{n_ma:,}")
P(f"   首板且相对前5日均量>2：{n_vol:,}")
P(f"   首板且此前20日高低振幅<15%：{n_box:,}")
P(f"   首板且此前20日无涨停：{n_nol:,}")
P(f"   主口径同时满足：{n_full:,} 次，涉及 {len(set(np.nonzero(setup_full & in_win)[0].tolist())):,} 只")

rows, cols = np.nonzero(setup_full & in_win)
nxt_ok = nxt_bad = nxt_none = t1_lu = 0
gap = []
above_ma10 = []
box_at = []
vr_at = []
for r, k in zip(rows, cols):
    nbar = int(pnl.n_bars[r])
    if k >= 1 and np.isfinite(box20[r, k - 1]):
        box_at.append(float(box20[r, k - 1]))
    if np.isfinite(vol_vs_prev[r, k]):
        vr_at.append(float(vol_vs_prev[r, k]))
    if k + 1 < nbar and lu[r, k + 1]:
        t1_lu += 1
    e = k + 2
    if e >= nbar:
        nxt_none += 1
        continue
    if untradable[r, e]:
        nxt_bad += 1
    else:
        nxt_ok += 1
        gap.append(o[r, e] / c[r, k] - 1.0)
        if np.isfinite(ma10[r, k]):
            above_ma10.append(c[r, k] / ma10[r, k] - 1.0)
P(f"   T+2 能开盘成交 {nxt_ok:,}（{nxt_ok / max(1, len(rows)) * 100:.1f}%）；"
  f"一字板买不到 {nxt_bad:,}；无后续K线 {nxt_none}")
P(f"   首板次日（T+1）仍涨停：{t1_lu:,}（{t1_lu / max(1, len(rows)) * 100:.1f}%）——T+2 开盘有机会接到第 3 板，也可能是开板")
if gap:
    g = np.array(gap)
    P(f"   T+2 开盘相对涨停收盘：均值 {g.mean() * 100:+.2f}%  中位数 {np.median(g) * 100:+.2f}%  "
      f"仍为正 {(g > 0).mean() * 100:.1f}%")
if above_ma10:
    a = np.array(above_ma10)
    P(f"   涨停收盘相对 MA10：均值 {a.mean() * 100:+.2f}%  中位数 {np.median(a) * 100:+.2f}%")
if box_at:
    b = np.array(box_at)
    P(f"   此前20日高低振幅：均值 {b.mean() * 100:.1f}%  中位数 {np.median(b) * 100:.1f}%")
if vr_at:
    vr = np.array(vr_at)
    P(f"   涨停日量/前5日均量：均值 {vr.mean():.2f}  中位数 {np.median(vr):.2f}")
byyear = Counter(int(pnl.dates[r, k]) // 10000 for r, k in zip(rows, cols))
P("   分年信号：" + "  ".join(f"{y}年 {byyear[y]:,}" for y in sorted(byyear)))

if len(rows):
    r0, k0 = int(rows[0]), int(cols[0])
    P(f"   例：{pnl.codes[r0]} {pnl.names[r0]} 首板 {d2s(pnl.dates[r0, k0])}  "
      f"箱体 {box20[r0, k0-1]*100:.1f}%  量比 {vol_vs_prev[r0, k0]:.2f}")
    P("        日         开     收    涨跌%   量/前均   MA5    MA10   MA20  涨停")
    for kk in range(max(0, k0 - 5), min(int(pnl.n_bars[r0]), k0 + 3)):
        vrk = vol_vs_prev[r0, kk] if np.isfinite(vol_vs_prev[r0, kk]) else float("nan")
        mark = " LU" if lu[r0, kk] else "   "
        star = " ←T" if kk == k0 else (" ←买" if kk == k0 + 2 else "")
        P(f"      {d2s(pnl.dates[r0, kk])} {o[r0,kk]:7.2f} {c[r0,kk]:7.2f} {chg[r0,kk]:6.2f} "
          f"{vrk:7.2f} {ma5[r0,kk]:7.2f} {ma10[r0,kk]:7.2f} {ma20[r0,kk]:7.2f}{mark}{star}")

# ── 2) 主口径 ──
P("")
P("2) 主口径：横盘15% + 20日无涨停 + 放量2倍 + 多头排列 + T+2开盘买 + 跌破MA10卖")
main_r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                           buy_m=buy_full, sell_m=sell_ma10)
main_m = E.signal_metrics(main_r)
print_m("主口径", main_m)

with open(os.path.join(RES, NAME + "_signal_trades.csv"), "w", encoding="utf-8") as f:
    f.write("序号,代码,名称,买入日,买入价,卖出日,卖出价,净收益率,毛收益率,持有交易日,卖出原因\n")
    for k, t in enumerate(main_r["trades"], 1):
        f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},{d2s(t['sell_date'])},"
                f"{t['sell_price']},{t['ret'] * 100:.4f}%,{t['gross_ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")
with open(os.path.join(RES, NAME + "_signal_equity.csv"), "w", encoding="utf-8") as f:
    f.write("日期,等权净值\n")
    for k, d in enumerate(main_r["dates"]):
        f.write(f"{d2s(d)},{main_r['equity_curve'][k]:.6f}\n")

P("")
P("   分年")
yr = defaultdict(list)
for t in main_r["trades"]:
    yr[t["buy_date"] // 10000].append(t["ret"])
P("   " + f"{'年份':<6}{'笔数':>7}{'胜率':>9}{'平均单笔':>10}{'中位数':>10}")
for y in sorted(yr):
    a = np.array(yr[y])
    P("   " + f"{y:<6}{len(a):>7,}{(a > 0).mean() * 100:>8.1f}%{a.mean() * 100:>9.2f}%{np.median(a) * 100:>9.2f}%")
is_r = np.array([t["ret"] for t in main_r["trades"] if t["buy_date"] < 20240817])
oos_r = np.array([t["ret"] for t in main_r["trades"] if t["buy_date"] >= 20240817])
if len(is_r) and len(oos_r):
    P(f"   样本内 ～2024-08-16：{len(is_r):,} 笔 胜率 {(is_r > 0).mean() * 100:.1f}% 平均 {is_r.mean() * 100:+.2f}%")
    P(f"   样本外 2024-08-17～：{len(oos_r):,} 笔 胜率 {(oos_r > 0).mean() * 100:.1f}% 平均 {oos_r.mean() * 100:+.2f}%")
P(f"   卖出原因：{dict(Counter(t['reason'] for t in main_r['trades']))}")
if main_r["trades"]:
    bars = np.array([t["bars"] for t in main_r["trades"]])
    P(f"   持有天数：均值 {bars.mean():.1f}  中位 {np.median(bars):.0f}  "
      f"≤3日 {(bars <= 3).mean() * 100:.1f}%  ≥10日 {(bars >= 10).mean() * 100:.1f}%  最长 {int(bars.max())}")

P("")
P("2b) 资金约束 1000万 / 单笔5% / 最多20只")
port = E.backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=buy_full, sell_m=sell_ma10)
pm = E.metrics(port)
P(f"   成交额优先：成交 {pm['trades']:,}  胜率 {pm['win_rate'] * 100:.2f}%  平均 {pct(pm['avg_ret'])}  "
  f"5年 {pct(pm['total_return'])}  回撤 {pm['max_drawdown'] * 100:.2f}%")
port_r = E.backtest(pnl, ind, EMPTY, START, END, params0(), pick="random",
                    buy_m=buy_full, sell_m=sell_ma10)
pmr = E.metrics(port_r)
P(f"   随机挑选：  成交 {pmr['trades']:,}  胜率 {pmr['win_rate'] * 100:.2f}%  平均 {pct(pmr['avg_ret'])}  "
  f"5年 {pct(pmr['total_return'])}  回撤 {pmr['max_drawdown'] * 100:.2f}%")
with open(os.path.join(RES, NAME + "_port_trades.csv"), "w", encoding="utf-8") as f:
    f.write("代码,名称,买入日,买入价,卖出日,卖出价,股数,盈亏,净收益率,持有交易日,卖出原因\n")
    for t in port["trades"]:
        f.write(f"{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},{d2s(t['sell_date'])},"
                f"{t['sell_price']},{t['shares']},{t['pnl']},{t['ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")
with open(os.path.join(RES, NAME + "_port_equity.csv"), "w", encoding="utf-8") as f:
    f.write("日期,权益\n")
    for d, v0 in port["equity"]:
        f.write(f"{d2s(d)},{v0:.2f}\n")

# ── 3) 条件拆解 ──
P("")
P("3) 从「裸首板 T+2」往上加条件（出场都是跌破 MA10）")
ab = {
    "S0 裸首板 T+2": first_lu,
    "S1 +多头排列": first_lu & ma_align,
    "S2 +放量2倍": first_lu & ma_align & vol2,
    "S3 +20日无涨停": first_lu & ma_align & vol2 & no_lu20,
    "S4 主口径 +横盘15%": setup_full,
}
abl = {}
eqs = {}
for label, st in ab.items():
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                          buy_m=t2_signal(st), sell_m=sell_ma10)
    m = E.signal_metrics(r)
    abl[label] = m
    eqs[label] = r["equity_curve"]
    P(f"   {label:<22} {m['trades']:>7,}  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  持有 {m['avg_bars']:5.1f}日  等权5年 {pct(m['total_return']):>9}  "
      f"夏普 {m['sharpe']:6.2f}")

# ── 4) 买入时点 ──
P("")
P("4) 同一主口径 setup，改买入时点（仍跌破 MA10 卖）")
timing = {
    "T+1 开盘（涨停次日，信号打在 T）": setup_full,
    "T+2 开盘（主口径，信号打在 T+1）": None,  # already have
    "T+3 开盘": lag(setup_full, 2),
}
# T+1: buy_m = setup (engine fills next open = T+1)
tm = {}
r_t1 = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                         buy_m=setup_full, sell_m=sell_ma10)
tm["T+1 开盘（涨停次日）"] = E.signal_metrics(r_t1)
print_m("T+1 开盘（涨停次日）", tm["T+1 开盘（涨停次日）"])
tm["T+2 开盘（主口径）"] = main_m
r_t3 = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                         buy_m=lag(setup_full, 2), sell_m=sell_ma10)
tm["T+3 开盘"] = E.signal_metrics(r_t3)
print_m("T+3 开盘", tm["T+3 开盘"])
eqs["T+1 开盘"] = r_t1["equity_curve"]
eqs["T+3 开盘"] = r_t3["equity_curve"]

# ── 5) 横盘 / 放量敏感度 ──
P("")
P("5) 横盘定义与放量倍数（其余用主口径，T+2，跌破 MA10）")
sens = {}
for label, st in (
    ("箱体10%", first_lu & ma_align & vol2 & no_lu20 & consol10),
    ("箱体15%（主）", setup_full),
    ("箱体20%", first_lu & ma_align & vol2 & no_lu20 & consol20),
    ("箱体25%", first_lu & ma_align & vol2 & no_lu20 & consol25),
    ("收盘波幅12%", first_lu & ma_align & vol2 & no_lu20 & consol_c12),
    ("30日箱体20%", first_lu & ma_align & vol2 & no_lu20 & consol30),
    ("放量1.5×", first_lu & ma_align & vol15 & no_lu20 & consol15),
    ("放量2×（主）", setup_full),
    ("放量3×", first_lu & ma_align & vol3 & no_lu20 & consol15),
    ("放量含当日均量2×", first_lu & ma_align & vol2_inc & no_lu20 & consol15),
):
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                          buy_m=t2_signal(st), sell_m=sell_ma10)
    m = E.signal_metrics(r)
    sens[label] = m
    P(f"   {label:<18} {m['trades']:>7,}  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  等权5年 {pct(m['total_return']):>9}")

# ── 6) 出场对照 ──
P("")
P("6) 主口径买入，换出场")
exits = {
    "E0 跌破MA10（主）": (params0(), sell_ma10),
    "E1 跌破MA5": (params0(), np.isfinite(c) & np.isfinite(ma5) & (c < ma5)),
    "E2 止盈8%止损5%最长10日": (params0(take_profit=0.08, stop_loss=0.05, max_bars=10), None),
    "E3 持有满5日": (params0(max_bars=5), None),
    "E4 收盘不涨停就卖": (params0(), ~lu),
}
exm = {}
for label, (p, sm) in exits.items():
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, p, buy_m=buy_full, sell_m=sm)
    m = E.signal_metrics(r)
    exm[label] = m
    P(f"   {label:<22} {m['trades']:>7,}  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  持有 {m['avg_bars']:5.1f}日  等权5年 {pct(m['total_return']):>9}")

# ── 7) 对照策略 26 ──
P("")
P("7) 对照：策略26 涨停次日跟进（近1日1个涨停+换手>5%，止盈8/止损5/最长5日）")
sp26 = json.load(open(os.path.join(OUT, "strategies", "26_涨停次日跟进.spec.json"), encoding="utf-8"))
r26 = E.signal_backtest(pnl, ind, sp26["spec"], START, END,
                        params0(take_profit=0.08, stop_loss=0.05, max_bars=5))
m26 = E.signal_metrics(r26)
print_m("26 涨停次日跟进", m26)

# 净值对照表
dates = [d2s(d) for d in main_r["dates"]]
with open(os.path.join(RES, "sideways_lu_equity_alters.csv"), "w", encoding="utf-8") as f:
    f.write("日期,主口径T2,裸首板T2,多头排列T2,T+1开盘\n")
    e0 = main_r["equity_curve"]
    e_s0 = eqs["S0 裸首板 T+2"]
    e_s1 = eqs["S1 +多头排列"]
    e_t1 = eqs["T+1 开盘"]
    for i in range(len(dates)):
        f.write(f"{dates[i]},{e0[i]:.6f},{e_s0[i]:.6f},{e_s1[i]:.6f},{e_t1[i]:.6f}\n")

summary = {
    "primary": dump_row(main_m),
    "port_amount": {"trades": pm["trades"], "win": pm["win_rate"], "avg": pm["avg_ret"],
                    "total": pm["total_return"], "mdd": pm["max_drawdown"]},
    "port_random": {"trades": pmr["trades"], "win": pmr["win_rate"], "avg": pmr["avg_ret"],
                    "total": pmr["total_return"], "mdd": pmr["max_drawdown"]},
    "signals": n_full, "fillable": nxt_ok, "unfillable": nxt_bad,
    "t1_still_lu": t1_lu,
    "ablation": {k: dump_row(v) for k, v in abl.items()},
    "timing": {k: dump_row(v) for k, v in tm.items()},
    "sens": {k: dump_row(v) for k, v in sens.items()},
    "exits": {k: dump_row(v) for k, v in exm.items()},
    "s26": dump_row(m26),
    "years": {str(y): {"n": len(a), "win": float((np.array(a) > 0).mean()),
                       "avg": float(np.array(a).mean())} for y, a in yr.items()},
}
json.dump(summary, open(os.path.join(OUT, "sideways_lu_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

with open(os.path.join(RES, "sideways_lu_variants.csv"), "w", encoding="utf-8") as f:
    f.write("组,名称,笔数,胜率,平均单笔,中位数,盈亏比,平均持有,等权5年,年化,最大回撤,夏普\n")

    def w(group, name, m):
        pf = m["profit_factor"] if np.isfinite(m["profit_factor"]) else 0.0
        f.write(f"{group},{name},{m['trades']},{m['win_rate'] * 100:.2f}%,{m['avg_ret'] * 100:.4f}%,"
                f"{m['median_ret'] * 100:.4f}%,{pf:.3f},{m['avg_bars']:.2f},"
                f"{m['total_return'] * 100:.2f}%,{m['annual_return'] * 100:.2f}%,"
                f"{m['max_drawdown'] * 100:.2f}%,{m['sharpe']:.3f}\n")

    w("主口径", "横盘放量首板 T+2 / 跌破MA10", main_m)
    for k, m in abl.items():
        w("加条件", k, m)
    for k, m in tm.items():
        w("买入时点", k, m)
    for k, m in sens.items():
        w("敏感度", k, m)
    for k, m in exm.items():
        w("出场", k, m)
    w("对照", "26 涨停次日跟进", m26)

P("")
P("已写出 sideways_lu_summary.json / results/sideways_lu_variants.csv / 策略 34 逐笔与净值")
P("=" * 100)
LOG.close()
