"""专项回测：横盘>20日放量首板后，等第一次回踩 5 日线再买，跌破 10 日线卖。

与 34 的差别只在买点：不是涨停后第 2 个交易日开盘追，而是等回踩。
「回踩五日线」主口径与页面编译器一致：收盘落在 MA5 ±2% 带内，且仍站在 MA10 上。
对照：最低价触及 MA5、第一次收盘跌破 MA5（S12 口径）。
最长等待 15 个交易日；等待期内若先跌破 MA10 则放弃该事件。
买入 = 回踩日收盘信号 → 次日开盘；卖出 = 收盘跌破 MA10 → 次日开盘。
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
NAME = "35_横盘放量首板回踩MA5"
LOG = open(os.path.join(OUT, "sideways_lu_ma5_run.log"), "w", encoding="utf-8")
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


P("=" * 100)
P("横盘>20日 + 放量上涨 + 价>MA5>MA10>MA20 + 首板后第一次回踩MA5买入 + 跌破MA10卖")
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
    hh20 = E.roll_max(h, 20)
    ll20 = E.roll_min(l, 20)
    box20 = (hh20 - ll20) / np.maximum(ll20, 1e-12)
    vol_vs_prev = v / lag(vma5, 1)

first_lu = lu & ~lag(lu, 1)
lu20 = ind.get("lu20")
no_lu20 = lag(np.isfinite(lu20) & (lu20 == 0), 1)
consol15 = lag(np.isfinite(box20) & (box20 < 0.15), 1)
vol2 = np.isfinite(vol_vs_prev) & (vol_vs_prev > 2.0)
setup_full = first_lu & consol15 & no_lu20 & vol2 & ma_align
setup_lu = first_lu & ma_align          # 无横盘/放量，只要求首板+多头
sell_ma10 = np.isfinite(c) & np.isfinite(ma10) & (c < ma10)


def pullback_hit(i, kk, mode):
    m5, m10 = ma5[i, kk], ma10[i, kk]
    if not (np.isfinite(m5) and np.isfinite(m10) and np.isfinite(c[i, kk])):
        return False
    if mode == "band":
        return (m5 * 0.98) <= c[i, kk] <= (m5 * 1.02)
    if mode == "touch":
        return np.isfinite(l[i, kk]) and np.isfinite(h[i, kk]) \
            and (l[i, kk] <= m5 <= h[i, kk]) and (c[i, kk] >= m5)
    if mode == "near":
        return (m5 <= c[i, kk] <= m5 * 1.02)
    if mode == "break5":
        return c[i, kk] < m5
    return False


def scan(setup, mode, max_wait=15, need_ma10=True):
    """对每个 setup 事件找第一次回踩。返回 buy_m、事件统计、等待天数列表。"""
    buy = np.zeros((N, L), dtype=bool)
    stats = Counter()
    waits = []
    dist_ma5 = []
    examples = []
    for i in range(N):
        n = int(pnl.n_bars[i])
        if n < 40:
            continue
        k = 20
        while k < n - 2:
            if not (setup[i, k] and in_win[i, k]):
                k += 1
                continue
            stats["事件"] += 1
            found = None
            why = "超时"
            hi = min(n - 1, k + 1 + max_wait)
            for kk in range(k + 1, hi):
                if not np.isfinite(ma10[i, kk]):
                    continue
                if need_ma10 and c[i, kk] < ma10[i, kk]:
                    why = "先破MA10"
                    break
                if pullback_hit(i, kk, mode):
                    if need_ma10 and c[i, kk] < ma10[i, kk]:
                        why = "回踩日已破MA10"
                        break
                    found = kk
                    break
            if found is None:
                stats[why] += 1
                k += 1
                continue
            stats["回踩"] += 1
            buy[i, found] = True
            waits.append(found - k)
            if np.isfinite(ma5[i, found]) and ma5[i, found] != 0:
                dist_ma5.append(c[i, found] / ma5[i, found] - 1.0)
            if len(examples) < 3:
                examples.append((i, k, found))
            k = found + 1
    return buy, stats, np.array(waits, dtype=np.int16) if waits else np.array([], dtype=np.int16), \
        np.array(dist_ma5) if dist_ma5 else np.array([]), examples


P("")
P("1) 主口径回踩扫描（收盘在 MA5±2%，等待期内先破 MA10 则放弃，最多等 15 日）")
buy_full, st, waits, dist, exs = scan(setup_full, "band", 15, True)
P(f"   横盘放量首板事件 {st['事件']:,}；形成回踩 {st['回踩']:,}；"
  f"等待中先破 MA10 {st['先破MA10']:,}；{st['超时']:,} 次 15 日内未回到 MA5 带")
if len(waits):
    P(f"   涨停→回踩等待：均值 {waits.mean():.1f} 日  中位 {np.median(waits):.0f} 日  "
      f"1日 {(waits == 1).mean() * 100:.1f}%  ≤3日 {(waits <= 3).mean() * 100:.1f}%  "
      f"≥10日 {(waits >= 10).mean() * 100:.1f}%")
if len(dist):
    P(f"   回踩日收盘相对 MA5：均值 {dist.mean() * 100:+.2f}%  中位数 {np.median(dist) * 100:+.2f}%")

for i, k, kp in exs[:2]:
    P(f"   例：{pnl.codes[i]} {pnl.names[i]} 首板 {d2s(pnl.dates[i, k])} → 回踩 {d2s(pnl.dates[i, kp])} "
      f"（等 {kp - k} 日）")
    P("        日         收     MA5    MA10   相对MA5  涨停")
    for kk in range(k, min(int(pnl.n_bars[i]), kp + 2)):
        rel = (c[i, kk] / ma5[i, kk] - 1) * 100 if np.isfinite(ma5[i, kk]) and ma5[i, kk] else float("nan")
        mark = " LU" if lu[i, kk] else "   "
        star = " ←T" if kk == k else (" ←回踩" if kk == kp else "")
        P(f"      {d2s(pnl.dates[i, kk])} {c[i,kk]:7.2f} {ma5[i,kk]:7.2f} {ma10[i,kk]:7.2f} "
          f"{rel:7.2f}%{mark}{star}")

# ── 2) 主口径回测 ──
P("")
P("2) 主口径：回踩日收盘信号 → 次日开盘买；跌破 MA10 卖")
main_r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(),
                           buy_m=buy_full, sell_m=sell_ma10)
main_m = E.signal_metrics(main_r)
print_m("主口径 回踩MA5±2%", main_m)

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
    P(f"   买入后持有：均值 {bars.mean():.1f}  中位 {np.median(bars):.0f}  "
      f"≤2日 {(bars <= 2).mean() * 100:.1f}%  ≥10日 {(bars >= 10).mean() * 100:.1f}%")

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

# ── 3) 回踩定义 ──
P("")
P("3) 回踩定义对照（同一横盘放量首板，最多等 15 日，跌破 MA10 卖）")
defs = [
    ("P0 不回踩：T+2 开盘追（策略34）", None, None),
    ("P1 收盘 MA5±2%（主口径）", "band", True),
    ("P2 最低价触及 MA5 且收盘仍在线上", "touch", True),
    ("P3 收盘落在 [MA5, MA5+2%]", "near", True),
    ("P4 第一次收盘跌破 MA5（S12口径）", "break5", False),
    ("P4b 跌破 MA5 但回踩日仍 ≥ MA10", "break5", True),
]
def_m = {}
eqs = {"主口径回踩": main_r["equity_curve"]}
# T+2 chase overlay
buy_t2 = lag(setup_full, 1)
r_t2 = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=buy_t2, sell_m=sell_ma10)
def_m["P0 不回踩：T+2 开盘追（策略34）"] = E.signal_metrics(r_t2)
eqs["T+2追"] = r_t2["equity_curve"]
print_m("P0 T+2 开盘追（对照）", def_m["P0 不回踩：T+2 开盘追（策略34）"])
def_m["P1 收盘 MA5±2%（主口径）"] = main_m

for label, mode, need10 in defs:
    if mode is None or label.startswith("P1"):
        continue
    b, st2, w2, _, _ = scan(setup_full, mode, 15, need10)
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=b, sell_m=sell_ma10)
    m = E.signal_metrics(r)
    def_m[label] = m
    if "P4 " in label:
        eqs["破MA5回踩"] = r["equity_curve"]
    if "P2" in label:
        eqs["触及MA5"] = r["equity_curve"]
    P(f"   {label}")
    P(f"     事件 {st2['事件']:,} → 回踩 {st2['回踩']:,}（先破MA10 {st2['先破MA10']:,}，超时 {st2['超时']:,}）"
      + (f"  等待中位 {np.median(w2):.0f}日" if len(w2) else ""))
    print_m(label, m)

# ── 4) 等待期 ──
P("")
P("4) 最长等待（主口径 band ±2%）")
wait_m = {"15日（主）": main_m}
for w in (5, 10, 20, 30):
    b, st2, w2, _, _ = scan(setup_full, "band", w, True)
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=b, sell_m=sell_ma10)
    m = E.signal_metrics(r)
    wait_m[f"{w}日"] = m
    P(f"   最多等 {w:>2} 日  回踩 {st2['回踩']:>5,} / 事件 {st2['事件']:,}  "
      f"超时 {st2['超时']:>5,}  先破MA10 {st2['先破MA10']:>5,}  "
      f"成交 {m['trades']:>5,}  胜率 {m['win_rate']*100:5.1f}%  平均 {pct(m['avg_ret'])}  "
      f"等权5年 {pct(m['total_return'])}")

# ── 5) 条件拆解：裸首板也做回踩 ──
P("")
P("5) 同一回踩（band±2%，等15日），是否需要横盘放量")
abl = {"横盘放量首板（主）": main_m}
b_lu, st_lu, _, _, _ = scan(setup_lu, "band", 15, True)
r_lu = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=b_lu, sell_m=sell_ma10)
abl["仅首板+多头排列"] = E.signal_metrics(r_lu)
eqs["仅首板回踩"] = r_lu["equity_curve"]
P(f"   仅首板+多头：事件 {st_lu['事件']:,} → 回踩 {st_lu['回踩']:,}")
print_m("仅首板+多头排列 再回踩MA5", abl["仅首板+多头排列"])

b_raw, st_raw, _, _, _ = scan(first_lu, "band", 15, True)
r_raw = E.signal_backtest(pnl, ind, EMPTY, START, END, params0(), buy_m=b_raw, sell_m=sell_ma10)
abl["裸首板回踩"] = E.signal_metrics(r_raw)
P(f"   裸首板：事件 {st_raw['事件']:,} → 回踩 {st_raw['回踩']:,}")
print_m("裸首板 再回踩MA5", abl["裸首板回踩"])

# ── 6) 出场 ──
P("")
P("6) 主口径买入，换出场")
exm = {}
for label, p, sm in (
    ("E0 跌破MA10（主）", params0(), sell_ma10),
    ("E1 跌破MA5", params0(), np.isfinite(c) & np.isfinite(ma5) & (c < ma5)),
    ("E2 止盈8%止损5%最长10日", params0(take_profit=0.08, stop_loss=0.05, max_bars=10), None),
    ("E3 持有满5日", params0(max_bars=5), None),
):
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, p, buy_m=buy_full, sell_m=sm)
    m = E.signal_metrics(r)
    exm[label] = m
    P(f"   {label:<22} {m['trades']:>7,}  胜率 {m['win_rate']*100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  持有 {m['avg_bars']:4.1f}日  等权5年 {pct(m['total_return']):>9}")

dates = [d2s(d) for d in main_r["dates"]]
with open(os.path.join(RES, "sideways_lu_ma5_equity.csv"), "w", encoding="utf-8") as f:
    cols = ["日期", "主口径回踩", "T+2追", "破MA5回踩", "触及MA5", "仅首板回踩"]
    f.write(",".join(cols) + "\n")
    n = len(dates)
    for i in range(n):
        row = [dates[i]]
        for k in cols[1:]:
            eq = eqs.get(k)
            row.append(f"{eq[i]:.6f}" if eq is not None else "")
        f.write(",".join(row) + "\n")

summary = {
    "primary": dump_row(main_m),
    "port_amount": {"trades": pm["trades"], "win": pm["win_rate"], "avg": pm["avg_ret"],
                    "total": pm["total_return"], "mdd": pm["max_drawdown"]},
    "port_random": {"trades": pmr["trades"], "win": pmr["win_rate"], "avg": pmr["avg_ret"],
                    "total": pmr["total_return"], "mdd": pmr["max_drawdown"]},
    "events": int(st["事件"]), "pullbacks": int(st["回踩"]),
    "broke_ma10": int(st["先破MA10"]), "timeout": int(st["超时"]),
    "wait_mean": float(waits.mean()) if len(waits) else None,
    "wait_med": float(np.median(waits)) if len(waits) else None,
    "defs": {k: dump_row(v) for k, v in def_m.items()},
    "waits": {k: dump_row(v) for k, v in wait_m.items()},
    "ablation": {k: dump_row(v) for k, v in abl.items()},
    "exits": {k: dump_row(v) for k, v in exm.items()},
    "years": {str(y): {"n": len(a), "win": float((np.array(a) > 0).mean()),
                       "avg": float(np.array(a).mean())} for y, a in yr.items()},
}
json.dump(summary, open(os.path.join(OUT, "sideways_lu_ma5_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

with open(os.path.join(RES, "sideways_lu_ma5_variants.csv"), "w", encoding="utf-8") as f:
    f.write("组,名称,笔数,胜率,平均单笔,中位数,盈亏比,平均持有,等权5年,年化,最大回撤,夏普\n")

    def w(group, name, m):
        pf = m["profit_factor"] if np.isfinite(m["profit_factor"]) else 0.0
        f.write(f"{group},{name},{m['trades']},{m['win_rate']*100:.2f}%,{m['avg_ret']*100:.4f}%,"
                f"{m['median_ret']*100:.4f}%,{pf:.3f},{m['avg_bars']:.2f},"
                f"{m['total_return']*100:.2f}%,{m['annual_return']*100:.2f}%,"
                f"{m['max_drawdown']*100:.2f}%,{m['sharpe']:.3f}\n")

    w("主口径", "横盘放量首板 回踩MA5±2%", main_m)
    for k, m in def_m.items():
        w("回踩定义", k, m)
    for k, m in wait_m.items():
        w("等待期", k, m)
    for k, m in abl.items():
        w("过滤", k, m)
    for k, m in exm.items():
        w("出场", k, m)

P("")
P("已写出 sideways_lu_ma5_summary.json / results/sideways_lu_ma5_variants.csv")
P("=" * 100)
LOG.close()
