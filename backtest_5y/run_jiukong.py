"""专项回测：九空一多。

公开资料没有统一的「九空一多」公式。主口径取字面：
  前 9 个交易日收阴（收盘 < 开盘），当日收阳（收盘 > 开盘）→ 次日开盘买。
对照：连续下跌后收涨、TD 神奇九转低九、连阴天数敏感度、常见过滤与出场。

页面可编译规则：当连续9天阴线后出现阳线时买入；当盈利超过8%或者亏损超过5%时卖出；最长持有10天。
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
NAME = "33_九空一多"
LOG = open(os.path.join(OUT, "jiukong_run.log"), "w", encoding="utf-8")

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


def window_mask(pnl):
    m = np.zeros((pnl.N, pnl.L), dtype=bool)
    for i in range(pnl.N):
        n = int(pnl.n_bars[i])
        if n:
            d = pnl.dates[i, :n]
            m[i, :n] = (d >= START) & (d <= END)
    return m


def down_then_up(close, n):
    """前 n 日收盘连跌，当日收涨。"""
    dn = np.zeros(close.shape, dtype=bool)
    up = np.zeros(close.shape, dtype=bool)
    dn[:, 1:] = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1]) & (close[:, 1:] < close[:, :-1])
    up[:, 1:] = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1]) & (close[:, 1:] > close[:, :-1])
    m = up.copy()
    for s in range(1, n + 1):
        prev = np.zeros_like(dn)
        prev[:, s:] = dn[:, :-s]
        m &= prev
    return m


def td_count(close):
    """TD Sequential 向下 setup：连续 C < C[4] 的根数。"""
    lower = np.zeros(close.shape, dtype=bool)
    lower[:, 4:] = np.isfinite(close[:, 4:]) & np.isfinite(close[:, :-4]) & (close[:, 4:] < close[:, :-4])
    cnt = np.zeros(close.shape, dtype=np.int16)
    for j in range(close.shape[1]):
        prev = cnt[:, j - 1] if j else 0
        cnt[:, j] = np.where(lower[:, j], prev + 1, 0)
    higher = np.zeros(close.shape, dtype=bool)
    higher[:, 4:] = np.isfinite(close[:, 4:]) & np.isfinite(close[:, :-4]) & (close[:, 4:] > close[:, :-4])
    return cnt, lower, higher


def params_e0(**kw):
    p = {"init_cash": 10_000_000, "pos_pct": 5, "max_hold": 20,
         "fee_permil": 0.5, "slip_permil": 0.5,
         "take_profit": 0.08, "stop_loss": 0.05, "max_bars": 10, "min_bars": 1}
    p.update(kw)
    return p


def dump_row(m):
    return {
        "trades": int(m["trades"]),
        "win": float(m["win_rate"]),
        "avg": float(m["avg_ret"]),
        "med": float(m["median_ret"]),
        "pf": float(m["profit_factor"]) if np.isfinite(m["profit_factor"]) else None,
        "avg_win": float(m["avg_win"]),
        "avg_loss": float(m["avg_loss"]),
        "bars": float(m["avg_bars"]),
        "total": float(m["total_return"]),
        "ann": float(m["annual_return"]),
        "mdd": float(m["max_drawdown"]),
        "sharpe": float(m["sharpe"]),
        "stocks": int(m["stocks_traded"]),
        "missed": int(m["missed_signals"]),
        "best": float(m["best"]),
        "worst": float(m["worst"]),
        "avg_pos": float(m["avg_positions"]),
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
P("九空一多：全市场回测")
P("主口径：前 9 日收阴 + 当日收阳 → 次日开盘买；止盈 8% / 止损 5% / 最长 10 日")
P("样本窗口 2021-08-17 ~ 2026-08-17；成本 手续费 0.5‰ + 滑点 0.5‰ 双边")
P("=" * 100)

pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
i0, i1 = pnl.col_range(START, END)
N, L, T = pnl.N, pnl.L, pnl.T
o, h, l, c, v, amt, chg, tr = (pnl.cf[k] for k in
                               ("open", "high", "low", "close", "volume", "amount", "chg", "turnover"))
P(f"样本：{N} 只、{int(pnl.n_bars.sum())} 根日线，区间 {d2s(START)} ~ {d2s(END)}（{i1 - i0} 个交易日）")

in_win = window_mask(pnl)
rsi6 = ind.get("rsi6")
vma5 = ind.get("vma5")
ma5 = ind.get("ma5")
with np.errstate(invalid="ignore", divide="ignore"):
    fcap = np.where(tr > 0, v * 1e4 / tr * c, np.nan)
    volr = v / vma5
    yang = (c > o) & np.isfinite(c) & np.isfinite(o)
    yin = (c < o) & np.isfinite(c) & np.isfinite(o)

codes = np.array([str(x) for x in pnl.codes])
main_bd = np.array([cd[:2] not in ("30", "68") and cd[0] not in ("4", "8") for cd in codes])

spec9 = {"buy": {"op": "and", "conds": [{"kind": "yinThenYang", "n": 9}]},
         "sell": {"op": "and", "conds": []}}
buy9 = E.group_mask(ind, spec9["buy"], c.shape)
buy9[pnl.n_bars < 30] = False

# 人工核对：引擎掩码 vs 直接数阴线
hand = yang.copy()
for s in range(1, 10):
    prev = np.zeros_like(yin)
    prev[:, s:] = yin[:, :-s]
    hand &= prev
hand[pnl.n_bars < 30] = False
diff = int(np.sum(buy9 != hand))
P(f"引擎 yinThenYang 与手算阴线掩码差异：{diff} 格（应为 0）")

sig = buy9 & in_win
rows, cols = np.nonzero(sig)
P("")
P(f"1) 九阴一阳信号：区间内 {len(rows):,} 次，涉及 {len(set(rows.tolist())):,} 只")

# 举例
if len(rows):
    r0, k0 = int(rows[0]), int(cols[0])
    P(f"   例：{pnl.codes[r0]} {pnl.names[r0]} 信号日 {d2s(pnl.dates[r0, k0])}")
    P("   日    开盘    收盘  阴阳")
    for kk in range(k0 - 9, k0 + 1):
        mark = "阳" if c[r0, kk] > o[r0, kk] else ("阴" if c[r0, kk] < o[r0, kk] else "平")
        star = "  ←信号" if kk == k0 else ""
        P(f"   {d2s(pnl.dates[r0, kk])}  {o[r0, kk]:7.2f}  {c[r0, kk]:7.2f}  {mark}{star}")

nxt_ok = nxt_bad = nxt_none = 0
gap = []
with np.errstate(invalid="ignore"):
    untradable = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
for r, k in zip(rows, cols):
    if k + 1 >= int(pnl.n_bars[r]):
        nxt_none += 1
        continue
    if untradable[r, k + 1]:
        nxt_bad += 1
    else:
        nxt_ok += 1
        gap.append(o[r, k + 1] / c[r, k] - 1)
P(f"   次日能开盘成交 {nxt_ok:,}（{nxt_ok / max(1, len(rows)) * 100:.1f}%）；"
  f"一字板买不到 {nxt_bad:,}；无后续K线 {nxt_none}")
if gap:
    g = np.array(gap)
    P(f"   能买到时次日开盘跳空：均值 {g.mean() * 100:+.2f}%  中位数 {np.median(g) * 100:+.2f}%  "
      f"正跳空 {(g > 0).mean() * 100:.1f}%")
byyear = Counter(int(pnl.dates[r, k]) // 10000 for r, k in zip(rows, cols))
P("   分年信号数：" + "  ".join(f"{y}年 {byyear[y]:,}" for y in sorted(byyear)))

# 九连阴期间跌幅
drop9 = []
for r, k in zip(rows, cols):
    if k >= 9 and np.isfinite(c[r, k - 9]) and c[r, k - 9] > 0:
        drop9.append(c[r, k - 1] / c[r, k - 9] - 1.0)
if drop9:
    d9 = np.array(drop9)
    P(f"   九连阴期间（信号日前 9 日→前 1 日）累计涨跌：均值 {d9.mean() * 100:+.2f}%  "
      f"中位数 {np.median(d9) * 100:+.2f}%")

# ── 2) 主口径 ──
P("")
P("2) 主口径（九阴一阳 + 止盈8% 止损5% 最长10日）")
main_r = E.signal_backtest(pnl, ind, spec9, START, END, params_e0())
main_m = E.signal_metrics(main_r)
print_m("A 九阴一阳 / TP8 SL5 max10", main_m)

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
oos_cut = 20240817
is_r = np.array([t["ret"] for t in main_r["trades"] if t["buy_date"] < oos_cut])
oos_r = np.array([t["ret"] for t in main_r["trades"] if t["buy_date"] >= oos_cut])
if len(is_r) and len(oos_r):
    P(f"   样本内 2021-08-17~2024-08-16：{len(is_r):,} 笔 胜率 {(is_r > 0).mean() * 100:.1f}% 平均 {is_r.mean() * 100:+.2f}%")
    P(f"   样本外 2024-08-17~2026-08-17：{len(oos_r):,} 笔 胜率 {(oos_r > 0).mean() * 100:.1f}% 平均 {oos_r.mean() * 100:+.2f}%")
P(f"   卖出原因：{dict(Counter(t['reason'] for t in main_r['trades']))}")

# 资金约束
P("")
P("2b) 资金约束组合 1000万 / 单笔5% / 最多20只")
port = E.backtest(pnl, ind, spec9, START, END, params_e0())
pm = E.metrics(port)
P(f"   成交额优先：成交 {pm['trades']:,}  胜率 {pm['win_rate'] * 100:.2f}%  平均单笔 {pct(pm['avg_ret'])}  "
  f"5 年 {pct(pm['total_return'])}  回撤 {pm['max_drawdown'] * 100:.2f}%")
port_r = E.backtest(pnl, ind, spec9, START, END, params_e0(), pick="random")
pmr = E.metrics(port_r)
P(f"   随机挑选：  成交 {pmr['trades']:,}  胜率 {pmr['win_rate'] * 100:.2f}%  平均单笔 {pct(pmr['avg_ret'])}  "
  f"5 年 {pct(pmr['total_return'])}  回撤 {pmr['max_drawdown'] * 100:.2f}%")
with open(os.path.join(RES, NAME + "_port_trades.csv"), "w", encoding="utf-8") as f:
    f.write("代码,名称,买入日,买入价,卖出日,卖出价,股数,盈亏,净收益率,持有交易日,卖出原因\n")
    for t in port["trades"]:
        f.write(f"{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},{d2s(t['sell_date'])},"
                f"{t['sell_price']},{t['shares']},{t['pnl']},{t['ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")
with open(os.path.join(RES, NAME + "_port_equity.csv"), "w", encoding="utf-8") as f:
    f.write("日期,权益\n")
    for d, v in port["equity"]:
        f.write(f"{d2s(d)},{v:.2f}\n")

# ── 3) 连阴天数敏感度 ──
P("")
P("3) 连阴天数敏感度（N 阴一阳，同一套 TP8/SL5/max10）")
sens = {}
for n in (4, 5, 6, 7, 8, 9, 10, 12):
    sp = {"buy": {"op": "and", "conds": [{"kind": "yinThenYang", "n": n}]},
          "sell": {"op": "and", "conds": []}}
    r = E.signal_backtest(pnl, ind, sp, START, END, params_e0())
    m = E.signal_metrics(r)
    sens[n] = m
    P(f"   {n:>2}阴一阳  {m['trades']:>7,} 笔  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  中位 {pct(m['median_ret']):>8}  等权5年 {pct(m['total_return']):>9}  "
      f"夏普 {m['sharpe']:6.2f}")

# ── 4) 其它定义 ──
P("")
P("4) 其它「九空一多」定义（同一套 TP8/SL5/max10）")
cnt, lower, higher = td_count(c)
td9 = (cnt == 9)
td_flip = np.zeros_like(higher)
td_flip[:, 1:] = higher[:, 1:] & (cnt[:, :-1] >= 9)

alts = {
    "B 九跌一涨（前9日收盘连跌+当日收涨）": down_then_up(c, 9),
    "C TD低九当日（连续9日 C<C[4]）": td9,
    "D TD低九后一多（低九后首日 C>C[4]）": td_flip,
    "A' 九阴一阳且当日收涨": buy9 & (np.concatenate(
        [np.zeros((N, 1), dtype=bool), (c[:, 1:] > c[:, :-1]) & np.isfinite(c[:, 1:]) & np.isfinite(c[:, :-1])],
        axis=1)),
}
alt_m = {}
alt_eq = {}
for label, mask in alts.items():
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params_e0(), buy_m=mask)
    m = E.signal_metrics(r)
    alt_m[label] = m
    alt_eq[label] = (r["dates"], r["equity_curve"])
    print_m(label, m)

# ── 5) 出场 ──
P("")
P("5) 同一买入（九阴一阳）换出场")
yin_sell = yin
ma5_sell = np.isfinite(ma5) & np.isfinite(c) & (c > ma5)
exits = {
    "E0 止盈8%止损5%最长10日": (params_e0(), None),
    "E1 持有满5日": (params_e0(take_profit=0.0, stop_loss=0.0, max_bars=5), None),
    "E2 持有满10日": (params_e0(take_profit=0.0, stop_loss=0.0, max_bars=10), None),
    "E3 下一根阴线卖": (params_e0(take_profit=0.0, stop_loss=0.0, max_bars=0), yin_sell),
    "E4 收盘站上MA5卖": (params_e0(take_profit=0.0, stop_loss=0.0, max_bars=15), ma5_sell),
    "E5 止盈8%止损5%最长5日": (params_e0(max_bars=5), None),
}
exit_m = {}
for label, (p, sm) in exits.items():
    r = E.signal_backtest(pnl, ind, spec9, START, END, p, sell_m=sm)
    m = E.signal_metrics(r)
    exit_m[label] = m
    P(f"   {label:<22} {m['trades']:>7,}  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  持有 {m['avg_bars']:4.1f}日  等权5年 {pct(m['total_return']):>9}")

# ── 6) 过滤 ──
P("")
P("6) 九阴一阳 + 过滤（仍用 TP8/SL5/max10）")
filters = {
    "F0 无过滤": buy9,
    "F1 RSI6<30": buy9 & np.isfinite(rsi6) & (rsi6 < 30),
    "F2 RSI6<20": buy9 & np.isfinite(rsi6) & (rsi6 < 20),
    "F3 阳线放量>1.5×均量": buy9 & np.isfinite(volr) & (volr > 1.5),
    "F4 阳线缩量<0.7×均量": buy9 & np.isfinite(volr) & (volr < 0.7),
    "F5 流通市值50-200亿": buy9 & np.isfinite(fcap) & (fcap >= 5e9) & (fcap <= 2e10),
    "F6 仅沪深主板": None,  # universe
    "F7 RSI6<30 且放量": buy9 & np.isfinite(rsi6) & (rsi6 < 30) & np.isfinite(volr) & (volr > 1.5),
}
filt_m = {}
for label, mask in filters.items():
    uni = main_bd if label == "F6 仅沪深主板" else None
    bm = buy9 if label == "F6 仅沪深主板" else mask
    r = E.signal_backtest(pnl, ind, EMPTY, START, END, params_e0(), universe=uni, buy_m=bm)
    m = E.signal_metrics(r)
    filt_m[label] = m
    P(f"   {label:<22} {m['trades']:>7,}  胜率 {m['win_rate'] * 100:5.2f}%  "
      f"平均 {pct(m['avg_ret']):>8}  等权5年 {pct(m['total_return']):>9}  夏普 {m['sharpe']:6.2f}")

# 对照策略 21 已有结果，这里用同一引擎再跑四连阴+RSI 方便并排
P("")
P("7) 对照：策略21 四连阴+RSI6<25（同一窗口同一成本）")
sp21 = json.load(open(os.path.join(OUT, "strategies", "21_四连阴+RSI超卖.spec.json"), encoding="utf-8"))
r21 = E.signal_backtest(pnl, ind, sp21["spec"], START, END,
                        params_e0(take_profit=0.08, stop_loss=0.05, max_bars=10))
m21 = E.signal_metrics(r21)
print_m("21 四连阴+RSI超卖", m21)

# ── 汇总 json / csv ──
summary = {
    "primary": dump_row(main_m),
    "port_amount": {"trades": pm["trades"], "win": pm["win_rate"], "avg": pm["avg_ret"],
                    "total": pm["total_return"], "mdd": pm["max_drawdown"]},
    "port_random": {"trades": pmr["trades"], "win": pmr["win_rate"], "avg": pmr["avg_ret"],
                    "total": pmr["total_return"], "mdd": pmr["max_drawdown"]},
    "signals": int(len(rows)),
    "fillable": nxt_ok,
    "unfillable": nxt_bad,
    "sens": {str(n): dump_row(sens[n]) for n in sens},
    "alts": {k: dump_row(v) for k, v in alt_m.items()},
    "exits": {k: dump_row(v) for k, v in exit_m.items()},
    "filters": {k: dump_row(v) for k, v in filt_m.items()},
    "s21": dump_row(m21),
    "years": {str(y): {"n": len(a), "win": float((np.array(a) > 0).mean()),
                       "avg": float(np.array(a).mean())} for y, a in yr.items()},
}
json.dump(summary, open(os.path.join(OUT, "jiukong_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

with open(os.path.join(RES, "jiukong_variants.csv"), "w", encoding="utf-8") as f:
    f.write("组,名称,笔数,胜率,平均单笔,中位数,盈亏比,平均持有,等权5年,年化,最大回撤,夏普\n")

    def w(group, name, m):
        pf = m["profit_factor"] if np.isfinite(m["profit_factor"]) else 0.0
        f.write(f"{group},{name},{m['trades']},{m['win_rate'] * 100:.2f}%,{m['avg_ret'] * 100:.4f}%,"
                f"{m['median_ret'] * 100:.4f}%,{pf:.3f},{m['avg_bars']:.2f},"
                f"{m['total_return'] * 100:.2f}%,{m['annual_return'] * 100:.2f}%,"
                f"{m['max_drawdown'] * 100:.2f}%,{m['sharpe']:.3f}\n")

    w("主口径", "A 九阴一阳 TP8 SL5 max10", main_m)
    for n, m in sens.items():
        w("连阴天数", f"{n}阴一阳", m)
    for k, m in alt_m.items():
        w("其它定义", k, m)
    for k, m in exit_m.items():
        w("出场", k, m)
    for k, m in filt_m.items():
        w("过滤", k, m)
    w("对照", "21 四连阴+RSI超卖", m21)

# 部分净值供作图
with open(os.path.join(RES, "jiukong_equity_alters.csv"), "w", encoding="utf-8") as f:
    f.write("日期,九阴一阳,TD低九当日,TD低九后一多\n")
    d0 = [d2s(d) for d in main_r["dates"]]
    e0 = main_r["equity_curve"]
    e_td = alt_eq["C TD低九当日（连续9日 C<C[4]）"][1]
    e_fl = alt_eq["D TD低九后一多（低九后首日 C>C[4]）"][1]
    for i in range(len(d0)):
        f.write(f"{d0[i]},{e0[i]:.6f},{e_td[i]:.6f},{e_fl[i]:.6f}\n")

P("")
P("已写出 jiukong_summary.json / results/jiukong_variants.csv / 策略 33 逐笔与净值")
P("=" * 100)
LOG.close()
