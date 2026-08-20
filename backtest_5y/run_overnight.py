"""隔夜持股法：尾盘（收盘价近似）买入，次日开盘卖出。

网上公开方案的可量化部分（日线能表达的），在全 A 股 5 年上逐条对照。
日线近似：买入=当日收盘价，卖出=次一交易日开盘价。分时均价线 / 14:30 新高无法回测，文中单独说明。
"""
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
NPZ = os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz")
DATA = "/home/ubuntu/ashare5y"
START = int(os.environ.get("START", "20210817"))
END = int(os.environ.get("END", "20260817"))
FEE = SLIP = 0.5 / 1000
LOG = open(os.path.join(OUT, "overnight_run.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def pct(x):
    return f"{x * 100:+.2f}%"


P("=" * 100)
P("隔夜持股法回测：尾盘买入（收盘价）→ 次日开盘卖出")
P("样本窗口 2021-08-17 ~ 2026-08-17；成本 手续费 0.5‰ + 滑点 0.5‰ 双边")
P("=" * 100)

pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
i0, i1 = pnl.col_range(START, END)
N, L, T = pnl.N, pnl.L, pnl.T
o, h, l, c, v, amt, chg, tr = (pnl.cf[k] for k in
                               ("open", "high", "low", "close", "volume", "amount", "chg", "turnover"))
ma5, ma10, ma20 = ind.get("ma5"), ind.get("ma10"), ind.get("ma20")
vma5 = ind.get("vma5")
lu4 = ind.get("lu4")
pct_lim = E.limit_pct_of(pnl.codes)
codes = np.array([str(x) for x in pnl.codes])
main = np.array([cd[:2] not in ("30", "68") and cd[0] not in ("4", "8") for cd in codes])
gem = np.array([cd[:2] in ("30", "68") for cd in codes])

with np.errstate(invalid="ignore", divide="ignore"):
    fcap = np.where(tr > 0, v * 1e4 / tr * c, np.nan)
    volr = v / vma5
    upper_sh = (h - c) / (h - l)
    untr = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
    lu = (chg >= pct_lim[:, None]) & (c >= h - 1e-6)
lu = np.where(np.isnan(chg), False, lu)
untr = np.where(np.isnan(chg), False, untr)

idx_rows = [ln.split(",") for ln in
            open(os.path.join(DATA, "index", "sh000001.csv"), encoding="utf-8").read().strip().split("\n")[1:]]
idx_d = np.array([int(r[0].replace("-", "")) for r in idx_rows])
idx_c = np.array([float(r[4]) for r in idx_rows], dtype=np.float64)
idx_map = {int(d): float(x) for d, x in zip(idx_d, idx_c)}
idx_on_cal = np.array([idx_map.get(int(d), np.nan) for d in pnl.cal])
idx_ma20 = np.full(T, np.nan)
for t in range(19, T):
    w = idx_on_cal[t - 19:t + 1]
    if np.isfinite(w).all():
        idx_ma20[t] = w.mean()
idx_above = idx_on_cal > idx_ma20

P(f"样本：{N} 只、{int(pnl.n_bars.sum())} 根日线，区间 {d2s(START)} ~ {d2s(END)}（{i1 - i0} 个交易日）")


def gather(filter_fn, topk=0, need_idx=False, keep=True):
    """按交易日历收集隔夜交易。filter_fn(idx, k0) -> bool mask。
    topk=0：当日全部候选等权；topk>0：按成交额取前 k 只。
    keep=False 时不保留逐笔（用于全市场无过滤）。
    """
    recs = []
    day_mean = []  # (date, mean_net_open, n)
    all_net_o, all_g_o, all_net_c, all_g_h, all_d = [], [], [], [], []
    for t in range(i0, i1 - 1):
        b0 = pnl.bar[:, t]
        b1 = pnl.bar[:, t + 1]
        ok = (b0 >= 0) & (b1 >= 0)
        if need_idx:
            if not np.isfinite(idx_above[t]) or not bool(idx_above[t]):
                continue
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        k0 = b0[idx]
        k1 = b1[idx]
        tradable = ~untr[idx, k0] & ~untr[idx, k1]
        sel = tradable & filter_fn(idx, k0)
        if not np.any(sel):
            continue
        ii = idx[sel]
        a0, a1 = k0[sel], k1[sel]
        if topk:
            am = amt[ii, a0]
            order = np.argsort(-am)[: min(topk, len(ii))]
            ii, a0, a1 = ii[order], a0[order], a1[order]
        entry = c[ii, a0] * (1 + SLIP)
        px_o = o[ii, a1] * (1 - SLIP)
        px_c = c[ii, a1] * (1 - SLIP)
        g_o = o[ii, a1] / c[ii, a0] - 1.0
        n_o = px_o / entry - 1.0 - 2 * FEE
        n_c = px_c / entry - 1.0 - 2 * FEE
        g_h = h[ii, a1] / c[ii, a0] - 1.0
        day = int(pnl.cal[t])
        all_net_o.append(n_o)
        all_g_o.append(g_o)
        all_net_c.append(n_c)
        all_g_h.append(g_h)
        all_d.append(np.full(len(n_o), day, dtype=np.int32))
        day_mean.append((day, float(n_o.mean()), int(len(n_o))))
        if keep:
            amv = amt[ii, a0]
            for j in range(len(ii)):
                recs.append((day, int(ii[j]), float(n_o[j]), float(g_o[j]),
                             float(n_c[j]), float(g_h[j]), float(amv[j])))
    pack = {
        "net_o": np.concatenate(all_net_o) if all_net_o else np.array([]),
        "g_o": np.concatenate(all_g_o) if all_g_o else np.array([]),
        "net_c": np.concatenate(all_net_c) if all_net_c else np.array([]),
        "g_h": np.concatenate(all_g_h) if all_g_h else np.array([]),
        "date": np.concatenate(all_d) if all_d else np.array([], dtype=np.int32),
        "days": day_mean, "recs": recs,
    }
    return pack


def summarize_arr(r, days, label):
    if r is None or len(r) == 0 or not days:
        P(f"   {label:<36}  无成交")
        return None
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    curve = []
    for d, m, n in days:
        eq *= 1 + m
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak if peak else 0)
        curve.append((d, eq))
    wins = r[r > 0]
    loss = r[r <= 0]
    ndays = len(days)
    out = {
        "label": label, "n": int(len(r)), "days": int(ndays),
        "per_day": len(r) / max(1, ndays),
        "win": float((r > 0).mean()),
        "mean": float(r.mean()), "med": float(np.median(r)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(loss.mean()) if len(loss) else 0.0,
        "pf": float(wins.sum() / -loss.sum()) if len(loss) and loss.sum() < 0 else float("inf"),
        "stack": float(eq - 1), "mdd": float(mdd),
        "best": float(r.max()), "worst": float(r.min()),
    }
    P(f"   {label:<36} {out['n']:>8,}笔/{out['days']:>4}日  日均{out['per_day']:>6.1f}  "
      f"胜率{out['win'] * 100:5.1f}%  平均{out['mean'] * 100:+6.2f}%  中位{out['med'] * 100:+6.2f}%  "
      f"盈亏比{out['pf']:4.2f}  等权累计{out['stack'] * 100:+8.1f}%  回撤{out['mdd'] * 100:5.1f}%")
    return out, curve


KEYMAP = {2: "net_o", 3: "g_o", 4: "net_c", 5: "g_h"}


# ── 过滤函数 ──
def F_all(idx, k0):
    return np.isfinite(c[idx, k0]) & np.isfinite(o[idx, k0])


def F_up3_5(idx, k0):
    x = chg[idx, k0]
    return (x >= 3) & (x <= 5)


def F_classic(idx, k0):
    """网上最常见可量化组合：涨幅3-5 + 换手5-10 + 市值50-200亿 + 量比>1"""
    return (F_up3_5(idx, k0)
            & (tr[idx, k0] >= 5) & (tr[idx, k0] <= 10)
            & (fcap[idx, k0] >= 50e8) & (fcap[idx, k0] <= 200e8)
            & (volr[idx, k0] > 1))


def F_sixstep(idx, k0):
    """同花顺六步的日线近似：3-5% + 市值<200亿 + 量比>1 + 换手5-10 + 近4日有涨停"""
    return (F_up3_5(idx, k0)
            & (fcap[idx, k0] > 0) & (fcap[idx, k0] < 200e8)
            & (volr[idx, k0] > 1)
            & (tr[idx, k0] >= 5) & (tr[idx, k0] <= 10)
            & (lu4[idx, k0] >= 1))


def F_eight(idx, k0):
    """八步可量化：3-5% + 换手5-15 + 市值50-200 + 量比>1 + 站上MA5/10/20 + 非长上影"""
    cc = c[idx, k0]
    return (F_up3_5(idx, k0)
            & (tr[idx, k0] >= 5) & (tr[idx, k0] <= 15)
            & (fcap[idx, k0] >= 50e8) & (fcap[idx, k0] <= 200e8)
            & (volr[idx, k0] > 1)
            & (cc > ma5[idx, k0]) & (cc > ma10[idx, k0]) & (cc > ma20[idx, k0])
            & (ma5[idx, k0] > ma10[idx, k0]) & (ma10[idx, k0] > ma20[idx, k0])
            & (upper_sh[idx, k0] <= 0.4))


def F_claw(idx, k0):
    """ClawHub 主板隔夜：主板 + 涨幅1-5 + 额>1亿 + 市值50-500 + 换手3-10 + 价>5 + 量比0.8-2"""
    return (main[idx]
            & (chg[idx, k0] >= 1) & (chg[idx, k0] <= 5)
            & (amt[idx, k0] >= 1e8)
            & (fcap[idx, k0] >= 50e8) & (fcap[idx, k0] <= 500e8)
            & (tr[idx, k0] >= 3) & (tr[idx, k0] <= 10)
            & (c[idx, k0] > 5)
            & (volr[idx, k0] >= 0.8) & (volr[idx, k0] <= 2.0))


def F_main_35(idx, k0):
    return main[idx] & F_up3_5(idx, k0)


def F_down(idx, k0):
    """对照：当日跌超 3%（学术上隔夜更负的一侧）"""
    return chg[idx, k0] <= -3


schemes = [
    ("S0 无过滤全市场", F_all, 0, False, False),
    ("S1 仅涨幅3%~5%", F_up3_5, 0, False, True),
    ("S2 经典四条件（3-5/换手/市值/量比）", F_classic, 0, False, True),
    ("S3 六步日线近似（含涨停记忆）", F_sixstep, 0, False, True),
    ("S4 八步日线近似（均线多头+非长上影）", F_eight, 0, False, True),
    ("S5 ClawHub主板隔夜", F_claw, 0, False, True),
    ("S6 主板仅3%~5%", F_main_35, 0, False, False),
    ("S7 对照：当日跌超3%", F_down, 0, False, False),
    ("S2b 经典四条件 + 指数在MA20上", F_classic, 0, True, False),
    ("S2c 经典四条件 每日成交额Top1", F_classic, 1, False, True),
    ("S2d 经典四条件 每日成交额Top5", F_classic, 5, False, False),
    ("S5b ClawHub 每日成交额Top1", F_claw, 1, False, True),
]

P("")
P("一、次日开盘卖（网上铁律，含成本）")
results = {}
store = {}
for name, fn, topk, need_idx, keep in schemes:
    pack = gather(fn, topk=topk, need_idx=need_idx, keep=keep)
    sm = summarize_arr(pack["net_o"], pack["days"], name)
    if sm:
        results[name] = sm[0]
        results[name]["_curve"] = sm[1]
        store[name] = pack

# 毛收益（无成本）对照
P("")
P("二、同一批交易的毛/净对照，以及「次日收盘卖 / 次日最高价卖」上界")
for name in ("S0 无过滤全市场", "S2 经典四条件（3-5/换手/市值/量比）", "S3 六步日线近似（含涨停记忆）",
             "S4 八步日线近似（均线多头+非长上影）", "S5 ClawHub主板隔夜"):
    pack = store.get(name)
    if not pack:
        continue
    P(f"   【{name}】")
    # 日均曲线仍按开盘净的交易日对齐没有意义；这里只报单笔统计
    for arr, lbl in ((pack["g_o"], "次日开盘卖 · 毛（无成本）"),
                     (pack["net_o"], "次日开盘卖 · 净（含成本）"),
                     (pack["net_c"], "次日收盘卖 · 净"),
                     (pack["g_h"], "次日最高价卖 · 毛（无法成交的上界）")):
        if len(arr) == 0:
            continue
        wins = arr[arr > 0]
        loss = arr[arr <= 0]
        pf = float(wins.sum() / -loss.sum()) if len(loss) and loss.sum() < 0 else float("inf")
        P(f"     {lbl:<28} {len(arr):>8,}笔  胜率{(arr > 0).mean() * 100:5.1f}%  "
          f"平均{arr.mean() * 100:+6.2f}%  中位{np.median(arr) * 100:+6.2f}%  盈亏比{pf:4.2f}")

# 分年：主方案
P("")
P("三、分年（次日开盘净收益，逐笔）")
for name in ("S0 无过滤全市场", "S2 经典四条件（3-5/换手/市值/量比）",
             "S2c 经典四条件 每日成交额Top1", "S5 ClawHub主板隔夜"):
    pack = store.get(name)
    if not pack or len(pack["net_o"]) == 0:
        continue
    P(f"   【{name}】")
    yr = pack["date"] // 10000
    P("      " + f"{'年':<6}{'笔数':>8}{'胜率':>8}{'平均':>9}{'中位':>9}")
    for y in sorted(set(yr.tolist())):
        a = pack["net_o"][yr == y]
        P("      " + f"{y:<6}{len(a):>8,}{(a > 0).mean() * 100:>7.1f}%{a.mean() * 100:>+8.2f}%{np.median(a) * 100:>+8.2f}%")

# 落盘：S2 全候选 + S2c Top1
def dump_trades(recs, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("买入日,代码,名称,开盘净收益,开盘毛收益,收盘净收益,最高毛收益,信号日成交额\n")
        for x in recs:
            i = x[1]
            f.write(f"{d2s(x[0])},{pnl.codes[i]},{pnl.names[i]},{x[2] * 100:.4f}%,{x[3] * 100:.4f}%,"
                    f"{x[4] * 100:.4f}%,{x[5] * 100:.4f}%,{x[6]:.0f}\n")


dump_trades(store["S2 经典四条件（3-5/换手/市值/量比）"]["recs"],
            os.path.join(RES, "overnight_S2_classic_trades.csv"))
dump_trades(store["S2c 经典四条件 每日成交额Top1"]["recs"],
            os.path.join(RES, "overnight_S2c_top1_trades.csv"))

# 净值曲线 CSV（S0 / S2 / S2c / S5）
with open(os.path.join(RES, "overnight_equity.csv"), "w", encoding="utf-8") as f:
    f.write("方案,日期,净值\n")
    for name in ("S0 无过滤全市场", "S2 经典四条件（3-5/换手/市值/量比）",
                 "S2c 经典四条件 每日成交额Top1", "S5 ClawHub主板隔夜"):
        if name not in results:
            continue
        for d, eq in results[name]["_curve"]:
            f.write(f"{name},{d2s(d)},{eq:.6f}\n")

# JSON：去掉曲线
js = {k: {kk: vv for kk, vv in v.items() if kk != "_curve"} for k, v in results.items()}
json.dump(js, open(os.path.join(OUT, "overnight_summary.json"), "w"), ensure_ascii=False, indent=2)

P("")
P("已保存 overnight_summary.json / results/overnight_*.csv")
LOG.close()
print("done")
