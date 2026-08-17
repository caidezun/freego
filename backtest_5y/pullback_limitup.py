"""结构性不同的入场思路：连板之后不追高，等第一次回调再买。

对每一次 2/3 连板事件：
  · 追板（对照）：第 N+1 日开盘买
  · 回调买：连板后等到第一根收盘跌破 MA5 的K线，次日开盘买（最多等 15 个交易日）
  · 回调且守住 MA10 才买：上面基础上要求回调日收盘仍 ≥ MA10（浪费掉直接破位的票）
出场统一测四种：持有3日 / 持有5日 / 跌破MA10 / 最高收盘回撤8%
"""
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
NPZ = "/home/ubuntu/ashare5y/all.npz"
START, END = 20210817, 20260817
FEE = SLIP = 0.5 / 1000
LOG = open(os.path.join(OUT, "limitup_pullback.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
o, h, l, c, chg = (pnl.cf[k] for k in ("open", "high", "low", "close", "chg"))
amt = pnl.cf["amount"]
ma5, ma10 = ind.get("ma5"), ind.get("ma10")
pct = E.limit_pct_of(pnl.codes)
with np.errstate(invalid="ignore"):
    lu = (chg >= pct[:, None]) & (c >= h - 1e-6)
    untr = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
lu = np.where(np.isnan(chg), False, lu)
streak = np.zeros(lu.shape, dtype=np.int16)
for j in range(lu.shape[1]):
    streak[:, j] = np.where(lu[:, j], (streak[:, j - 1] if j else 0) + 1, 0)
i0, i1 = pnl.col_range(START, END)


def exits(i, kb, n):
    """给定买入bar kb，返回四种出场的净收益"""
    entry = o[i, kb] * (1 + SLIP)
    out = {}
    for tag, step in (("持有3日", 3), ("持有5日", 5)):
        kk = kb + step
        out[tag] = (o[i, kk] * (1 - SLIP) / entry - 1 - 2 * FEE) if kk < n else np.nan
    for tag, ref in (("跌破MA10", ma10),):
        r = np.nan
        for kk in range(kb + 1, min(n - 1, kb + 120)):
            if ref[i, kk] == ref[i, kk] and c[i, kk] < ref[i, kk]:
                r = o[i, kk + 1] * (1 - SLIP) / entry - 1 - 2 * FEE
                break
        if r != r:
            kk = min(n - 1, kb + 120)
            r = c[i, kk] / entry - 1 - 2 * FEE
        out[tag] = r
    peak, r = -1e18, np.nan
    for kk in range(kb, min(n - 1, kb + 120)):
        peak = max(peak, c[i, kk])
        if kk > kb and c[i, kk] < peak * 0.92:
            r = o[i, kk + 1] * (1 - SLIP) / entry - 1 - 2 * FEE
            break
    if r != r:
        kk = min(n - 1, kb + 120)
        r = c[i, kk] / entry - 1 - 2 * FEE
    out["回撤8%"] = r
    return out


groups = defaultdict(lambda: defaultdict(list))
for i in range(pnl.N):
    n = int(pnl.n_bars[i])
    if n < 60:
        continue
    cols = np.searchsorted(pnl.cal, pnl.dates[i, :n])
    for k in range(5, n - 6):
        s = int(streak[i, k])
        if s not in (2, 3):
            continue
        if not (i0 <= int(cols[k]) < i1):
            continue
        big = amt[i, k] >= 8e8
        tags = ("全部", "成交额≥8亿" if big else "成交额<8亿")
        # 追板
        if not untr[i, k + 1]:
            ex = exits(i, k + 1, n)
            for g in tags:
                for tag, v in ex.items():
                    groups[(s, "追板：第N+1日开盘买", g)][tag].append(v)
        # 回调买
        kp = None
        for kk in range(k + 1, min(n - 2, k + 16)):
            if ma5[i, kk] == ma5[i, kk] and c[i, kk] < ma5[i, kk]:
                kp = kk
                break
        if kp is not None and not untr[i, kp + 1]:
            ex = exits(i, kp + 1, n)
            hold10 = ma10[i, kp] == ma10[i, kp] and c[i, kp] >= ma10[i, kp]
            for g in tags:
                for tag, v in ex.items():
                    groups[(s, "回调买：首破MA5后次日开盘买", g)][tag].append(v)
                if hold10:
                    for tag, v in ex.items():
                        groups[(s, "回调且守住MA10才买", g)][tag].append(v)

P("=" * 104)
P("连板后「追板」与「等回调」的对比    全A股 2021-08-17 ~ 2026-08-17，成本双边 1‰ + 滑点 1‰")
P("=" * 104)
for big in ("全部", "成交额<8亿", "成交额≥8亿"):
    P("")
    P(f"■ 样本：{big}")
    P("   " + f"{'板数':>4}{'入场方式':<28}{'出场':<10}{'笔数':>7}{'胜率':>8}{'平均单笔':>10}{'中位数':>9}")
    for s in (2, 3):
        for way in ("追板：第N+1日开盘买", "回调买：首破MA5后次日开盘买", "回调且守住MA10才买"):
            g = groups.get((s, way, big))
            if not g:
                continue
            for tag in ("持有3日", "持有5日", "跌破MA10", "回撤8%"):
                a = np.array([x for x in g[tag] if x == x])
                if len(a) < 50:
                    continue
                P("   " + f"{s:>4}{way:<26}{tag:<10}{len(a):>7,}{(a > 0).mean() * 100:>7.1f}%"
                          f"{a.mean() * 100:>+9.2f}%{np.median(a) * 100:>+8.2f}%")
LOG.close()

# ── 推荐版本的逐笔明细：2板 + 首破MA5回调买 + 持有3日（不加成交额过滤，样本更大更稳） ──
recs = []
for i in range(pnl.N):
    n = int(pnl.n_bars[i])
    if n < 60:
        continue
    cols = np.searchsorted(pnl.cal, pnl.dates[i, :n])
    for k in range(5, n - 6):
        if int(streak[i, k]) != 2 or not (i0 <= int(cols[k]) < i1):
            continue
        kp = None
        for kk in range(k + 1, min(n - 2, k + 16)):
            if ma5[i, kk] == ma5[i, kk] and c[i, kk] < ma5[i, kk]:
                kp = kk
                break
        if kp is None or untr[i, kp + 1]:
            continue
        kb = kp + 1
        kx = kb + 3
        if kx >= n:
            continue
        entry = o[i, kb] * (1 + SLIP)
        exit_px = o[i, kx] * (1 - SLIP)
        recs.append((int(pnl.dates[i, k]), pnl.codes[i], pnl.names[i], int(pnl.dates[i, kb]),
                     entry, int(pnl.dates[i, kx]), exit_px, exit_px / entry - 1 - 2 * FEE,
                     kb - k, amt[i, k] / 1e8))
recs.sort()
with open(os.path.join(OUT, "results", "S12_2板回调买入_trades.csv"), "w", encoding="utf-8") as f:
    f.write("序号,代码,名称,第二板日期,买入日,买入价,卖出日,卖出价,净收益率,回调等待天数,第二板成交额(亿)\n")
    for k, r in enumerate(recs, 1):
        f.write(f"{k},{r[1]},{r[2]},{r[0]},{r[3]},{r[4]:.3f},{r[5]},{r[6]:.3f},"
                f"{r[7] * 100:.4f}%,{r[8]},{r[9]:.2f}\n")
rr = np.array([r[7] for r in recs])
print(f"\n推荐版本逐笔明细已保存：results/S12_2板回调买入_trades.csv  {len(recs)} 笔，"
      f"胜率 {(rr > 0).mean() * 100:.2f}%，平均单笔 {rr.mean() * 100:+.2f}%，中位数 {np.median(rr) * 100:+.2f}%")
