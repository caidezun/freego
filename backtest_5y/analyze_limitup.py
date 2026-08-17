"""连板票多维度画像：把每一次连板事件展开成特征表，再逐维度看胜率与单笔收益。

特征分四类：
  个股静态  流通市值、所属板块（主板/创业板科创板）、所属行业、上市时长
  个股动态  连板高度、连板累计涨幅、换手率与其相对自身的分位、量比、振幅、
            距 MA10/MA20 乖离、第三板是否一字板/收在最高、第四日开盘跳空
  行业活跃  同行业当日涨停家数、同行业当日平均涨幅、同行业成交额占全市场比重
  大盘活跃  全市场涨停家数、全市场成交额及其相对 20 日均值、上涨家数占比、
            指数当日涨幅、指数是否在 MA20 之上
结果列同时给出多种卖出口径的收益，供后面优化出场规则。
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/ubuntu/ashare5y"
NPZ = os.path.join(DATA, "all.npz")
START, END = 20210817, 20260817
FEE, SLIP = 0.5 / 1000, 0.5 / 1000
LOG = open(os.path.join(OUT, "limitup_analysis.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


# ─────────────────────────── 载入 ───────────────────────────
pnl = E.Panel(NPZ, skip_new=60)
ind = E.Indicators(pnl)
N, L, T = pnl.N, pnl.L, pnl.T
o, h, l, c, v, amt, chg, tr = (pnl.cf[k] for k in
                               ("open", "high", "low", "close", "volume", "amount", "chg", "turnover"))
meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
indu = json.load(open(os.path.join(DATA, "industry.json"), encoding="utf-8"))
fshares = np.array([float(meta.get(cd, {}).get("fshares") or 0) for cd in pnl.codes])
industry = np.array([indu.get(cd, {}).get("industry", "未知") for cd in pnl.codes])
board = np.array(["创业板科创板" if cd[:2] in ("30", "68") else "沪深主板" for cd in pnl.codes])

idx_close = {}
for sym, nm in (("sh000001", "上证指数"), ("sz399006", "创业板指")):
    rows = [ln.split(",") for ln in
            open(os.path.join(DATA, "index", sym + ".csv"), encoding="utf-8").read().strip().split("\n")[1:]]
    dates = np.array([int(r[0].replace("-", "")) for r in rows])
    idx_close[nm] = (dates, np.array([float(r[4]) for r in rows]))

ma5, ma10, ma20 = ind.get("ma5"), ind.get("ma10"), ind.get("ma20")
vma5 = ind.get("vma5")

# ─────────────────────────── 涨停与连板 ───────────────────────────
pct = E.limit_pct_of(pnl.codes)
with np.errstate(invalid="ignore"):
    lu = (chg >= pct[:, None]) & (c >= h - 1e-6)
lu = np.where(np.isnan(chg), False, lu)
streak = np.zeros((N, L), dtype=np.int16)
for j in range(L):
    streak[:, j] = np.where(lu[:, j], (streak[:, j - 1] if j else 0) + 1, 0)

# ─────────────── 市场与行业的日度聚合（按交易日历） ───────────────
mk_lu = np.zeros(T)
mk_amt = np.zeros(T)
mk_up = np.zeros(T)
mk_cnt = np.zeros(T)
ind_lu = {}
ind_chg_sum = {}
ind_cnt = {}
ind_amt = {}
uniq_ind = sorted(set(industry))
for name in uniq_ind:
    ind_lu[name] = np.zeros(T)
    ind_chg_sum[name] = np.zeros(T)
    ind_cnt[name] = np.zeros(T)
    ind_amt[name] = np.zeros(T)
for i in range(N):
    n = int(pnl.n_bars[i])
    if not n:
        continue
    cols = np.searchsorted(pnl.cal, pnl.dates[i, :n])
    good = ~np.isnan(chg[i, :n])
    cc = cols[good]
    mk_cnt[cc] += 1
    mk_amt[cc] += amt[i, :n][good]
    mk_up[cc] += (chg[i, :n][good] > 0)
    mk_lu[cc] += lu[i, :n][good]
    nm = industry[i]
    ind_cnt[nm][cc] += 1
    ind_amt[nm][cc] += amt[i, :n][good]
    ind_chg_sum[nm][cc] += chg[i, :n][good]
    ind_lu[nm][cc] += lu[i, :n][good]
mk_up_ratio = np.where(mk_cnt > 0, mk_up / np.maximum(mk_cnt, 1), np.nan)
mk_amt_ma20 = np.full(T, np.nan)
for t in range(19, T):
    mk_amt_ma20[t] = mk_amt[t - 19:t + 1].mean()
mk_lu_ma20 = np.full(T, np.nan)
for t in range(19, T):
    mk_lu_ma20[t] = mk_lu[t - 19:t + 1].mean()

idx_feat = {}
for nm, (dts, cl) in idx_close.items():
    col = np.searchsorted(pnl.cal, dts)
    close_t = np.full(T, np.nan)
    close_t[col] = cl
    ma = np.full(T, np.nan)
    for t in range(19, T):
        w = close_t[t - 19:t + 1]
        if not np.isnan(w).any():
            ma[t] = w.mean()
    ret1 = np.full(T, np.nan)
    ret5 = np.full(T, np.nan)
    ret1[1:] = close_t[1:] / close_t[:-1] - 1
    ret5[5:] = close_t[5:] / close_t[:-5] - 1
    idx_feat[nm] = {"close": close_t, "ma20": ma, "ret1": ret1, "ret5": ret5}

P("=" * 100)
P("连板票多维度分析：全A股 5404 只、2021-08-17 ~ 2026-08-17")
P(f"全市场日均涨停家数 {np.nanmean(mk_lu[np.searchsorted(pnl.cal, START):]):.0f} 家，"
  f"日均成交额 {np.nanmean(mk_amt[np.searchsorted(pnl.cal, START):]) / 1e8:.0f} 亿元")
P("=" * 100)

# ─────────────────────────── 事件表 ───────────────────────────
FWD = 20
recs = []
i0c, i1c = pnl.col_range(START, END)
with np.errstate(invalid="ignore"):
    untradable = (o == c) & (c == h) & (h == l) & (np.abs(chg) > 9.5)
for i in range(N):
    n = int(pnl.n_bars[i])
    if n < 60:
        continue
    cols = np.searchsorted(pnl.cal, pnl.dates[i, :n])
    for k in range(5, n - 2):
        s = int(streak[i, k])
        if s < 2 or s > 8:
            continue
        if streak[i, k + 1] if k + 1 < n else 0:      # 只在「连板刚好 s 天」这一天记录
            pass
        t = int(cols[k])
        if not (i0c <= t < i1c):
            continue
        k4 = k + 1
        if untradable[i, k4]:
            continue                                   # 第四日一字板买不到
        entry = o[i, k4] * (1 + SLIP)
        if not (entry > 0):
            continue
        tcol = t
        nm = industry[i]
        base_c = c[i, k - s] if k - s >= 0 else c[i, 0]  # 连板启动前一日收盘
        fw = {}
        for step in (1, 2, 3, 5, 10, 20):
            kk = k4 + step
            fw[f"fwd_open{step}"] = (o[i, kk] * (1 - SLIP) / entry - 1) if kk < n else np.nan
        # 基准出场：收盘跌破 MA10 → 次日开盘
        def exit_by(cond_arr, maxn=120):
            for kk in range(k4 + 1, min(n - 1, k4 + maxn)):
                if cond_arr[i, kk] == cond_arr[i, kk] and c[i, kk] < cond_arr[i, kk]:
                    if kk + 1 < n:
                        return (o[i, kk + 1] * (1 - SLIP) / entry - 1), kk + 1 - k4
                    break
            kk = min(n - 1, k4 + maxn)
            return (c[i, kk] / entry - 1), kk - k4
        r10, b10 = exit_by(ma10)
        r5, b5 = exit_by(ma5)
        # 移动止盈：从持仓期最高收盘回撤 8% 就走
        rtr, btr = np.nan, np.nan
        peak = -1e18
        for kk in range(k4, min(n - 1, k4 + 120)):
            peak = max(peak, c[i, kk])
            if kk > k4 and c[i, kk] < peak * 0.92:
                rtr, btr = o[i, kk + 1] * (1 - SLIP) / entry - 1, kk + 1 - k4
                break
        if rtr != rtr:
            kk = min(n - 1, k4 + 120)
            rtr, btr = c[i, kk] / entry - 1, kk - k4
        rec = {
            "code": pnl.codes[i], "name": pnl.names[i], "date3": int(pnl.dates[i, k]),
            "date4": int(pnl.dates[i, k4]), "boards": s, "board": board[i], "industry": nm,
            "entry": entry,
            "mktcap": (c[i, k] * fshares[i]) if fshares[i] > 0 else np.nan,
            "listed_bars": k,
            "run_up": (c[i, k] / base_c - 1) if base_c > 0 else np.nan,
            "turnover3": tr[i, k],
            "turnover_rel": tr[i, k] / np.nanmean(tr[i, max(0, k - 20):k]) if k >= 5 else np.nan,
            "vol_ratio": v[i, k] / vma5[i, k - 1] if k >= 6 and vma5[i, k - 1] > 0 else np.nan,
            "amp3": pnl.cf["amp"][i, k],
            "bias10": c[i, k] / ma10[i, k] - 1 if ma10[i, k] == ma10[i, k] else np.nan,
            "bias20": c[i, k] / ma20[i, k] - 1 if ma20[i, k] == ma20[i, k] else np.nan,
            "amount3": amt[i, k],
            "yizi3": bool((o[i, k] == c[i, k]) and (c[i, k] == l[i, k])),
            "gap4": o[i, k4] / c[i, k] - 1,
            "lu60": float(np.sum(lu[i, max(0, k - 59):k + 1])),
            "ind_lu": ind_lu[nm][tcol], "ind_chg": (ind_chg_sum[nm][tcol] / max(1, ind_cnt[nm][tcol])),
            "ind_amt_share": ind_amt[nm][tcol] / mk_amt[tcol] if mk_amt[tcol] > 0 else np.nan,
            "mkt_lu": mk_lu[tcol], "mkt_lu_rel": mk_lu[tcol] / mk_lu_ma20[tcol] if mk_lu_ma20[tcol] else np.nan,
            "mkt_amt": mk_amt[tcol], "mkt_amt_rel": mk_amt[tcol] / mk_amt_ma20[tcol] if mk_amt_ma20[tcol] else np.nan,
            "mkt_up_ratio": mk_up_ratio[tcol],
            "idx_ret1": idx_feat["上证指数"]["ret1"][tcol], "idx_ret5": idx_feat["上证指数"]["ret5"][tcol],
            "idx_above_ma20": float(idx_feat["上证指数"]["close"][tcol] > idx_feat["上证指数"]["ma20"][tcol]),
            "gem_ret5": idx_feat["创业板指"]["ret5"][tcol],
            "ret_ma10": r10 - 2 * FEE, "bars_ma10": b10,
            "ret_ma5": r5 - 2 * FEE, "bars_ma5": b5,
            "ret_trail8": rtr - 2 * FEE, "bars_trail8": btr,
        }
        for k2, v2 in fw.items():
            rec[k2] = v2 - 2 * FEE if v2 == v2 else np.nan
        recs.append(rec)

P(f"事件表：{len(recs):,} 条（连板 2~8 板、第四日可成交），"
  f"其中 3 板起 {sum(1 for r in recs if r['boards'] >= 3):,} 条")
keys = list(recs[0].keys())
with open(os.path.join(OUT, "limitup_events.csv"), "w", encoding="utf-8") as f:
    f.write(",".join(keys) + "\n")
    for r in recs:
        f.write(",".join("" if (isinstance(r[k], float) and r[k] != r[k]) else str(r[k]) for k in keys) + "\n")
P(f"已保存 limitup_events.csv（{len(keys)} 列）")

# ─────────────────────────── 分维度统计 ───────────────────────────
A = {k: np.array([r[k] for r in recs], dtype=object if isinstance(recs[0][k], str) else float)
     for k in keys if k not in ("code", "name")}
for k in ("code", "name", "board", "industry"):
    A[k] = np.array([r[k] for r in recs])
sel3 = np.array([r["boards"] == 3 for r in recs])
Y = A["ret_ma10"]


def stat(mask, y=None):
    y = Y if y is None else y
    m = mask & ~np.isnan(y)
    if m.sum() < 20:
        return None
    a = y[m]
    return dict(n=int(m.sum()), win=float((a > 0).mean()), mean=float(a.mean()),
                med=float(np.median(a)), tail=float((a > 0.2).mean()))


def show_groups(title, groups, y=None):
    P("")
    P(f"── {title}")
    P("   " + f"{'分组':<26}{'笔数':>7}{'胜率':>9}{'平均':>9}{'中位数':>9}{'>20%占比':>10}")
    for lbl, m in groups:
        s = stat(m, y)
        if s:
            P("   " + f"{lbl:<24}{s['n']:>7,}{s['win'] * 100:>8.1f}%{s['mean'] * 100:>8.2f}%"
                      f"{s['med'] * 100:>8.2f}%{s['tail'] * 100:>9.1f}%")


def qgroups(name, mask, nq=5, fmt=lambda x: f"{x:.2f}"):
    x = A[name]
    m = mask & ~np.isnan(x)
    qs = np.nanpercentile(x[m], np.linspace(0, 100, nq + 1))
    out = []
    for i in range(nq):
        lo, hi = qs[i], qs[i + 1]
        g = m & (x >= lo) & ((x <= hi) if i == nq - 1 else (x < hi))
        out.append((f"Q{i + 1} [{fmt(lo)},{fmt(hi)}]", g))
    return out


P("")
P("#" * 100)
P("一、买入维度：先看「买第几板」——这是最基础的入场参数")
show_groups("连板高度（在第 N 板收盘后、第 N+1 日开盘买入）",
            [(f"{n} 板后买入", A["boards"] == n) for n in range(2, 8)])

P("")
P("#" * 100)
P("二、以下均只统计「3 板后买入」这一子集（用户策略），基准出场=跌破MA10")
base_stat = stat(sel3)
P(f"   基准：{base_stat['n']:,} 笔，胜率 {base_stat['win'] * 100:.1f}%，"
  f"平均 {base_stat['mean'] * 100:.2f}%，中位数 {base_stat['med'] * 100:.2f}%")
show_groups("流通市值（亿元）", qgroups("mktcap", sel3, 5, lambda x: f"{x / 1e8:.0f}亿"))
show_groups("板块", [(b, sel3 & (A["board"] == b)) for b in ("沪深主板", "创业板科创板")])
show_groups("第三板换手率(%)", qgroups("turnover3", sel3, 5))
show_groups("换手率/自身20日均值", qgroups("turnover_rel", sel3, 5))
show_groups("量比（当日量/5日均量）", qgroups("vol_ratio", sel3, 5))
show_groups("第三板成交额（亿元）", qgroups("amount3", sel3, 5, lambda x: f"{x / 1e8:.1f}亿"))
show_groups("三板累计涨幅", qgroups("run_up", sel3, 5, lambda x: f"{x * 100:.0f}%"))
show_groups("距MA10乖离", qgroups("bias10", sel3, 5, lambda x: f"{x * 100:.0f}%"))
show_groups("第四日开盘跳空", qgroups("gap4", sel3, 6, lambda x: f"{x * 100:+.1f}%"))
show_groups("第三板是否一字板", [("一字板(开=收=最低)", sel3 & (A["yizi3"] > 0)),
                                ("非一字板", sel3 & (A["yizi3"] == 0))])
show_groups("上市时长", [("上市<250日(次新)", sel3 & (A["listed_bars"] < 250)),
                        ("250~750日", sel3 & (A["listed_bars"] >= 250) & (A["listed_bars"] < 750)),
                        (">=750日", sel3 & (A["listed_bars"] >= 750))])
show_groups("近60日涨停次数", [("<=4次", sel3 & (A["lu60"] <= 4)), ("5~8次", sel3 & (A["lu60"] > 4) & (A["lu60"] <= 8)),
                              (">8次", sel3 & (A["lu60"] > 8))])
P("")
P("#" * 100)
P("三、行业与大盘活跃度")
show_groups("同行业当日涨停家数", [("0~1家", sel3 & (A["ind_lu"] <= 1)), ("2~3家", sel3 & (A["ind_lu"] > 1) & (A["ind_lu"] <= 3)),
                                  ("4~6家", sel3 & (A["ind_lu"] > 3) & (A["ind_lu"] <= 6)), (">6家", sel3 & (A["ind_lu"] > 6))])
show_groups("同行业当日平均涨幅", qgroups("ind_chg", sel3, 5, lambda x: f"{x:+.1f}%"))
show_groups("行业成交额占全市场比", qgroups("ind_amt_share", sel3, 5, lambda x: f"{x * 100:.1f}%"))
show_groups("全市场当日涨停家数", qgroups("mkt_lu", sel3, 5, lambda x: f"{x:.0f}"))
show_groups("全市场涨停家数/20日均值", qgroups("mkt_lu_rel", sel3, 5))
show_groups("全市场成交额（万亿）", qgroups("mkt_amt", sel3, 5, lambda x: f"{x / 1e12:.2f}"))
show_groups("全市场成交额/20日均值", qgroups("mkt_amt_rel", sel3, 5))
show_groups("上涨家数占比", qgroups("mkt_up_ratio", sel3, 5, lambda x: f"{x * 100:.0f}%"))
show_groups("上证指数当日涨幅", qgroups("idx_ret1", sel3, 5, lambda x: f"{x * 100:+.2f}%"))
show_groups("上证指数近5日涨幅", qgroups("idx_ret5", sel3, 5, lambda x: f"{x * 100:+.1f}%"))
show_groups("上证指数是否在MA20之上", [("在MA20之上", sel3 & (A["idx_above_ma20"] > 0)),
                                      ("在MA20之下", sel3 & (A["idx_above_ma20"] == 0))])

P("")
P("#" * 100)
P("四、卖出维度：同一批 3 板买入的交易，换不同出场规则")
for lbl, key, bars in (("跌破MA10（原策略）", "ret_ma10", "bars_ma10"),
                       ("跌破MA5", "ret_ma5", "bars_ma5"),
                       ("最高收盘回撤8%", "ret_trail8", "bars_trail8"),
                       ("次日开盘就卖", "fwd_open1", None),
                       ("持有2日开盘卖", "fwd_open2", None),
                       ("持有3日开盘卖", "fwd_open3", None),
                       ("持有5日开盘卖", "fwd_open5", None),
                       ("持有10日开盘卖", "fwd_open10", None),
                       ("持有20日开盘卖", "fwd_open20", None)):
    s = stat(sel3, A[key])
    bb = f"{np.nanmean(A[bars][sel3]):.1f}日" if bars else lbl.replace("持有", "").replace("日开盘卖", "日")
    if s:
        P("   " + f"{lbl:<20}{s['n']:>7,}{s['win'] * 100:>8.1f}%{s['mean'] * 100:>8.2f}%"
                  f"{s['med'] * 100:>8.2f}%   平均持有 {bb}")
LOG.close()
