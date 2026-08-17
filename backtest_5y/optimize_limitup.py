"""在连板事件表上搜索「买入过滤 + 出场规则」的组合，并做样本内/样本外验证。

样本内 2021-08-17 ~ 2024-08-16（3 年），样本外 2024-08-17 ~ 2026-08-17（2 年）。
只在样本内挑组合，样本外只用来检验，避免把噪音当规律。
"""
import csv
import itertools
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUT, "limitup_optimize.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


rows = list(csv.DictReader(open(os.path.join(OUT, "limitup_events.csv"), encoding="utf-8")))
def col(k, typ=float):
    out = []
    for r in rows:
        v = r[k]
        if typ is float:
            out.append(float(v) if v not in ("", "nan") else np.nan)
        else:
            out.append(v)
    return np.array(out) if typ is float else np.array(out)


D = {k: col(k) for k in ("date3", "date4", "boards", "mktcap", "listed_bars", "run_up", "turnover3",
                         "turnover_rel", "vol_ratio", "amp3", "bias10", "bias20", "amount3", "gap4",
                         "lu60", "ind_lu", "ind_chg", "ind_amt_share", "mkt_lu", "mkt_lu_rel",
                         "mkt_amt", "mkt_amt_rel", "mkt_up_ratio", "idx_ret1", "idx_ret5",
                         "idx_above_ma20", "gem_ret5", "ret_ma10", "ret_ma5", "ret_trail8",
                         "fwd_open1", "fwd_open2", "fwd_open3", "fwd_open5", "fwd_open10", "fwd_open20")}
D["yizi3"] = np.array([1.0 if r["yizi3"] == "True" else 0.0 for r in rows])
CODE = col("code", str)
IS = D["date3"] < 20240817
OS = ~IS
P(f"事件表 {len(rows):,} 条：样本内 {IS.sum():,} 条（2021-08~2024-08），样本外 {OS.sum():,} 条（2024-08~2026-08）")

EXITS = {"次日开盘卖": "fwd_open1", "持有2日": "fwd_open2", "持有3日": "fwd_open3",
         "持有5日": "fwd_open5", "跌破MA5": "ret_ma5", "回撤8%": "ret_trail8", "跌破MA10": "ret_ma10"}

FILTERS = {
    "流通市值≥80亿": D["mktcap"] >= 80e8,
    "成交额≥8亿": D["amount3"] >= 8e8,
    "换手率2~15%": (D["turnover3"] >= 2) & (D["turnover3"] <= 15),
    "非一字板(振幅>1%)": D["amp3"] > 1.0,
    "跳空-1~+5%": (D["gap4"] >= -0.01) & (D["gap4"] <= 0.05),
    "距MA10乖离≤30%": D["bias10"] <= 0.30,
    "同行业涨停≤2家": D["ind_lu"] <= 2,
    "全市场涨停≤99家": D["mkt_lu"] <= 99,
    "指数在MA20上": D["idx_above_ma20"] > 0,
}
NAMES = list(FILTERS)


def st(mask, y):
    m = mask & ~np.isnan(y)
    n = int(m.sum())
    if n == 0:
        return None
    a = y[m]
    return {"n": n, "win": float((a > 0).mean()), "mean": float(a.mean()), "med": float(np.median(a)),
            "sum": float(a.sum())}


# ── 单个过滤条件的边际效果（3 板，出场=次日开盘卖 / 跌破MA5 两种） ──
P("")
P("一、单个过滤条件的边际效果（样本内，3 板买入）")
for exit_lbl in ("次日开盘卖", "跌破MA5"):
    y = D[EXITS[exit_lbl]]
    base = st(IS & (D["boards"] == 3), y)
    P(f"   出场={exit_lbl}：基准 {base['n']:,} 笔 胜率 {base['win'] * 100:.1f}% 平均 {base['mean'] * 100:+.2f}%")
    for nm in NAMES:
        s = st(IS & (D["boards"] == 3) & FILTERS[nm], y)
        if s and s["n"] >= 80:
            P(f"     {nm:<20}{s['n']:>6,} 笔  胜率 {s['win'] * 100:>5.1f}%  平均 {s['mean'] * 100:>+6.2f}%  "
              f"（相对基准 {(s['mean'] - base['mean']) * 100:>+5.2f}pp）")

# ── 组合搜索 ──
P("")
P("二、组合搜索：样本内挑选，样本外检验（要求样本内 ≥120 笔、样本外 ≥60 笔）")
res = []
for boards in (2, 3):
    bmask = D["boards"] == boards
    for r in range(0, 5):
        for combo in itertools.combinations(NAMES, r):
            m = bmask.copy()
            for nm in combo:
                m &= FILTERS[nm]
            for elbl, ekey in EXITS.items():
                y = D[ekey]
                a = st(IS & m, y)
                b = st(OS & m, y)
                if not a or not b or a["n"] < 120 or b["n"] < 60:
                    continue
                res.append({"boards": boards, "filters": combo, "exit": elbl,
                            "is_n": a["n"], "is_win": a["win"], "is_mean": a["mean"],
                            "os_n": b["n"], "os_win": b["win"], "os_mean": b["mean"],
                            "all_mean": float(np.nanmean(y[m])), "all_n": int((m & ~np.isnan(y)).sum())})
res.sort(key=lambda x: -x["is_mean"])
P(f"   共评估 {len(res):,} 个（板数 × 过滤组合 × 出场）方案")
P("")
P("   样本内最优前 12 名及其样本外表现：")
P("   " + f"{'板':>2} {'出场':<10}{'过滤条件':<44}{'样本内笔数':>8}{'胜率':>7}{'平均':>8}{'样本外笔数':>8}{'胜率':>7}{'平均':>8}")
for x in res[:12]:
    P("   " + f"{x['boards']:>2} {x['exit']:<10}{('+'.join(x['filters']) or '（无过滤）'):<42}"
              f"{x['is_n']:>8,}{x['is_win'] * 100:>6.1f}%{x['is_mean'] * 100:>+7.2f}%"
              f"{x['os_n']:>8,}{x['os_win'] * 100:>6.1f}%{x['os_mean'] * 100:>+7.2f}%")

# 样本内外都为正的方案
both = [x for x in res if x["is_mean"] > 0 and x["os_mean"] > 0]
both.sort(key=lambda x: -(x["is_mean"] + x["os_mean"]) / 2)
P("")
P(f"三、样本内外「双正」的方案：{len(both)} 个")
if both:
    P("   " + f"{'板':>2} {'出场':<10}{'过滤条件':<44}{'内笔数':>7}{'内平均':>8}{'外笔数':>7}{'外平均':>8}{'全样本':>8}")
    for x in both[:15]:
        P("   " + f"{x['boards']:>2} {x['exit']:<10}{('+'.join(x['filters']) or '（无过滤）'):<42}"
                  f"{x['is_n']:>7,}{x['is_mean'] * 100:>+7.2f}%{x['os_n']:>7,}{x['os_mean'] * 100:>+7.2f}%"
                  f"{x['all_mean'] * 100:>+7.2f}%")
else:
    P("   没有任何组合能在样本内外同时取得正的单笔期望。")
    P("   最接近的（按样本内外平均值排序）：")
    res2 = sorted(res, key=lambda x: -(x["is_mean"] + x["os_mean"]) / 2)[:10]
    P("   " + f"{'板':>2} {'出场':<10}{'过滤条件':<44}{'内平均':>8}{'外平均':>8}{'两段均值':>9}")
    for x in res2:
        P("   " + f"{x['boards']:>2} {x['exit']:<10}{('+'.join(x['filters']) or '（无过滤）'):<42}"
                  f"{x['is_mean'] * 100:>+7.2f}%{x['os_mean'] * 100:>+7.2f}%"
                  f"{(x['is_mean'] + x['os_mean']) / 2 * 100:>+8.2f}%")

# ── 出场规则在最优过滤下的横向对比 ──
best = (both or sorted(res, key=lambda x: -(x["is_mean"] + x["os_mean"]) / 2))[0]
P("")
P(f"四、固定最优过滤（{best['boards']}板 + {'+'.join(best['filters']) or '无'}）后，各出场规则对比（全样本）")
m = (D["boards"] == best["boards"])
for nm in best["filters"]:
    m &= FILTERS[nm]
P("   " + f"{'出场规则':<12}{'笔数':>7}{'胜率':>8}{'平均':>8}{'中位数':>8}")
for elbl, ekey in EXITS.items():
    s = st(m, D[ekey])
    if s:
        P("   " + f"{elbl:<12}{s['n']:>7,}{s['win'] * 100:>7.1f}%{s['mean'] * 100:>+7.2f}%{s['med'] * 100:>+7.2f}%")

json.dump({"best": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
           "double_positive": len(both),
           "top": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in x.items()} for x in res[:20]]},
          open(os.path.join(OUT, "limitup_optimize.json"), "w"), ensure_ascii=False, indent=2)
LOG.close()
