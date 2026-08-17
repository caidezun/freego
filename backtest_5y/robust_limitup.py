"""对优化出来的 S4 做稳健性检验，判断它是真规律还是搜索噪音。

  1) 参数敏感性：每个阈值上下挪动，看结果是否成片为正（而不是只有一个点好）
  2) 逐条剔除：每个过滤条件单独去掉，看贡献
  3) 分年 / 分半年稳定性
  4) 多重比较：把 2940 个候选方案的全样本收益分布画出来，看 S4 在什么位置
  5) 随机对照：随机抽取同等笔数的连板交易，看 +0.79% 有多难得
"""
import csv
import itertools
import json
import os
from collections import defaultdict

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUT, "limitup_robust.log"), "w", encoding="utf-8")


def P(m=""):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


rows = list(csv.DictReader(open(os.path.join(OUT, "limitup_events.csv"), encoding="utf-8")))
F = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
D = {k: F(k) for k in ("date3", "date4", "boards", "amount3", "turnover3", "ind_lu", "mkt_lu",
                       "idx_above_ma20", "mktcap", "gap4", "amp3", "fwd_open1", "fwd_open2",
                       "fwd_open3", "fwd_open5", "ret_ma5", "ret_ma10")}
Y = D["fwd_open1"]
ok = ~np.isnan(Y)


def stat(m):
    m = m & ok
    n = int(m.sum())
    if n < 30:
        return None
    a = Y[m]
    return {"n": n, "win": float((a > 0).mean()), "mean": float(a.mean()), "med": float(np.median(a)),
            "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(n))) if n > 2 and a.std() > 0 else 0.0}


def S4(boards=2, amt=8e8, tlo=2, thi=15, indlu=2, idx=True):
    m = (D["boards"] == boards) & (D["amount3"] >= amt) & (D["turnover3"] >= tlo) & (D["turnover3"] <= thi) \
        & (D["ind_lu"] <= indlu)
    if idx:
        m = m & (D["idx_above_ma20"] > 0)
    return m


base = stat(S4())
P("=" * 100)
P("S4 稳健性检验    基准：2板 + 成交额≥8亿 + 换手率2~15% + 同行业涨停≤2家 + 指数在MA20上 + 次日开盘卖")
P(f"基准结果：{base['n']} 笔，胜率 {base['win'] * 100:.2f}%，平均单笔 {base['mean'] * 100:+.3f}%，"
  f"t 值 {base['t']:.2f}")
P("=" * 100)

P("")
P("1) 参数敏感性（每次只挪动一个阈值）")
P("   " + f"{'参数设定':<28}{'笔数':>7}{'胜率':>8}{'平均单笔':>10}{'t值':>7}")
grid = [("连板高度=2（基准）", dict()), ("连板高度=3", dict(boards=3)),
        ("成交额≥5亿", dict(amt=5e8)), ("成交额≥8亿（基准）", dict()), ("成交额≥12亿", dict(amt=12e8)),
        ("成交额≥20亿", dict(amt=20e8)),
        ("换手率1~20%", dict(tlo=1, thi=20)), ("换手率2~15%（基准）", dict()),
        ("换手率3~12%", dict(tlo=3, thi=12)), ("换手率无限制", dict(tlo=0, thi=100)),
        ("同行业涨停≤1家", dict(indlu=1)), ("同行业涨停≤2家（基准）", dict()),
        ("同行业涨停≤4家", dict(indlu=4)), ("同行业涨停无限制", dict(indlu=99)),
        ("不看指数位置", dict(idx=False))]
for lbl, kw in grid:
    s = stat(S4(**kw))
    if s:
        P("   " + f"{lbl:<26}{s['n']:>7,}{s['win'] * 100:>7.2f}%{s['mean'] * 100:>+9.3f}%{s['t']:>7.2f}")

P("")
P("2) 逐条剔除某个过滤条件后的结果（其余保持）")
P("   " + f"{'剔除的条件':<26}{'笔数':>7}{'胜率':>8}{'平均单笔':>10}{'相对基准':>10}")
for lbl, kw in (("成交额", dict(amt=0)), ("换手率区间", dict(tlo=0, thi=100)),
                ("同行业涨停家数", dict(indlu=99)), ("指数位置", dict(idx=False))):
    s = stat(S4(**kw))
    P("   " + f"{lbl:<24}{s['n']:>7,}{s['win'] * 100:>7.2f}%{s['mean'] * 100:>+9.3f}%"
              f"{(s['mean'] - base['mean']) * 100:>+9.3f}pp")

P("")
P("3) 分年 / 分半年稳定性")
m4 = S4() & ok
yr = defaultdict(list)
half = defaultdict(list)
for d, r in zip(D["date4"][m4], Y[m4]):
    d = int(d)
    yr[d // 10000].append(r)
    half[f"{d // 10000}{'H1' if d % 10000 < 700 else 'H2'}"].append(r)
P("   分年：" + "  ".join(f"{y} {np.mean(v) * 100:+.2f}%({len(v)})" for y, v in sorted(yr.items())))
P("   分半年：" + "  ".join(f"{k} {np.mean(v) * 100:+.2f}%({len(v)})" for k, v in sorted(half.items())))
pos_half = sum(1 for v in half.values() if np.mean(v) > 0)
P(f"   {pos_half}/{len(half)} 个半年度为正")

P("")
P("4) 多重比较：2940 个候选方案的全样本单笔收益分布")
FILTERS = {
    "流通市值≥80亿": D["mktcap"] >= 80e8, "成交额≥8亿": D["amount3"] >= 8e8,
    "换手率2~15%": (D["turnover3"] >= 2) & (D["turnover3"] <= 15), "非一字板": D["amp3"] > 1.0,
    "跳空-1~+5%": (D["gap4"] >= -0.01) & (D["gap4"] <= 0.05), "距MA10≤30%": np.ones(len(rows), bool),
    "同行业涨停≤2家": D["ind_lu"] <= 2, "全市场涨停≤99家": D["mkt_lu"] <= 99,
    "指数在MA20上": D["idx_above_ma20"] > 0,
}
names = list(FILTERS)
means = []
for boards in (2, 3):
    for r in range(0, 5):
        for combo in itertools.combinations(names, r):
            m = (D["boards"] == boards)
            for nm in combo:
                m &= FILTERS[nm]
            for ykey in ("fwd_open1", "fwd_open2", "fwd_open3", "fwd_open5", "ret_ma5", "ret_ma10"):
                yy = D[ykey]
                mm = m & ~np.isnan(yy)
                if mm.sum() >= 180:
                    means.append(float(yy[mm].mean()))
means = np.array(means)
P(f"   {len(means):,} 个方案的全样本平均单笔：中位数 {np.median(means) * 100:+.2f}%，"
  f"最好 {means.max() * 100:+.2f}%，为正的仅 {(means > 0).sum()} 个（占 {(means > 0).mean() * 100:.1f}%）")
P(f"   S4 的 {base['mean'] * 100:+.2f}% 位于第 {(means < base['mean']).mean() * 100:.1f} 百分位")
P("   → 绝大多数方案都是负的，说明「连板追进去」整体是负期望；正收益只出现在很窄的角落，")
P("     这既可能是真实的结构性差异，也可能是搜索带来的幸存者偏差，需要用样本外与稳定性来判断。")

P("")
P("5) 随机对照：在全部 2 板事件里随机抽同样笔数，重复 5000 次")
pool = (D["boards"] == 2) & ok
pool_y = Y[pool]
n4 = base["n"]
rng = np.random.default_rng(20260817)
sims = np.array([rng.choice(pool_y, n4, replace=False).mean() for _ in range(5000)])
P(f"   随机组合平均单笔：均值 {sims.mean() * 100:+.2f}%，标准差 {sims.std() * 100:.2f}pp，"
  f"95% 分位 {np.percentile(sims, 95) * 100:+.2f}%")
P(f"   随机抽样中出现 ≥{base['mean'] * 100:+.2f}% 的比例：{(sims >= base['mean']).mean() * 100:.2f}%"
  f"（即经验 p 值 ≈ {(sims >= base['mean']).mean():.4f}）")

json.dump({"base": base, "pos_half": pos_half, "halves": len(half),
           "pctile_in_search": float((means < base["mean"]).mean()),
           "p_random": float((sims >= base["mean"]).mean())},
          open(os.path.join(OUT, "limitup_robust.json"), "w"), ensure_ascii=False, indent=2)
LOG.close()
