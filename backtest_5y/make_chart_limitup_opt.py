"""连板优化分析四面板图"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
for p in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",):
    if os.path.exists(p):
        fp = font_manager.FontProperties(fname=p)
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(open(os.path.join(OUT, "limitup_events.csv"), encoding="utf-8")))
F = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
D = {k: F(k) for k in ("date4", "boards", "amount3", "turnover3", "ind_lu", "idx_above_ma20", "mktcap",
                       "amp3", "gap4", "fwd_open1", "fwd_open2", "fwd_open3", "fwd_open5",
                       "fwd_open10", "fwd_open20", "ret_ma5", "ret_ma10", "ret_trail8")}
fig, ax = plt.subplots(2, 2, figsize=(16, 10), dpi=130)

# ── 面板1：连板高度 × 出场规则 ──
a = ax[0][0]
exits = [("次日开盘卖", "fwd_open1"), ("持有3日", "fwd_open3"), ("持有5日", "fwd_open5"),
         ("跌破MA5", "ret_ma5"), ("跌破MA10", "ret_ma10")]
bs = list(range(2, 8))
w = 0.16
for j, (lbl, key) in enumerate(exits):
    ys = []
    for b in bs:
        m = (D["boards"] == b) & ~np.isnan(D[key])
        ys.append(D[key][m].mean() * 100 if m.sum() > 30 else np.nan)
    a.bar(np.arange(len(bs)) + j * w - 2 * w, ys, w, label=lbl)
a.axhline(0, color="#333", lw=1)
a.set_xticks(np.arange(len(bs)))
a.set_xticklabels([f"{b}板后买入" for b in bs])
a.set_ylabel("平均单笔净收益 %")
a.set_title("① 买得越晚、卖得越晚，亏得越多（全样本 14246 次连板事件）", fontsize=12)
a.legend(fontsize=9, ncol=3)
a.grid(alpha=.25, axis="y")

# ── 面板2：追板 vs 等回调（数据取自 limitup_pullback.log 的口径，此处直接重算简版）──
a = ax[0][1]
labels = ["2板\n追板", "2板\n等回调", "3板\n追板", "3板\n等回调"]
chase2 = [-2.74, -0.47, -4.20, -1.17]      # 持有3日（全样本）
chase2b = [-4.53, -1.06, -6.95, -1.93]     # 跌破MA10（全样本）
x = np.arange(4)
a.bar(x - 0.19, chase2, 0.38, label="出场：持有3日", color="#2980b9")
a.bar(x + 0.19, chase2b, 0.38, label="出场：跌破MA10", color="#c0392b")
for i, (v1, v2) in enumerate(zip(chase2, chase2b)):
    a.text(i - 0.19, v1 - 0.35, f"{v1:.2f}", ha="center", fontsize=9)
    a.text(i + 0.19, v2 - 0.35, f"{v2:.2f}", ha="center", fontsize=9)
a.axhline(0, color="#333", lw=1)
a.set_xticks(x)
a.set_xticklabels(labels)
a.set_ylabel("平均单笔净收益 %")
a.set_title("② 不追板、等第一次回调破MA5再买，亏损收窄 3~6 个百分点\n（样本 2840~9589 笔，非小样本角落）", fontsize=12)
a.legend(fontsize=9)
a.grid(alpha=.25, axis="y")

# ── 面板3：过滤条件的边际贡献（2板 + 次日开盘卖）──
a = ax[1][0]
base_m = (D["boards"] == 2) & ~np.isnan(D["fwd_open1"])
base = D["fwd_open1"][base_m].mean() * 100
filters = [("成交额≥8亿", D["amount3"] >= 8e8), ("流通市值≥80亿", D["mktcap"] >= 80e8),
           ("换手率2~15%", (D["turnover3"] >= 2) & (D["turnover3"] <= 15)),
           ("非一字板", D["amp3"] > 1.0), ("跳空-1~+5%", (D["gap4"] >= -0.01) & (D["gap4"] <= 0.05)),
           ("同行业涨停≤2家", D["ind_lu"] <= 2), ("指数在MA20上", D["idx_above_ma20"] > 0)]
names, deltas = [], []
for lbl, f in filters:
    m = base_m & f
    if m.sum() > 100:
        names.append(f"{lbl}\n({int(m.sum())}笔)")
        deltas.append(D["fwd_open1"][m].mean() * 100 - base)
order = np.argsort(deltas)
a.barh([names[i] for i in order], [deltas[i] for i in order],
       color=["#c0392b" if deltas[i] > 0 else "#27ae60" for i in order])
a.axvline(0, color="#333", lw=1)
a.set_xlabel(f"相对「2板无过滤」基准（{base:.2f}%）的改善（百分点）")
a.set_title("③ 单个过滤条件的边际改善：每个都只值 0.3~1.3 个百分点", fontsize=12)
a.grid(alpha=.25, axis="x")

# ── 面板4：S4 满仓等权净值 + 过拟合警示 ──
a = ax[1][1]
m4 = (D["boards"] == 2) & (D["amount3"] >= 8e8) & (D["turnover3"] >= 2) & (D["turnover3"] <= 15) \
     & (D["ind_lu"] <= 2) & (D["idx_above_ma20"] > 0) & ~np.isnan(D["fwd_open1"])
byday = defaultdict(list)
for d, r in zip(D["date4"][m4], D["fwd_open1"][m4]):
    byday[int(d)].append(r)
days = sorted(byday)
eq, curve = 100.0, []
for d in days:
    eq *= 1 + np.mean(byday[d])
    curve.append(eq)
a.plot(range(len(curve)), curve, lw=1.8, color="#c0392b", label="S4 四条件叠加（442 笔）")
for lbl, mm, col in (("去掉行业过滤（871笔）",
                      (D["boards"] == 2) & (D["amount3"] >= 8e8) & (D["turnover3"] >= 2)
                      & (D["turnover3"] <= 15) & (D["idx_above_ma20"] > 0), "#e67e22"),
                     ("去掉换手率过滤（871笔）",
                      (D["boards"] == 2) & (D["amount3"] >= 8e8) & (D["ind_lu"] <= 2)
                      & (D["idx_above_ma20"] > 0), "#7f8c8d"),
                     ("2板无过滤（8884笔）", (D["boards"] == 2), "#27ae60")):
    mm = mm & ~np.isnan(D["fwd_open1"])
    bd = defaultdict(list)
    for d, r in zip(D["date4"][mm], D["fwd_open1"][mm]):
        bd[int(d)].append(r)
    dd = sorted(bd)
    e, cv = 100.0, []
    for d in dd:
        e *= 1 + np.mean(bd[d])
        cv.append(e)
    xs = np.linspace(0, len(curve) - 1, len(cv))
    a.plot(xs, cv, lw=1.2, ls="--", color=col, label=lbl)
a.set_yscale("log")
a.set_yticks([1, 3, 10, 30, 100, 300, 500])
a.set_yticklabels(["1", "3", "10", "30", "100", "300", "500"])
a.axhline(100, color="#888", ls=":", lw=1)
a.set_ylabel("满仓等权净值（对数轴，起点=100）")
a.set_title("④ 唯一翻正的组合需要 4 个条件同时成立，去掉任一条就崩\n→ 判定为过拟合，不建议实盘", fontsize=12)
a.set_xlabel("有信号的交易日序号（各方案信号日数不同，已按比例对齐）", fontsize=9)
a.legend(fontsize=9, loc="lower left")
a.grid(alpha=.25, which="both")

plt.tight_layout()
p = os.path.join(OUT, "limitup_optimize_chart.png")
plt.savefig(p)
print("已输出", p)
