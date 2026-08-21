"""横盘放量首板后回踩 MA5：净值与回踩定义对照"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
for path in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",):
    if os.path.exists(path):
        fp = font_manager.FontProperties(fname=path)
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

js = json.load(open(os.path.join(OUT, "sideways_lu_ma5_summary.json"), encoding="utf-8"))
eq = list(csv.DictReader(open(os.path.join(OUT, "results", "sideways_lu_ma5_equity.csv"), encoding="utf-8")))
dates = [r["日期"] for r in eq]

fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=130)
a1, a2 = ax
series = [
    ("仅首板回踩", "仅首板+多头，回踩MA5", "#8e44ad", 1.2),
    ("T+2追", "同一批票 T+2 开盘追（策略34）", "#7f8c8d", 1.4),
    ("破MA5回踩", "第一次收盘跌破MA5再买", "#2980b9", 1.4),
    ("主口径回踩", "主口径：收盘回踩MA5±2%", "#c0392b", 2.0),
]
for col, lab, color, lw in series:
    ys = [float(r[col]) * 100 for r in eq if r.get(col)]
    if len(ys) != len(dates):
        ys = [float(r[col]) * 100 if r.get(col) else np.nan for r in eq]
    a1.plot(range(len(ys)), ys, lw=lw, color=color, label=lab)
a1.axhline(100, color="#888", ls=":", lw=1)
a1.axhline(142.41, color="#27ae60", ls="-.", lw=1.2, label="全市场等权买入持有 +42.4%")
a1.set_yscale("log")
a1.set_ylabel("净值（对数轴，起点=100）")
a1.set_title("等回踩比直接追好一点，5 年照样亏掉本金", fontsize=12)
step = max(1, len(dates) // 10)
a1.set_xticks(range(0, len(dates), step))
a1.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)], rotation=30, fontsize=9)
a1.legend(fontsize=8, loc="lower left")
a1.grid(alpha=.25, which="both")

order = [
    ("P0 不回踩：T+2 开盘追（策略34）", "T+2追"),
    ("P2 最低价触及 MA5 且收盘仍在线上", "触及MA5"),
    ("P1 收盘 MA5±2%（主口径）", "MA5±2%"),
    ("P3 收盘落在 [MA5, MA5+2%]", "贴着MA5"),
    ("P4 第一次收盘跌破 MA5（S12口径）", "跌破MA5"),
]
means, wins, cnts, labs = [], [], [], []
for k, lab in order:
    m = js["defs"][k]
    means.append(m["avg"] * 100)
    wins.append(m["win"] * 100)
    cnts.append(m["trades"])
    labs.append(lab)
cols = ["#27ae60" if m > 0 else "#c0392b" for m in means]
x = np.arange(len(order))
a2.bar(x, means, color=cols)
for i, (m, w, n) in enumerate(zip(means, wins, cnts)):
    a2.text(i, m - 0.06, f"{m:+.2f}\n胜率{w:.0f}%\n{n:,}笔", ha="center", va="top", fontsize=8)
a2.axhline(0, color="#333", lw=1)
a2.set_xticks(x)
a2.set_xticklabels(labs, fontsize=9)
a2.set_ylabel("平均单笔净收益 %（含成本）")
a2.set_title("怎么定义「回踩」，单笔都在 −0.8%～−1.7%", fontsize=12)
a2.grid(alpha=.25, axis="y")

plt.tight_layout()
p = os.path.join(OUT, "sideways_lu_ma5_chart.png")
plt.savefig(p)
print("已输出", p)
