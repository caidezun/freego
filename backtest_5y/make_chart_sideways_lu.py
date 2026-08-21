"""横盘放量首板 T+2：净值与加条件对照"""
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

js = json.load(open(os.path.join(OUT, "sideways_lu_summary.json"), encoding="utf-8"))
eq = list(csv.DictReader(open(os.path.join(OUT, "results", "sideways_lu_equity_alters.csv"), encoding="utf-8")))
dates = [r["日期"] for r in eq]

fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=130)
a1, a2 = ax
series = [
    ("裸首板T2", "裸首板 T+2", "#7f8c8d", 1.2),
    ("多头排列T2", "首板+多头排列 T+2", "#8e44ad", 1.2),
    ("T+1开盘", "主口径但 T+1 开盘", "#2980b9", 1.4),
    ("主口径T2", "主口径 T+2（横盘+放量+多头）", "#c0392b", 2.0),
]
for col, lab, color, lw in series:
    ys = [float(r[col]) * 100 for r in eq]
    a1.plot(range(len(ys)), ys, lw=lw, color=color, label=lab)
a1.axhline(100, color="#888", ls=":", lw=1)
a1.axhline(142.41, color="#27ae60", ls="-.", lw=1.2, label="全市场等权买入持有 +42.4%")
a1.set_yscale("log")
a1.set_ylabel("净值（对数轴，起点=100）")
a1.set_title("信号级等权净值：横盘放量再追首板，5 年亏掉本金", fontsize=12)
step = max(1, len(dates) // 10)
a1.set_xticks(range(0, len(dates), step))
a1.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)], rotation=30, fontsize=9)
a1.legend(fontsize=8, loc="lower left")
a1.grid(alpha=.25, which="both")

order = [
    ("S0 裸首板 T+2", "裸首板"),
    ("S1 +多头排列", "+多头"),
    ("S2 +放量2倍", "+放量"),
    ("S3 +20日无涨停", "+无涨停"),
    ("S4 主口径 +横盘15%", "主口径"),
]
means = [js["ablation"][k]["avg"] * 100 for k, _ in order]
wins = [js["ablation"][k]["win"] * 100 for k, _ in order]
cnts = [js["ablation"][k]["trades"] for k, _ in order]
cols = ["#27ae60" if m > 0 else "#c0392b" for m in means]
x = np.arange(len(order))
a2.bar(x, means, color=cols)
for i, (m, w, n) in enumerate(zip(means, wins, cnts)):
    a2.text(i, m - 0.08, f"{m:+.2f}\n胜率{w:.0f}%\n{n:,}笔", ha="center", va="top", fontsize=8)
a2.axhline(0, color="#333", lw=1)
a2.set_xticks(x)
a2.set_xticklabels([s for _, s in order], fontsize=9)
a2.set_ylabel("平均单笔净收益 %（含成本）")
a2.set_title("条件加得越「像网上说的」，单笔并不更好", fontsize=12)
a2.grid(alpha=.25, axis="y")

plt.tight_layout()
p = os.path.join(OUT, "sideways_lu_chart.png")
plt.savefig(p)
print("已输出", p)
