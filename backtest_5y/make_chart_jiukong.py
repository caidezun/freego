"""九空一多：连阴天数敏感度 + 净值对照"""
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

js = json.load(open(os.path.join(OUT, "jiukong_summary.json"), encoding="utf-8"))
eq = list(csv.DictReader(open(os.path.join(OUT, "results", "jiukong_equity_alters.csv"), encoding="utf-8")))
dates = [r["日期"] for r in eq]

fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=130)
a1, a2 = ax

series = [
    ("五阴一阳", "五阴一阳", "#27ae60", 1.8),
    ("六阴一阳", "六阴一阳", "#16a085", 1.4),
    ("九阴一阳", "九阴一阳（主口径）", "#c0392b", 2.0),
    ("TD低九后一多", "TD低九后一多", "#2980b9", 1.3),
    ("TD低九当日", "TD低九当日", "#8e44ad", 1.1),
]
for col, lab, color, lw in series:
    ys = [float(r[col]) * 100 for r in eq]
    a1.plot(range(len(ys)), ys, lw=lw, color=color, label=lab)
a1.axhline(100, color="#888", ls=":", lw=1)
a1.axhline(142.41, color="#2c3e50", ls="-.", lw=1.2, label="全市场等权买入持有 +42.4%")
a1.set_yscale("log")
a1.set_ylabel("净值（对数轴，起点=100）")
a1.set_title("信号级等权净值：连阴越长，越接飞刀", fontsize=12)
step = max(1, len(dates) // 10)
a1.set_xticks(range(0, len(dates), step))
a1.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)], rotation=30, fontsize=9)
a1.legend(fontsize=8, loc="upper left")
a1.grid(alpha=.25, which="both")

ns = [4, 5, 6, 7, 8, 9, 10, 12]
means = [js["sens"][str(n)]["avg"] * 100 for n in ns]
wins = [js["sens"][str(n)]["win"] * 100 for n in ns]
cnts = [js["sens"][str(n)]["trades"] for n in ns]
cols = ["#27ae60" if m > 0 else "#c0392b" for m in means]
x = np.arange(len(ns))
a2.bar(x, means, color=cols)
for i, (m, w, ntr) in enumerate(zip(means, wins, cnts)):
    va = "bottom" if m >= 0 else "top"
    off = 0.04 if m >= 0 else -0.04
    a2.text(i, m + off, f"{m:+.2f}\n{w:.0f}%\n{ntr//1000}k笔" if ntr >= 1000 else f"{m:+.2f}\n{w:.0f}%\n{ntr}笔",
            ha="center", va=va, fontsize=7.5)
a2.axhline(0, color="#333", lw=1)
a2.set_xticks(x)
a2.set_xticklabels([f"{n}阴一阳" for n in ns], fontsize=9)
a2.set_ylabel("平均单笔净收益 %（含成本）")
a2.set_title("「九」没有魔法：4–6 根阴线后的反弹还在，9 根已经接刀", fontsize=12)
a2.grid(alpha=.25, axis="y")

plt.tight_layout()
p = os.path.join(OUT, "jiukong_chart.png")
plt.savefig(p)
print("已输出", p)
