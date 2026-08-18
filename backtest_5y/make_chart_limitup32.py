"""三连板后持有到开板：净值曲线与单笔收益分布"""
import csv
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

N = "32_三连板后持有到开板"
eq = list(csv.reader(open(os.path.join(OUT, "results", N + "_signal_equity.csv"), encoding="utf-8")))[1:]
dates = [r[0] for r in eq]
vals = [float(r[1]) * 100 for r in eq]
tr = list(csv.DictReader(open(os.path.join(OUT, "results", N + "_signal_trades.csv"), encoding="utf-8")))
rets = np.array([float(r["净收益率"].rstrip("%")) for r in tr])
bars = np.array([int(r["持有交易日"]) for r in tr])

fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=130, gridspec_kw={"width_ratios": [1.45, 1]})
a1, a2 = ax
a1.plot(range(len(vals)), vals, lw=1.6, color="#2980b9",
        label=f"三连板后持有到开板（等权，{len(rets)} 笔）")
a1.axhline(100, color="#888", ls=":", lw=1)
a1.axhline(142.41, color="#27ae60", ls="-.", lw=1.2, label="全市场等权买入持有 +42.4%")
a1.set_yscale("log")
ymin = max(0.05, min(vals) * 0.6)
a1.set_ylim(ymin, max(200, max(vals) * 1.2))
step = max(1, len(dates) // 10)
a1.set_xticks(range(0, len(dates), step))
a1.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)], rotation=30, fontsize=9)
a1.set_ylabel("净值（对数轴，起点=100）")
a1.set_title("净值曲线：信号级等权（收盘不涨停 → 次日开盘卖）", fontsize=12)
a1.legend(fontsize=9)
a1.grid(alpha=.25, which="both")

bins = [-80, -40, -20, -15, -10, -5, 0, 5, 10, 20, 40, 80, 200]
cnt, _ = np.histogram(rets, bins=bins)
lbl = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins) - 1)]
colors = ["#27ae60" if bins[i] < 0 else "#c0392b" for i in range(len(bins) - 1)]
a2.bar(range(len(cnt)), cnt, color=colors)
a2.set_xticks(range(len(cnt)))
a2.set_xticklabels(lbl, rotation=60, fontsize=8)
a2.set_title(f"单笔净收益（%）：胜率 {(rets > 0).mean() * 100:.1f}%  "
             f"平均 {rets.mean():+.1f}%  中位数 {np.median(rets):+.1f}%\n"
             f"平均持有 {bars.mean():.1f} 日",
             fontsize=11)
a2.set_ylabel("交易笔数")
a2.grid(alpha=.25, axis="y")
for i, v in enumerate(cnt):
    if v:
        a2.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
p = os.path.join(OUT, "limitup32_chart.png")
plt.savefig(p)
print("已输出", p)
