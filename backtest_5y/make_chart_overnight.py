"""隔夜持股法：各方案平均单笔 + 净值曲线"""
import csv
import os
from collections import defaultdict

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

# 从日志/json 读汇总
import json
js = json.load(open(os.path.join(OUT, "overnight_summary.json"), encoding="utf-8"))
# 净值
eq = defaultdict(list)
for r in csv.DictReader(open(os.path.join(OUT, "results", "overnight_equity.csv"), encoding="utf-8")):
    eq[r["方案"]].append((r["日期"], float(r["净值"])))

fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=130)
a1, a2 = ax

# 左：净值
colors = {
    "S0 无过滤全市场": "#7f8c8d",
    "S2 经典四条件（3-5/换手/市值/量比）": "#c0392b",
    "S2c 经典四条件 每日成交额Top1": "#8e44ad",
    "S5 ClawHub主板隔夜": "#2980b9",
}
for name, col in colors.items():
    rows = eq.get(name)
    if not rows:
        continue
    ys = [v * 100 for _, v in rows]
    a1.plot(range(len(ys)), ys, lw=1.5, color=col, label=name.split("（")[0])
a1.axhline(100, color="#888", ls=":", lw=1)
a1.set_yscale("log")
a1.set_ylabel("满仓等权净值（对数轴，起点=100）")
a1.set_title("尾盘买、次日开盘卖：过滤越「精」、亏得越快", fontsize=12)
a1.legend(fontsize=8, loc="lower left")
a1.grid(alpha=.25, which="both")

# 右：平均单笔
order = [
    "S0 无过滤全市场",
    "S1 仅涨幅3%~5%",
    "S6 主板仅3%~5%",
    "S5 ClawHub主板隔夜",
    "S2 经典四条件（3-5/换手/市值/量比）",
    "S4 八步日线近似（均线多头+非长上影）",
    "S2c 经典四条件 每日成交额Top1",
    "S3 六步日线近似（含涨停记忆）",
]
short = ["无过滤", "仅3-5%", "主板3-5%", "ClawHub主板", "经典四条件",
         "八步+均线", "四条件Top1", "六步+涨停记忆"]
means = [js[k]["mean"] * 100 for k in order]
wins = [js[k]["win"] * 100 for k in order]
cols = ["#27ae60" if m > 0 else "#c0392b" for m in means]
x = np.arange(len(order))
a2.bar(x, means, color=cols)
for i, (m, w) in enumerate(zip(means, wins)):
    a2.text(i, m - 0.08, f"{m:.2f}\n胜率{w:.0f}%", ha="center", va="top", fontsize=8)
a2.axhline(0, color="#333", lw=1)
a2.set_xticks(x)
a2.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
a2.set_ylabel("平均单笔净收益 %（次日开盘卖，含成本）")
a2.set_title("网上越强调的过滤（涨停记忆、每天挑最热的），隔夜亏得越多", fontsize=12)
a2.grid(alpha=.25, axis="y")

plt.tight_layout()
p = os.path.join(OUT, "overnight_chart.png")
plt.savefig(p)
print("已输出", p)
