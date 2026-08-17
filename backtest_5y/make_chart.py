"""画出信号级等权组合的净值曲线：收益最好的 6 个与最差的 4 个策略，另附基准"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
rows = S["rows"]

# 尝试找一个能显示中文的字体，找不到就退化为英文标签
zh, HAS_ZH = None, False
for path in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if os.path.exists(path):
        zh = font_manager.FontProperties(fname=path)
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [zh.get_name(), "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        HAS_ZH = True
        break


def load_curve(rec):
    base = f"{rec['id']}_{rec['name'].replace('/', '')}_signal_equity.csv"
    xs, ys = [], []
    with open(os.path.join(OUT, "results", base), encoding="utf-8") as f:
        f.readline()
        for ln in f:
            q = ln.split(",")
            xs.append(q[0])
            ys.append(float(q[1]))
    return xs, ys


by_ret = sorted(rows, key=lambda r: -r["signal"]["total_return"])
show = by_ret[:6] + by_ret[-4:]
fig, ax = plt.subplots(figsize=(13, 7.2), dpi=130)
for k, rec in enumerate(show):
    xs, ys = load_curve(rec)
    lbl = (f"{rec['id']} {rec['name']}  {rec['signal']['total_return'] * 100:+.1f}%"
           if HAS_ZH else f"{rec['id']}  {rec['signal']['total_return'] * 100:+.1f}%")
    ax.plot(range(len(ys)), [v * 100 for v in ys], lw=1.7 if k < 6 else 1.2,
            ls="-" if k < 6 else "--", label=lbl)
    if k == 0:
        dates = xs
ax.axhline(100, color="#888", lw=1, ls=":")
bmv = (1 + S["benchmark"]["equal_weight_mean"]) * 100
ax.axhline(bmv, color="#c00", lw=1.2, ls="-.",
           label=(f"基准·全市场等权买入持有 {S['benchmark']['equal_weight_mean'] * 100:+.1f}%"
                  if HAS_ZH else f"benchmark {S['benchmark']['equal_weight_mean'] * 100:+.1f}%"))
step = max(1, len(dates) // 10)
ax.set_xticks(range(0, len(dates), step))
ax.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)], rotation=30, fontsize=9)
ax.set_ylabel("净值（起点=100）" if HAS_ZH else "equity (start=100)",
              fontproperties=zh if HAS_ZH else None)
ax.set_title("A股30个常用量化策略·信号级等权组合净值（2021-08-17 ~ 2026-08-17，全A股5404只）"
             if HAS_ZH else "30 A-share strategies, signal-level equal-weight equity (2021-08 ~ 2026-08)",
             fontproperties=zh if HAS_ZH else None, fontsize=13)
leg = ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=.92,
                prop=zh if HAS_ZH else None)
ax.grid(alpha=.25)
for t in ax.get_yticklabels():
    t.set_fontsize(9)
plt.tight_layout()
p = os.path.join(OUT, "equity_curves.png")
plt.savefig(p)
print("已输出", p, "中文字体:", HAS_ZH)
