"""汇总 30 个策略的回测结果，输出按胜率、按收益率排序的表格（CSV + Markdown）"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
rows, ctx, bm = S["rows"], S["ctx"], S["benchmark"]


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


COLS = [
    ("编号", lambda r: r["id"]),
    ("策略名称", lambda r: r["name"]),
    ("类别", lambda r: r["cat"]),
    ("信号成交笔数", lambda r: r["signal"]["trades"]),
    ("覆盖标的数", lambda r: r["signal"]["stocks_traded"]),
    ("胜率%", lambda r: round(r["signal"]["win_rate"] * 100, 2)),
    ("平均单笔净收益%", lambda r: round(r["signal"]["avg_ret"] * 100, 3)),
    ("单笔中位数%", lambda r: round(r["signal"]["median_ret"] * 100, 3)),
    ("盈亏比", lambda r: round(r["signal"]["profit_factor"], 3)),
    ("平均盈利%", lambda r: round(r["signal"]["avg_win"] * 100, 2)),
    ("平均亏损%", lambda r: round(r["signal"]["avg_loss"] * 100, 2)),
    ("平均持有交易日", lambda r: round(r["signal"]["avg_bars"], 1)),
    ("单仓复利年化%", lambda r: round(r["signal"]["annual_single"] * 100, 2)),
    ("等权组合5年总收益%", lambda r: round(r["signal"]["total_return"] * 100, 2)),
    ("等权组合年化%", lambda r: round(r["signal"]["annual_return"] * 100, 2)),
    ("等权最大回撤%", lambda r: round(r["signal"]["max_drawdown"] * 100, 2)),
    ("等权夏普", lambda r: round(r["signal"]["sharpe"], 2)),
    ("平均并发持仓", lambda r: round(r["signal"]["avg_positions"], 0)),
    ("资金约束组合总收益%(成交额优先)", lambda r: round(r["portfolio"]["total_return"] * 100, 2)),
    ("资金约束组合总收益%(随机挑选)", lambda r: round(r["portfolio_random_pick"]["total_return"] * 100, 2)),
    ("买入规则", lambda r: " 且 ".join(r["buy"]) if r["buy"] else ""),
    ("卖出规则", lambda r: (" 或 ".join(r["sell"]) if r["sell"] else "") +
        ("".join([f" 或 止盈{r['risk']['takeProfit'] * 100:.0f}%" if r["risk"]["takeProfit"] else "",
                  f" 或 止损{r['risk']['stopLoss'] * 100:.0f}%" if r["risk"]["stopLoss"] else "",
                  f" 或 持满{r['risk']['maxHold']}日" if r["risk"]["maxHold"] else ""]))),
    ("策略出处", lambda r: r["src"]),
]


def write_csv(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(c[0] for c in COLS) + "\n")
        for r in data:
            vals = []
            for _, fn in COLS:
                v = fn(r)
                v = "" if v is None else str(v)
                vals.append('"' + v.replace('"', "'") + '"' if ("," in v or " 且 " in v) else v)
            f.write(",".join(vals) + "\n")


by_win = sorted(rows, key=lambda r: (-r["signal"]["win_rate"], -r["signal"]["total_return"]))
by_ret = sorted(rows, key=lambda r: (-r["signal"]["total_return"], -r["signal"]["win_rate"]))
write_csv(os.path.join(OUT, "summary_by_winrate.csv"), by_win)
write_csv(os.path.join(OUT, "summary_by_return.csv"), by_ret)

MD = ["# A股30个常用量化策略 · 全市场5年回测汇总", "",
      f"- 回测区间：{d2s(S['start'])} ~ {d2s(S['end'])}（5 年，1211 个交易日）",
      f"- 标的范围：全A股 {bm['stocks']} 只（沪深两市；剔除无历史数据的北交所与新股/退市股），"
      f"共 657 万根日线",
      f"- 成本：手续费 {ctx['fee']}‰ + 滑点 {ctx['slip']}‰，双边，单次往返约 {(ctx['fee'] + ctx['slip']) * 2:.1f}‰",
      f"- 撮合：收盘出信号、次日开盘成交，T+1，一字涨跌停不成交",
      f"- 基准：全市场等权买入持有 5 年 **{bm['equal_weight_mean'] * 100:.2f}%**"
      f"（中位数 {bm['median'] * 100:.2f}%，上涨个股占比 {bm['positive_ratio'] * 100:.1f}%）", "",
      "## 一、按胜率从高到低", "",
      "| 排名 | 编号 | 策略 | 类别 | 信号数 | 胜率 | 平均单笔 | 盈亏比 | 平均持有 | 单仓复利年化 | 等权5年收益 | 等权最大回撤 | 等权夏普 |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for i, r in enumerate(by_win, 1):
    s = r["signal"]
    MD.append(f"| {i} | {r['id']} | {r['name']} | {r['cat']} | {s['trades']:,} | "
              f"**{s['win_rate'] * 100:.2f}%** | {s['avg_ret'] * 100:+.2f}% | {s['profit_factor']:.2f} | "
              f"{s['avg_bars']:.1f}日 | {s['annual_single'] * 100:+.1f}% | {s['total_return'] * 100:+.2f}% | "
              f"{s['max_drawdown'] * 100:.1f}% | {s['sharpe']:.2f} |")
MD += ["", "## 二、按收益率从高到低（等权组合 5 年总收益）", "",
       "| 排名 | 编号 | 策略 | 类别 | 等权5年收益 | 等权年化 | 单仓复利年化 | 胜率 | 平均单笔 | 盈亏比 | 等权最大回撤 | 等权夏普 |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for i, r in enumerate(by_ret, 1):
    s = r["signal"]
    MD.append(f"| {i} | {r['id']} | {r['name']} | {r['cat']} | **{s['total_return'] * 100:+.2f}%** | "
              f"{s['annual_return'] * 100:+.2f}% | {s['annual_single'] * 100:+.1f}% | {s['win_rate'] * 100:.2f}% | "
              f"{s['avg_ret'] * 100:+.2f}% | {s['profit_factor']:.2f} | {s['max_drawdown'] * 100:.1f}% | "
              f"{s['sharpe']:.2f} |")
MD += ["", "## 三、资金约束组合（1000 万本金 / 单笔 5% / 最多 20 只）", "",
       "同一批信号在资金与仓位约束下的表现。候选多于剩余仓位时的挑选规则影响极大，故同时给出两种规则：", "",
       "| 编号 | 策略 | 成交笔数 | 胜率 | 总收益(成交额优先) | 总收益(随机挑选) | 最大回撤(成交额优先) |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    p, q = r["portfolio"], r["portfolio_random_pick"]
    MD.append(f"| {r['id']} | {r['name']} | {p['trades']:,} | {p['win_rate'] * 100:.2f}% | "
              f"{p['total_return'] * 100:+.2f}% | {q['total_return'] * 100:+.2f}% | {p['max_drawdown'] * 100:.1f}% |")
MD += ["", "## 四、策略规则与出处", "",
       "| 编号 | 策略 | 中文规则（喂给策略编译器的原文） | 出处 |", "| --- | --- | --- | --- |"]
for r in rows:
    MD.append(f"| {r['id']} | {r['name']} | {r['text']} | {r['src']} |")
MD += ["", "## 五、口径说明", "",
       "1. **信号级等权**：每个买入信号都按等额成交、不设持仓数量上限，因此不受「候选多于仓位时先挑谁」",
       "   这一人为规则影响，是衡量策略本身胜率与单笔收益的口径。等权组合曲线按「当日所有在持仓位等权」",
       "   每日再平衡合成。",
       "2. **单仓复利年化**：按单笔收益的几何平均与平均持有期折算「始终只持一个仓位、反复交易」的年化收益，",
       "   与并发持仓数无关，可用于跨策略比较；等权组合收益则会受策略并发广度影响（并发少的策略波动拖累更大）。",
       "3. **胜率**统计的是扣除手续费与滑点后的净收益为正的比例。",
       "4. 数据为前复权日线；成交额与换手率由成交量推算（已用同源实时快照校验，中位偏差 0.1% 以内）。",
       "5. 科创板（688/689）成交量在数据源中以「股」为单位，已统一换算为「手」，否则其成交额/换手率会放大 100 倍。",
       ""]
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(MD))
print("已输出 summary_by_winrate.csv / summary_by_return.csv / summary.md")
print(f"胜率最高：{by_win[0]['id']} {by_win[0]['name']} {by_win[0]['signal']['win_rate'] * 100:.2f}%")
print(f"收益最高：{by_ret[0]['id']} {by_ret[0]['name']} {by_ret[0]['signal']['total_return'] * 100:+.2f}%")
