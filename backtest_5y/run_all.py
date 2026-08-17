"""对 30 个策略在「全A股 × 最近5年」上逐个回测，保留每个策略的完整过程与结果。

两种度量口径（同一批信号、同一套退出规则）：
  A 信号级等权：每个买入信号都成交、等权、不设持仓上限 —— 衡量策略本身的胜率与单笔收益，
    不受「候选多于仓位时先挑谁」这一人为规则影响；并按当日在持仓位等权合成日收益曲线。
  B 资金约束组合：1000 万本金、单笔 5%、最多同时持 20 只 —— 与页面生成的 Python 代码
    逐笔一致（parity_check.py 已验证），另跑一版「候选随机挑选」用于衡量分配规则的影响。

每个策略输出：
  results/<id>_<name>_signal_trades.csv  信号级逐笔交易
  results/<id>_<name>_signal_equity.csv  信号级等权日收益曲线
  results/<id>_<name>_port_trades.csv    资金约束组合逐笔交易
  results/<id>_<name>_port_equity.csv    资金约束组合每日权益
  results/<id>_<name>_metrics.json       两种口径的全部指标
汇总：summary.json / summary_by_winrate.csv / summary_by_return.csv / summary.md / run_all.log
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
NPZ = os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz")
START = int(os.environ.get("START", "20210817"))
END = int(os.environ.get("END", "20260817"))
ONLY = os.environ.get("ONLY", "")


def d2s(d):
    d = int(d)
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def benchmark(pnl, i0, i1):
    """基准：全市场等权买入持有（区间内有足够数据的标的）"""
    rets, first, last = [], [], []
    for i in range(pnl.N):
        cols = np.nonzero(pnl.has[i, i0:i1])[0]
        if len(cols) < 100 or pnl.n_bars[i] < 30:
            continue
        a = int(pnl.bar[i, i0 + cols[0]])
        b = int(pnl.bar[i, i0 + cols[-1]])
        c0, c1 = pnl.cf["close"][i, a], pnl.cf["close"][i, b]
        if c0 > 0:
            rets.append(c1 / c0 - 1)
    r = np.array(rets)
    return {"stocks": len(r), "equal_weight_mean": float(r.mean()),
            "median": float(np.median(r)), "positive_ratio": float((r > 0).mean())}


def write_trades(path, trades, with_shares):
    head = ("序号,代码,名称,买入日,买入价,卖出日,卖出价,股数,净收益率,盈亏(元),持有交易日,卖出原因\n"
            if with_shares else "序号,代码,名称,买入日,买入价,卖出日,卖出价,净收益率,毛收益率,持有交易日,卖出原因\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head)
        for k, t in enumerate(trades, 1):
            if with_shares:
                f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},"
                        f"{d2s(t['sell_date'])},{t['sell_price']},{t['shares']},"
                        f"{t['ret'] * 100:.4f}%,{t['pnl']:.2f},{t['bars']},{t['reason']}\n")
            else:
                f.write(f"{k},{t['code']},{t['name']},{d2s(t['buy_date'])},{t['buy_price']},"
                        f"{d2s(t['sell_date'])},{t['sell_price']},{t['ret'] * 100:.4f}%,"
                        f"{t['gross_ret'] * 100:.4f}%,{t['bars']},{t['reason']}\n")


def main():
    t00 = time.time()
    log = open(os.path.join(OUT, "run_all.log"), "w", encoding="utf-8")

    def P(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    P(f"回测区间 {d2s(START)} ~ {d2s(END)}    数据 {NPZ}")
    t0 = time.time()
    pnl = E.Panel(NPZ, skip_new=60)
    i0, i1 = pnl.col_range(START, END)
    P(f"面板：{pnl.N} 只标的、{int(pnl.n_bars.sum())} 根K线，区间内 {i1 - i0} 个交易日，"
      f"载入 {time.time() - t0:.1f}s")
    P("样本处理：区间内新上市标的跳过上市后前 60 根K线（次新期）；总K线不足 30 根不参与；"
      "一字涨跌停日不成交")

    bm = benchmark(pnl, i0, i1)
    P(f"基准（全市场等权买入持有）：{bm['stocks']} 只，平均 {bm['equal_weight_mean'] * 100:.2f}%，"
      f"中位数 {bm['median'] * 100:.2f}%，上涨比例 {bm['positive_ratio'] * 100:.1f}%")
    json.dump(bm, open(os.path.join(OUT, "benchmark.json"), "w"), ensure_ascii=False, indent=2)

    index = json.load(open(os.path.join(OUT, "strategies_index.json"), encoding="utf-8"))
    ctx = index["ctx"]
    base = {"init_cash": ctx["cash"], "pos_pct": ctx["posPct"], "max_hold": ctx["maxPos"],
            "fee_permil": ctx["fee"], "slip_permil": ctx["slip"]}
    P(f"成本口径：手续费 {ctx['fee']}‰ 双边（单笔最低 5 元）+ 滑点 {ctx['slip']}‰ 双边，"
      f"合计单次往返约 {(ctx['fee'] + ctx['slip']) * 2:.1f}‰")
    P(f"资金约束口径：本金 {ctx['cash']:,} 元，单笔 {ctx['posPct']}%，最多同时持 {ctx['maxPos']} 只\n")

    ind = E.Indicators(pnl)
    rows = []
    for rec in index["list"]:
        if ONLY and rec["id"] not in ONLY.split(","):
            continue
        spec = json.load(open(os.path.join(OUT, rec["py_file"]).replace(".py", ".spec.json"),
                              encoding="utf-8"))["spec"]
        p = dict(base)
        p.update(take_profit=spec["risk"]["takeProfit"], stop_loss=spec["risk"]["stopLoss"],
                 max_bars=spec["risk"]["maxHold"], min_bars=spec["risk"]["minHold"])
        base_name = f"{rec['id']}_{rec['name'].replace('/', '')}"
        t0 = time.time()

        rs = E.signal_backtest(pnl, ind, spec, START, END, p)
        ms = E.signal_metrics(rs)
        write_trades(os.path.join(RES, base_name + "_signal_trades.csv"), rs["trades"], False)
        with open(os.path.join(RES, base_name + "_signal_equity.csv"), "w", encoding="utf-8") as f:
            f.write("日期,等权净值,当日在持仓位数\n")
            cnt = rs["daily"]
            for k, d in enumerate(rs["dates"]):
                f.write(f"{d2s(d)},{rs['equity_curve'][k]:.6f},{'' if k else ''}\n")

        rp = E.backtest(pnl, ind, spec, START, END, p, pick="amount")
        mp = E.metrics(rp)
        write_trades(os.path.join(RES, base_name + "_port_trades.csv"), rp["trades"], True)
        with open(os.path.join(RES, base_name + "_port_equity.csv"), "w", encoding="utf-8") as f:
            f.write("日期,权益(元)\n")
            for d, v in rp["equity"]:
                f.write(f"{d2s(d)},{v}\n")

        rr = E.backtest(pnl, ind, spec, START, END, p, pick="random", seed=7)
        mr = E.metrics(rr)

        reasons = {}
        for t in rs["trades"]:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        out = {**{k: rec[k] for k in ("id", "name", "cat", "src", "text", "buy", "sell", "risk")},
               "signal": ms, "portfolio": mp, "portfolio_random_pick": mr,
               "exit_reasons": reasons, "seconds": round(time.time() - t0, 1)}
        json.dump(out, open(os.path.join(RES, base_name + "_metrics.json"), "w"),
                  ensure_ascii=False, indent=2)
        rows.append(out)
        P(f"{rec['id']} {rec['name']:<20} 信号 {ms['trades']:>6} 笔 胜率 {ms['win_rate'] * 100:>5.2f}% "
          f"单笔 {ms['avg_ret'] * 100:>6.2f}% 盈亏比 {ms['profit_factor']:>4.2f} "
          f"等权 {ms['total_return'] * 100:>7.2f}%/年化 {ms['annual_return'] * 100:>6.2f}%/回撤 {ms['max_drawdown'] * 100:>5.1f}% "
          f"｜组合 {mp['total_return'] * 100:>7.2f}%（随机挑选 {mr['total_return'] * 100:>7.2f}%） {time.time() - t0:.1f}s")

    json.dump({"ctx": ctx, "benchmark": bm, "start": START, "end": END, "rows": rows},
              open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
    P(f"\n全部完成，总用时 {(time.time() - t00) / 60:.1f} 分钟")
    log.close()


if __name__ == "__main__":
    main()
