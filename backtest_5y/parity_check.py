"""一致性校验：同一批标的、同一区间、同一参数下，
把「页面策略编译器生成的 Python 代码」（纯 Python 逐只序列实现）
与「本目录的 numpy 全市场引擎」逐笔交易比对。

30 个策略全部逐笔一致，才说明用 numpy 跑全市场的结果与页面生成的代码等价。
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E

OUT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz")
SUB = os.path.join(OUT, "parity_subset_csv")
STEP = int(os.environ.get("STEP", "18"))          # 每隔 STEP 只取 1 只做子集
START = int(os.environ.get("START", "20210817"))
END = int(os.environ.get("END", "20260817"))


def subset(pnl, idx):
    sub = E.Panel.__new__(E.Panel)
    sub.codes = [pnl.codes[i] for i in idx]
    sub.names = [pnl.names[i] for i in idx]
    sub.cal, sub.T, sub.N = pnl.cal, pnl.T, len(idx)
    L = int(pnl.n_bars[idx].max())
    sub.L = L
    sub.cf = {k: v[idx][:, :L].copy() for k, v in pnl.cf.items()}
    sub.dates = pnl.dates[idx][:, :L].copy()
    sub.bar = pnl.bar[idx].copy()
    sub.n_bars = pnl.n_bars[idx].copy()
    sub.has = sub.bar >= 0
    return sub


def load_module(path):
    spec = importlib.util.spec_from_file_location("strat_" + os.path.basename(path)[:2], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def key(t):
    return (t["code"], t["buy_date"], round(t["buy_price"], 3), t["sell_date"],
            round(t["sell_price"], 3), t["shares"], t["reason"], round(t["ret"], 6))


def main():
    print(f"载入 {NPZ} …", flush=True)
    pnl = E.Panel(NPZ, skip_new=60)
    idx = np.arange(0, pnl.N, STEP)
    print(f"全市场 {pnl.N} 只 → 校验子集 {len(idx)} 只，区间 {START}~{END}", flush=True)
    sub = subset(pnl, idx)
    if not os.path.isdir(SUB) or len(os.listdir(SUB)) != len(idx):
        print("导出子集 CSV（供生成的 Python 代码读取，确保两边输入完全相同）…", flush=True)
        sub.dump_csv(range(sub.N), SUB)

    index = json.load(open(os.path.join(OUT, "strategies_index.json"), encoding="utf-8"))
    ctx = index["ctx"]
    params = {"init_cash": ctx["cash"], "pos_pct": ctx["posPct"], "max_hold": ctx["maxPos"],
              "fee_permil": ctx["fee"], "slip_permil": ctx["slip"]}

    ok_all = True
    report = []
    for rec in index["list"]:
        py = os.path.join(OUT, rec["py_file"])
        spec = json.load(open(py.replace(".py", ".spec.json"), encoding="utf-8"))["spec"]
        mod = load_module(py)
        ds = mod.load_dataset_from_csv(SUB)
        r_py = mod.run(ds, str(START), str(END), dict(params))
        ind = E.Indicators(sub)
        pp = dict(params)
        pp.update(take_profit=spec["risk"]["takeProfit"], stop_loss=spec["risk"]["stopLoss"],
                  max_bars=spec["risk"]["maxHold"], min_bars=spec["risk"]["minHold"])
        r_np = E.backtest(sub, ind, spec, START, END, pp)
        a = [key(t) for t in r_py["trades"]]
        b = [key(t) for t in r_np["trades"]]
        same = a == b and abs(r_py["final_cash"] - r_np["final_cash"]) < 0.02
        ok_all = ok_all and bool(same)
        first_diff = ""
        if not same:
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else None
                y = b[i] if i < len(b) else None
                if x != y:
                    first_diff = f"  首个差异 #{i}\n    生成代码 {x}\n    numpy   {y}"
                    break
        print(f"{'✓' if same else '✗'} {rec['id']} {rec['name']}  "
              f"生成代码 {len(a)} 笔/期末 {r_py['final_cash']:.2f}   "
              f"numpy {len(b)} 笔/期末 {r_np['final_cash']:.2f}", flush=True)
        if first_diff:
            print(first_diff, flush=True)
        report.append({"id": rec["id"], "name": rec["name"], "same": bool(same),
                       "py_trades": len(a), "np_trades": len(b),
                       "py_final": r_py["final_cash"], "np_final": r_np["final_cash"]})
    json.dump({"subset_stocks": len(idx), "start": START, "end": END, "all_same": bool(ok_all),
               "detail": report}, open(os.path.join(OUT, "parity_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n一致性结论：{'全部 30 个策略逐笔一致' if ok_all else '存在不一致，见上方差异'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
