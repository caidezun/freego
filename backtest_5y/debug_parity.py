"""定位 numpy 引擎与生成代码的分歧：对同一策略逐日比对候选、成交与现金"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_np as E
from parity_check import subset, SUB

OUT = os.path.dirname(os.path.abspath(__file__))
SID = sys.argv[1] if len(sys.argv) > 1 else "05"
STEP = int(os.environ.get("STEP", "200"))
START, END = 20210817, 20260817

index = json.load(open(os.path.join(OUT, "strategies_index.json"), encoding="utf-8"))
rec = next(r for r in index["list"] if r["id"] == SID)
ctx = index["ctx"]
params = {"init_cash": ctx["cash"], "pos_pct": ctx["posPct"], "max_hold": ctx["maxPos"],
          "fee_permil": ctx["fee"], "slip_permil": ctx["slip"]}

# ── 给生成的代码打上逐日跟踪 ──
src = open(os.path.join(OUT, rec["py_file"]), encoding="utf-8").read()
src = src.replace(
    '        equity.append([day, round(cash + mv, 2)])',
    '        equity.append([day, round(cash + mv, 2)])\n'
    '        TRACE.append((day, sorted(c for _a, c in pending_buy), sorted(pos_of.keys()),\n'
    '                      round(cash, 2), sorted(c for c, _r in pending_sell)))')
src = "TRACE = []\n" + src
path = "/tmp/dbg_strategy.py"
open(path, "w", encoding="utf-8").write(src)
spec_mod = importlib.util.spec_from_file_location("dbg", path)
mod = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(mod)

pnl = E.Panel(os.environ.get("NPZ", "/home/ubuntu/ashare5y/all.npz"), skip_new=60)
sub = subset(pnl, np.arange(0, pnl.N, STEP))
ds = mod.load_dataset_from_csv(SUB)
r_py = mod.run(ds, str(START), str(END), dict(params))
py_trace = mod.TRACE

spec = json.load(open(os.path.join(OUT, rec["py_file"]).replace(".py", ".spec.json"),
                     encoding="utf-8"))["spec"]
pp = dict(params)
pp.update(take_profit=spec["risk"]["takeProfit"], stop_loss=spec["risk"]["stopLoss"],
          max_bars=spec["risk"]["maxHold"], min_bars=spec["risk"]["minHold"])
ind = E.Indicators(sub)
E.TRACE = []
r_np = E.backtest_traced(sub, ind, spec, START, END, pp)
np_trace = E.TRACE

print(f"策略 {SID} {rec['name']}   生成代码 {len(r_py['trades'])} 笔 / numpy {len(r_np['trades'])} 笔")
print(f"trace 长度 py={len(py_trace)} np={len(np_trace)}")
for k in range(min(len(py_trace), len(np_trace))):
    a, b = py_trace[k], np_trace[k]
    if a != b:
        print(f"\n第一处分歧在第 {k} 个交易日 {a[0]}")
        for lbl, x in (("生成代码", a), ("numpy  ", b)):
            print(f"  {lbl} 明日买入候选={x[1][:8]}{'...' if len(x[1]) > 8 else ''}({len(x[1])}) "
                  f"持仓={x[2]} 现金={x[3]} 明日卖出={x[4]}")
        if k:
            print(f"  上一日 {py_trace[k-1][0]} py 持仓={py_trace[k-1][2]} 候选数={len(py_trace[k-1][1])}")
            print(f"  上一日 {np_trace[k-1][0]} np 持仓={np_trace[k-1][2]} 候选数={len(np_trace[k-1][1])}")
        break
else:
    print("逐日 trace 完全一致")
