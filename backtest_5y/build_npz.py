"""把下载好的逐只 CSV 合并成一个列式 npz，便于回测时一次性载入。

输出 all.npz：
  codes/names  标的代码与名称
  offsets      每只标的在长数组中的起止位置（offsets[i]..offsets[i+1]）
  dates        YYYYMMDD 整数
  open/high/low/close/volume/amount/amp/chg/turnover  float64 长数组（与 CSV 十进制值一致）
"""
import json
import os
import sys

import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/ashare5y"
CSV = os.path.join(DATA, "csv")
OUT = os.path.join(DATA, "all.npz")

files = sorted(f for f in os.listdir(CSV) if f.endswith(".csv"))
codes, names, offsets = [], [], [0]
cols = {k: [] for k in ("d", "o", "h", "l", "c", "v", "a", "amp", "chg", "tr")}
bad = 0
for k, fn in enumerate(files):
    code, name = fn[:-4].split("_", 1)
    try:
        arr = np.loadtxt(os.path.join(CSV, fn), delimiter=",", skiprows=1,
                         dtype=np.float64, converters={0: lambda s: float(s.replace("-", ""))})
    except Exception:
        bad += 1
        continue
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < 30:                      # 上市不足 30 个交易日的直接跳过
        continue
    codes.append(code)
    names.append(name)
    cols["d"].append(arr[:, 0].astype(np.int64))
    for j, key in enumerate(("o", "h", "l", "c", "v", "a", "amp", "chg", "tr"), start=1):
        cols[key].append(arr[:, j])
    offsets.append(offsets[-1] + arr.shape[0])
    if len(codes) % 500 == 0:
        print(f"  {len(codes)}/{len(files)}", flush=True)

np.savez(OUT,
         codes=np.array(codes), names=np.array(names),
         offsets=np.array(offsets, dtype=np.int64),
         dates=np.concatenate(cols["d"]),
         **{k: np.concatenate(cols[v])
            for k, v in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"),
                         ("volume", "v"), ("amount", "a"), ("amp", "amp"),
                         ("chg", "chg"), ("turnover", "tr"))})
n = offsets[-1]
print(f"合并完成：{len(codes)} 只标的、{n} 条日线，跳过 {bad} 个异常文件")
print(f"日期范围：{cols['d'][0].min()} ~ {max(int(d.max()) for d in cols['d'])}")
print(f"输出：{OUT}  {os.path.getsize(OUT)/1048576:.0f}MB")
json.dump({"stocks": len(codes), "rows": int(n)},
          open(os.path.join(DATA, "npz_meta.json"), "w"))
