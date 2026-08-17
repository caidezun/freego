# -*- coding: utf-8 -*-
"""
A股量化回测脚本（由页面「策略生成」自动产出，可自由修改后重新回测）

策略描述：
  当5日均线上穿20日均线时买入；当5日均线下穿20日均线时卖出；最长持有60天。

已识别的规则：
  买入（全部满足）：
    - MA5 上穿(金叉) MA20
  卖出（全部满足）：
    - MA5 下穿(死叉) MA20
  风控：止盈 关 · 止损 关 · 最长持有 60 个交易日

数据来源：页面第 1 步下载到浏览器 IndexedDB 的东方财富日线数据，
字段与本脚本 load_dataset() 返回结构一致；也可用导出的 CSV 在本地直接运行。
"""
import json, math, os, sys

# ============ 回测区间（由页面第 5 步注入，可手工修改）============
START_DATE = "2021-08-17"
END_DATE   = "2026-08-17"

PARAMS = {
    "init_cash":   10000000,       # 初始资金
    "pos_pct":     5,              # 单笔资金占比 %（按初始资金计）
    "max_hold":    20,             # 同时最多持仓只数
    "fee_permil":  0.5,            # 双边手续费 ‰（单笔最低 5 元）
    "slip_permil": 0.5,            # 滑点 ‰
    "take_profit": 0,              # 止盈，0=关闭
    "stop_loss":   0,              # 止损，0=关闭
    "max_bars":    60,             # 最长持有交易日，0=不限
    "min_bars":    1,              # T+1：买入次日起才能卖出
}

# ============ 基础工具 ============
NA = float("nan")

def sma(a, n):
    out, s = [NA] * len(a), 0.0
    for i, v in enumerate(a):
        s += v
        if i >= n: s -= a[i - n]
        if i >= n - 1: out[i] = s / n
    return out

def ema(a, n):
    out, k, prev = [NA] * len(a), 2.0 / (n + 1), None
    for i, v in enumerate(a):
        prev = v if prev is None else v * k + prev * (1 - k)
        out[i] = prev
    return out

def macd(close, fast=12, slow=26, sig=9):
    ef, es = ema(close, fast), ema(close, slow)
    dif = [ef[i] - es[i] for i in range(len(close))]
    dea = ema(dif, sig)
    return dif, dea, [(dif[i] - dea[i]) * 2 for i in range(len(close))]

def rsi(close, n=6):
    out = [NA] * len(close); au = ad = 0.0
    for i in range(1, len(close)):
        ch = close[i] - close[i - 1]
        u, d = max(ch, 0.0), max(-ch, 0.0)
        if i <= n:
            au += u / n; ad += d / n
            if i == n: out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
        else:
            au = (au * (n - 1) + u) / n; ad = (ad * (n - 1) + d) / n
            out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out

def kdj(high, low, close, n=9):
    K, D, J = [NA]*len(close), [NA]*len(close), [NA]*len(close)
    pk = pd = 50.0
    for i in range(len(close)):
        s = max(0, i - n + 1)
        hh, ll = max(high[s:i+1]), min(low[s:i+1])
        rsv = 50.0 if hh == ll else (close[i] - ll) / (hh - ll) * 100
        pk = (rsv + 2 * pk) / 3; pd = (pk + 2 * pd) / 3
        K[i], D[i], J[i] = pk, pd, 3 * pk - 2 * pd
    return K, D, J

def boll(close, n=20, k=2.0):
    mid = sma(close, n)
    up, dn = [NA]*len(close), [NA]*len(close)
    for i in range(n - 1, len(close)):
        m = mid[i]
        var = sum((close[j] - m) ** 2 for j in range(i - n + 1, i + 1)) / n
        sd = math.sqrt(var)
        up[i], dn[i] = m + k * sd, m - k * sd
    return mid, up, dn

def bias(close, n):
    # 乖离率：(收盘 - MAn) / MAn × 100
    ma = sma(close, n)
    return [ (close[i] - ma[i]) / ma[i] * 100 if ma[i] == ma[i] and ma[i] != 0 else NA
             for i in range(len(close)) ]

def wr(high, low, close, n):
    # 威廉指标：越接近 100 越处于区间低位
    out = [NA] * len(close)
    for i in range(n - 1, len(close)):
        hh, ll = max(high[i-n+1:i+1]), min(low[i-n+1:i+1])
        out[i] = 50.0 if hh == ll else (hh - close[i]) / (hh - ll) * 100
    return out

def cci(high, low, close, n):
    tp = [(high[i] + low[i] + close[i]) / 3.0 for i in range(len(close))]
    ma = sma(tp, n)
    out = [NA] * len(close)
    for i in range(n - 1, len(close)):
        md = sum(abs(tp[j] - ma[i]) for j in range(i - n + 1, i + 1)) / n
        out[i] = 0.0 if md == 0 else (tp[i] - ma[i]) / (0.015 * md)
    return out

def limit_up_count(chg, n):
    # 近 n 个交易日内的涨停次数（涨幅 >= 9.8%）
    out, c = [NA] * len(chg), 0
    for i in range(len(chg)):
        if chg[i] >= 9.8: c += 1
        if i >= n and chg[i - n] >= 9.8: c -= 1
        if i >= n - 1: out[i] = float(c)
    return out

def roll_max(a, n):
    return [NA if i < n - 1 else max(a[i-n+1:i+1]) for i in range(len(a))]

def roll_min(a, n):
    return [NA if i < n - 1 else min(a[i-n+1:i+1]) for i in range(len(a))]

def cross_up(a, b, i):
    if i < 1: return False
    if a[i] != a[i] or b[i] != b[i] or a[i-1] != a[i-1] or b[i-1] != b[i-1]: return False
    return a[i-1] <= b[i-1] and a[i] > b[i]

def cross_dn(a, b, i):
    if i < 1: return False
    if a[i] != a[i] or b[i] != b[i] or a[i-1] != a[i-1] or b[i-1] != b[i-1]: return False
    return a[i-1] >= b[i-1] and a[i] < b[i]

def streak(close, i, n, up=True):
    if i < n: return False
    for k in range(n):
        d = close[i-k] - close[i-k-1]
        if (d <= 0) if up else (d >= 0): return False
    return True

def ok(v):
    return v == v          # NaN 检查（NaN != NaN）

# ============ 指标预计算 ============
def prepare(S):
    c, h, l, v = S["close"], S["high"], S["low"], S["volume"]
    S["ma5"] = sma(c, 5)
    S["ma10"] = sma(c, 10)
    S["ma20"] = sma(c, 20)
    for n in (5, 10):
        S["vma%d" % n] = sma(v, n)
    S["rsi6"] = rsi(c, 6)
    S["macd_dif"], S["macd_dea"], S["macd_hist"] = macd(c)
    S["kdj_k"], S["kdj_d"], S["kdj_j"] = kdj(h, l, c)
    S["boll_mid"], S["boll_up"], S["boll_dn"] = boll(c)
    # 流通市值：换手率(%) = 成交量(手)*100 / 流通股 *100
    S["fcap"] = [ (v[i] * 1e4 / S["turnover"][i]) * c[i] if S["turnover"][i] > 0 else NA
                  for i in range(len(c)) ]
    return S

# ============ 策略信号（由中文策略翻译而来）============
def buy_signal(S, i):
    """第 i 根K线收盘后是否发出买入信号（次日开盘成交）"""
    return ((cross_up(S["ma5"], S["ma20"], i)))

def sell_signal(S, i, pos):
    """第 i 根K线收盘后是否发出卖出信号，返回 (是否卖出, 原因)"""
    p = PARAMS
    ret = S["close"][i] / pos["price"] - 1.0
    if p["take_profit"] and ret >= p["take_profit"]: return True, "止盈"
    if p["stop_loss"] and ret <= -p["stop_loss"]:    return True, "止损"
    if p["max_bars"] and (i - pos["i"]) >= p["max_bars"]: return True, "到期"
    if ((cross_dn(S["ma5"], S["ma20"], i))):
        return True, "信号"
    return False, ""

# ============ 回测引擎（组合级，A股 T+1 / 次日开盘成交）============
def tradable(S, i):
    """一字涨跌停当日无法成交"""
    return not (S["open"][i] == S["close"][i] == S["high"][i] == S["low"][i] and abs(S["chg"][i]) > 9.5)

def run(dataset, start=None, end=None, params=None):
    p = dict(PARAMS); p.update(params or {})
    start = int((start or START_DATE).replace("-", ""))
    end   = int((end or END_DATE).replace("-", ""))
    stocks = []
    for S in dataset:
        if len(S["dates"]) < 30: continue
        stocks.append(prepare(S))
    # 统一交易日历
    cal = sorted({d for S in stocks for d in S["dates"] if start <= d <= end})
    pos_of = {}                     # code -> position
    by_code = {S["code"]: S for S in stocks}
    date_pos = {S["code"]: {d: i for i, d in enumerate(S["dates"])} for S in stocks}
    cash = float(p["init_cash"])
    trades, equity, pending_buy, pending_sell = [], [], [], []
    fee_r, slip_r = p["fee_permil"] / 1000.0, p["slip_permil"] / 1000.0

    for day in cal:
        # --- 1) 先执行昨日收盘产生的委托（今日开盘价成交）---
        for code, reason in pending_sell:
            S, pos = by_code[code], pos_of.get(code)
            i = date_pos[code].get(day)
            if pos is None or i is None or not tradable(S, i): continue
            px = S["open"][i] * (1 - slip_r)
            gross = px * pos["shares"]
            fee = max(gross * fee_r, 5.0)
            cash += gross - fee
            cost = pos["price"] * pos["shares"] + pos["fee"]
            pnl = gross - fee - cost
            trades.append({
                "code": code, "name": S["name"],
                "buy_date": pos["date"], "buy_price": round(pos["price"], 3),
                "sell_date": day, "sell_price": round(px, 3),
                "shares": pos["shares"], "pnl": round(pnl, 2),
                "ret": round(pnl / cost, 6) if cost else 0.0,
                "bars": i - pos["i"], "reason": reason,
            })
            pos_of.pop(code, None)
        pending_sell = []
        # 候选多于剩余仓位时按信号日成交额从大到小优先（流动性优先，撮合更现实）
        for _amt, code in sorted(pending_buy, reverse=True):
            if len(pos_of) >= p["max_hold"] or code in pos_of: continue
            S = by_code[code]; i = date_pos[code].get(day)
            if i is None or not tradable(S, i): continue
            px = S["open"][i] * (1 + slip_r)
            budget = min(cash, p["init_cash"] * p["pos_pct"] / 100.0)
            shares = int(budget / px / 100) * 100
            if shares < 100: continue
            gross = px * shares
            fee = max(gross * fee_r, 5.0)
            if gross + fee > cash: continue
            cash -= gross + fee
            pos_of[code] = {"i": i, "date": day, "price": px, "shares": shares, "fee": fee}
        pending_buy = []

        # --- 2) 收盘后计算信号，生成次日委托 ---
        for S in stocks:
            i = date_pos[S["code"]].get(day)
            if i is None or i < 1: continue
            pos = pos_of.get(S["code"])
            if pos is not None:
                if (i - pos["i"]) >= p["min_bars"]:
                    sig, reason = sell_signal(S, i, pos)
                    if sig: pending_sell.append((S["code"], reason))
            elif S["code"] not in pos_of:
                if buy_signal(S, i): pending_buy.append((S["amount"][i], S["code"]))

        # --- 3) 结算当日权益 ---
        mv = 0.0
        for code, pos in pos_of.items():
            i = date_pos[code].get(day)
            px = by_code[code]["close"][i] if i is not None else pos["price"]
            mv += px * pos["shares"]
        equity.append([day, round(cash + mv, 2)])

    # --- 区间结束：按最后一根K线收盘价强制平仓，便于统计 ---
    for code, pos in list(pos_of.items()):
        S = by_code[code]
        i = max([j for j, d in enumerate(S["dates"]) if d <= end], default=None)
        if i is None: continue
        px = S["close"][i]
        gross = px * pos["shares"]; fee = max(gross * fee_r, 5.0)
        cash += gross - fee
        cost = pos["price"] * pos["shares"] + pos["fee"]
        trades.append({"code": code, "name": S["name"], "buy_date": pos["date"],
                       "buy_price": round(pos["price"], 3), "sell_date": S["dates"][i],
                       "sell_price": round(px, 3), "shares": pos["shares"],
                       "pnl": round(gross - fee - cost, 2),
                       "ret": round((gross - fee - cost) / cost, 6) if cost else 0.0,
                       "bars": i - pos["i"], "reason": "区间结束平仓"})
        pos_of.pop(code)

    trades.sort(key=lambda t: (t["buy_date"], t["code"]))
    return {"trades": trades, "equity": equity, "init_cash": p["init_cash"],
            "final_cash": round(cash, 2), "start": start, "end": end,
            "engine": "python"}

# ============ 本地独立运行入口（可选）============
def load_dataset_from_csv(folder="./data"):
    """读取页面「导出CSV」得到的文件：date,open,high,low,close,volume,amount,amp,chg,turnover"""
    ds = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".csv"): continue
        code = fn.split("_")[0]; name = fn.split("_")[-1].replace(".csv", "")
        S = {"code": code, "name": name, "dates": [], "open": [], "high": [], "low": [],
             "close": [], "volume": [], "amount": [], "amp": [], "chg": [], "turnover": []}
        with open(os.path.join(folder, fn), encoding="utf-8-sig") as f:
            head = f.readline().strip().split(",")
            col = {k: i for i, k in enumerate(head)}
            for line in f:
                q = line.strip().split(",")
                if len(q) < 10: continue
                S["dates"].append(int(q[col["date"]].replace("-", "")))
                for k in ("open", "high", "low", "close", "volume", "amount", "amp", "chg", "turnover"):
                    S[k].append(float(q[col[k]]))
        ds.append(S)
    return ds

def main():
    ds = load_dataset_from_csv(sys.argv[1] if len(sys.argv) > 1 else "./data")
    r = run(ds, START_DATE, END_DATE)
    wins = [t for t in r["trades"] if t["pnl"] > 0]
    print("交易 %d 笔，胜率 %.2f%%，期末资金 %.2f" % (
        len(r["trades"]), 100.0 * len(wins) / max(1, len(r["trades"])), r["final_cash"]))
    print(json.dumps(r["trades"][:20], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()