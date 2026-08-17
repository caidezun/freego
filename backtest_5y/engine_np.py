"""全市场回测引擎（numpy 实现），撮合与统计口径与页面「策略生成」产出的 Python 代码逐条对应：

  · 收盘后产生信号，次日开盘价成交；买入价上浮滑点、卖出价下浮滑点
  · T+1：买入当日不可卖（i - pos.i >= min_bars）
  · 一字涨跌停（开=高=低=收 且 |涨跌幅|>9.5）当日不成交
  · 双边手续费按 ‰ 计，单笔最低 5 元
  · 单笔预算 = min(可用现金, 初始资金 × 单笔仓位%)，股数向下取整到 100 股
  · 候选多于剩余仓位时按信号日成交额从大到小优先（并列按代码逆序，与 Python
    sorted(..., reverse=True) 一致）
  · 卖出优先级：止盈 > 止损 > 到期 > 信号
  · 区间结束时按 <= 结束日的最后一根 K 线收盘价强制平仓

指标一律在「每只标的自己的连续 K 线序列」上计算（停牌不产生空洞），这一点与逐只序列的
Python 代码完全一致；为了向量化，这里把所有标的的序列左对齐成 [标的 × 序号] 的矩阵，
另用 bar[标的, 交易日] 记录某个交易日对应该标的的第几根 K 线。
等价性由 parity_check.py 用同一批标的、同一批参数逐笔比对来保证。
"""
import numpy as np


# ────────────────────────────── 数据 ──────────────────────────────
class Panel:
    """紧凑面板：cf[field] 形状 [N × L]，第 i 行是第 i 只标的的连续K线（右侧 NaN 补齐）。"""

    def __init__(self, npz, skip_new=60):
        z = np.load(npz, allow_pickle=False)
        self.codes = [str(c) for c in z["codes"]]
        self.names = [str(n) for n in z["names"]]
        off = z["offsets"]
        dates_all = z["dates"]
        self.cal = np.unique(dates_all)
        T = self.T = len(self.cal)
        N = self.N = len(self.codes)
        cal_first = int(self.cal[0])
        fields = ("open", "high", "low", "close", "volume", "amount", "amp", "chg", "turnover")
        raw = {k: z[k] for k in fields}          # 一次性读入，避免在循环里反复解包 npz
        lens = np.zeros(N, dtype=np.int64)
        starts = np.zeros(N, dtype=np.int64)
        for i in range(N):
            a, b = off[i], off[i + 1]
            d0 = int(dates_all[a])
            # 区间内新上市（首个交易日明显晚于数据起点）→ 跳过次新期前 skip_new 根
            sk = skip_new if (skip_new and d0 > cal_first and (b - a) > skip_new + 5) else 0
            starts[i] = a + sk
            lens[i] = (b - a) - sk
        L = self.L = int(lens.max())
        self.cf = {k: np.full((N, L), np.nan, dtype=np.float64) for k in fields}
        self.dates = np.zeros((N, L), dtype=np.int64)
        self.bar = np.full((N, T), -1, dtype=np.int32)
        self.n_bars = lens.astype(np.int32)
        for i in range(N):
            a, n = int(starts[i]), int(lens[i])
            if n <= 0:
                continue
            for k in fields:
                self.cf[k][i, :n] = raw[k][a:a + n]
            d = dates_all[a:a + n]
            self.dates[i, :n] = d
            self.bar[i, np.searchsorted(self.cal, d)] = np.arange(n, dtype=np.int32)
        z.close()
        self.has = self.bar >= 0

    def col_range(self, start, end):
        return (int(np.searchsorted(self.cal, start, "left")),
                int(np.searchsorted(self.cal, end, "right")))

    def dump_csv(self, idx, folder):
        """把面板里某些标的的序列导出为 CSV（表头与页面导出一致），用于与生成的 Python 代码对照"""
        import os
        os.makedirs(folder, exist_ok=True)
        for i in idx:
            n = int(self.n_bars[i])
            rows = ["date,open,high,low,close,volume,amount,amp,chg,turnover"]
            for j in range(n):
                d = int(self.dates[i, j])
                rows.append("%d-%02d-%02d,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%.3f,%.3f,%.4f" % (
                    d // 10000, d // 100 % 100, d % 100,
                    self.cf["open"][i, j], self.cf["high"][i, j], self.cf["low"][i, j],
                    self.cf["close"][i, j], self.cf["volume"][i, j], self.cf["amount"][i, j],
                    self.cf["amp"][i, j], self.cf["chg"][i, j], self.cf["turnover"][i, j]))
            name = self.names[i].replace("/", "")
            open(os.path.join(folder, f"{self.codes[i]}_{name}.csv"), "w", encoding="utf-8").write(
                "\n".join(rows) + "\n")


# ─────────────────────────── 指标（矩阵级） ───────────────────────────
def sma(x, n):
    """滚动均值。刻意采用「加新值、减出窗值」的增量写法，与生成代码里的 sma 逐次浮点
    运算顺序完全一致——A股价格常出现 MA 完全相等的临界情形，累加顺序不同会让
    `MA5 > MA20` 这类严格比较在最后一位翻转。"""
    N, T = x.shape
    out = np.full((N, T), np.nan)
    s = np.zeros(N, dtype=np.float64)
    for t in range(T):
        s = s + x[:, t]
        if t >= n:
            s = s - x[:, t - n]
        if t >= n - 1:
            out[:, t] = s / n
    return out


def ema(x, n):
    k = 2.0 / (n + 1)
    out = np.full(x.shape, np.nan)
    prev = np.full(x.shape[0], np.nan)
    for t in range(x.shape[1]):
        col = x[:, t]
        has = ~np.isnan(col)
        prev = np.where(has & np.isnan(prev), col, np.where(has, col * k + prev * (1 - k), prev))
        out[:, t] = np.where(has, prev, np.nan)
    return out


def macd(close, fast=12, slow=26, sig=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, sig)
    return dif, dea, (dif - dea) * 2


def rsi(close, n):
    """Wilder 平滑：前 n 步累加 u/n、d/n 并在第 n 步输出，之后递推（与生成代码同步）"""
    N, T = close.shape
    out = np.full((N, T), np.nan)
    au = np.zeros(N)
    ad = np.zeros(N)
    cnt = np.zeros(N, dtype=np.int64)
    prev = np.full(N, np.nan)
    for t in range(T):
        c = close[:, t]
        has = ~np.isnan(c)
        step = has & ~np.isnan(prev)
        ch = np.where(step, c - prev, 0.0)
        u, d = np.maximum(ch, 0.0), np.maximum(-ch, 0.0)
        k = cnt + 1
        early, late = step & (k <= n), step & (k > n)
        au = np.where(early, au + u / n, np.where(late, (au * (n - 1) + u) / n, au))
        ad = np.where(early, ad + d / n, np.where(late, (ad * (n - 1) + d) / n, ad))
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(ad == 0, 100.0, 100.0 - 100.0 / (1.0 + au / np.where(ad == 0, 1.0, ad)))
        out[:, t] = np.where((early & (k == n)) | late, val, np.nan)
        cnt = np.where(step, k, cnt)
        prev = np.where(has, c, prev)
    return out


def _roll(x, n, fn):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(x.shape, np.nan)
    if x.shape[1] >= n:
        with np.errstate(invalid="ignore"):
            out[:, n - 1:] = fn(sliding_window_view(x, n, axis=1), axis=2)
    return out


def roll_max(x, n):
    return _roll(x, n, np.max)


def roll_min(x, n):
    return _roll(x, n, np.min)


def kdj(high, low, close, n=9):
    """RSV 用近 n 根（不足 n 根时用已有全部，与生成代码 max(0, i-n+1) 一致），K/D 三分递推"""
    hh = np.full(close.shape, np.nan)
    ll = np.full(close.shape, np.nan)
    for j in range(close.shape[1]):
        s = max(0, j - n + 1)
        with np.errstate(invalid="ignore"):
            hh[:, j] = np.nanmax(high[:, s:j + 1], axis=1)
            ll[:, j] = np.nanmin(low[:, s:j + 1], axis=1)
    with np.errstate(invalid="ignore"):
        rsv = np.where(hh == ll, 50.0, (close - ll) / np.where(hh == ll, 1.0, hh - ll) * 100.0)
    N, T = close.shape
    K = np.full((N, T), np.nan)
    D = np.full((N, T), np.nan)
    J = np.full((N, T), np.nan)
    pk = np.full(N, 50.0)
    pd = np.full(N, 50.0)
    for t in range(T):
        has = ~np.isnan(close[:, t])
        r = np.where(has, rsv[:, t], 50.0)
        pk = np.where(has, (r + 2 * pk) / 3.0, pk)
        pd = np.where(has, (pk + 2 * pd) / 3.0, pd)
        K[:, t] = np.where(has, pk, np.nan)
        D[:, t] = np.where(has, pd, np.nan)
        J[:, t] = np.where(has, 3 * pk - 2 * pd, np.nan)
    return K, D, J


def boll(close, n=20, k=2.0):
    """方差按窗口内从左到右累加，与生成代码的 sum(...) 求和顺序一致"""
    mid = sma(close, n)
    from numpy.lib.stride_tricks import sliding_window_view
    up = np.full(close.shape, np.nan)
    dn = np.full(close.shape, np.nan)
    if close.shape[1] >= n:
        w = sliding_window_view(close, n, axis=1)
        m = mid[:, n - 1:]
        acc = np.zeros(m.shape, dtype=np.float64)
        for j in range(n):
            acc = acc + (w[:, :, j] - m) ** 2
        sd = np.sqrt(acc / n)
        up[:, n - 1:] = m + k * sd
        dn[:, n - 1:] = m - k * sd
    return mid, up, dn


def bias(close, n):
    m = sma(close, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((~np.isnan(m)) & (m != 0), (close - m) / m * 100.0, np.nan)


def wr(high, low, close, n):
    hh, ll = roll_max(high, n), roll_min(low, n)
    with np.errstate(invalid="ignore"):
        return np.where(hh == ll, 50.0, (hh - close) / np.where(hh == ll, 1.0, hh - ll) * 100.0)


def cci(high, low, close, n):
    """平均绝对偏差同样按窗口从左到右累加，保持与生成代码一致的求和顺序"""
    from numpy.lib.stride_tricks import sliding_window_view
    tp = (high + low + close) / 3.0
    m = sma(tp, n)
    out = np.full(close.shape, np.nan)
    if close.shape[1] >= n:
        w = sliding_window_view(tp, n, axis=1)
        mm = m[:, n - 1:]
        acc = np.zeros(mm.shape, dtype=np.float64)
        for j in range(n):
            acc = acc + np.abs(w[:, :, j] - mm)
        md = acc / n
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, n - 1:] = np.where(md == 0, 0.0, (tp[:, n - 1:] - mm) / (0.015 * md))
    return out


def limit_pct_of(codes):
    """涨停幅度上限：主板 10%、创业板/科创板 20%、北交所 30%（ST 的 5% 不单独识别）"""
    out = np.full(len(codes), 9.5)
    for i, c in enumerate(codes):
        c = str(c)
        if c[:2] in ("30", "68"):
            out[i] = 19.5
        elif c[:2] in ("43", "83", "87", "92") or c[0] in ("4", "8"):
            out[i] = 29.5
    return out


def limit_up_count(chg, high, close, n, pct):
    """近 n 日涨停收盘次数：涨幅达板块上限且收盘价=最高价"""
    with np.errstate(invalid="ignore"):
        hit = (chg >= pct[:, None]) & (close >= high - 1e-6)
    lu = np.where(np.isnan(chg), 0.0, hit.astype(np.float64))
    cs = np.cumsum(lu, axis=1)
    s = cs[:, n - 1:].copy()
    s[:, 1:] -= cs[:, :-n]
    out = np.full(chg.shape, np.nan)
    out[:, n - 1:] = s
    return out


class Indicators:
    """按需计算并缓存指标矩阵"""

    def __init__(self, pnl):
        self.p = pnl
        self.c = {}

    def get(self, key):
        if key in self.c:
            return self.c[key]
        f = self.p.cf
        if key in f:
            v = f[key]
        elif key.startswith("macd_"):
            dif, dea, hist = macd(f["close"])
            self.c.update(macd_dif=dif, macd_dea=dea, macd_hist=hist)
            return self.c[key]
        elif key.startswith("kdj_"):
            K, D, J = kdj(f["high"], f["low"], f["close"])
            self.c.update(kdj_k=K, kdj_d=D, kdj_j=J)
            return self.c[key]
        elif key.startswith("boll_"):
            m, u, d = boll(f["close"])
            self.c.update(boll_mid=m, boll_up=u, boll_dn=d)
            return self.c[key]
        elif key.startswith("vma"):
            v = sma(f["volume"], int(key[3:]))
        elif key.startswith("ma"):
            v = sma(f["close"], int(key[2:]))
        elif key.startswith("rsi"):
            v = rsi(f["close"], int(key[3:]))
        elif key.startswith("bias"):
            v = bias(f["close"], int(key[4:]))
        elif key.startswith("wr"):
            v = wr(f["high"], f["low"], f["close"], int(key[2:]))
        elif key.startswith("cci"):
            v = cci(f["high"], f["low"], f["close"], int(key[3:]))
        elif key.startswith("lu"):
            v = limit_up_count(f["chg"], f["high"], f["close"], int(key[2:]),
                               limit_pct_of(self.p.codes))
        elif key.startswith("hh"):
            v = roll_max(f["high"], int(key[2:]))
        elif key.startswith("ll"):
            v = roll_min(f["low"], int(key[2:]))
        elif key == "fcap":
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(f["turnover"] > 0, f["volume"] * 1e4 / f["turnover"] * f["close"], np.nan)
        else:
            raise KeyError(key)
        self.c[key] = v
        return v


RAW = {"close": "close", "open": "open", "high": "high", "low": "low",
       "turnover": "turnover", "chg": "chg", "amp": "amp", "volume": "volume",
       "amount": "amount", "fcap": "fcap", "kdjK": "kdj_k", "kdjD": "kdj_d",
       "kdjJ": "kdj_j", "macdDif": "macd_dif", "bollUp": "boll_up",
       "bollDn": "boll_dn", "bollMid": "boll_mid"}
SPAN = {"ma": "ma", "rsi": "rsi", "bias": "bias", "wr": "wr", "cci": "cci", "limitup": "lu"}


def operand(ind, o):
    if o.get("const") is not None:
        return float(o["const"])
    key = SPAN[o["ind"]] + str(o["n"]) if o["ind"] in SPAN else RAW[o["ind"]]
    v = ind.get(key)
    return v * o["k"] if o.get("k") else v


def _prev(x):
    y = np.full_like(x, np.nan)
    y[:, 1:] = x[:, :-1]
    return y


def cond_mask(ind, c):
    k = c["kind"]
    if k in ("cross", "macdCross", "kdjCross"):
        if k == "cross":
            a, b = ind.get("ma" + str(c["a"]["n"])), ind.get("ma" + str(c["b"]["n"]))
        elif k == "macdCross":
            a, b = ind.get("macd_dif"), ind.get("macd_dea")
        else:
            a, b = ind.get("kdj_k"), ind.get("kdj_d")
        pa, pb = _prev(a), _prev(b)
        with np.errstate(invalid="ignore"):
            m = (pa <= pb) & (a > b) if c["dir"] == "up" else (pa >= pb) & (a < b)
        return m & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(pa) & ~np.isnan(pb)
    if k == "breakout":
        ref = _prev(ind.get(("hh" if c["dir"] == "up" else "ll") + str(c["n"])))
        close = ind.get("close")
        with np.errstate(invalid="ignore"):
            m = close > ref if c["dir"] == "up" else close < ref
        return m & ~np.isnan(ref) & ~np.isnan(close)
    if k == "streak":
        close = ind.get("close")
        m = np.ones(close.shape, dtype=bool)
        cur = close
        for _ in range(c["n"]):
            prv = _prev(cur)
            with np.errstate(invalid="ignore"):
                d = cur - prv
                step = (d > 0) if c["dir"] == "up" else (d < 0)
            m &= step & ~np.isnan(d)
            cur = prv
        return m
    if k in ("volSpike", "volShrink"):
        v, m0 = ind.get("volume"), ind.get("vma" + str(c["n"]))
        with np.errstate(invalid="ignore"):
            m = v > m0 * c["mult"] if k == "volSpike" else v < m0 * c["mult"]
        return m & ~np.isnan(m0) & ~np.isnan(v)
    if k == "cmp":
        a, b = operand(ind, c["a"]), operand(ind, c["b"])
        op = c["op"]
        with np.errstate(invalid="ignore"):
            m = {">": lambda: a > b, "<": lambda: a < b, ">=": lambda: a >= b,
                 "<=": lambda: a <= b, "==": lambda: a == b}[op]()
        m = np.asarray(m)
        if np.ndim(a):
            m = m & ~np.isnan(a)
        if np.ndim(b):
            m = m & ~np.isnan(b)
        return m
    raise ValueError("未知条件类型：" + k)


def group_mask(ind, g, shape):
    if not g["conds"]:
        return np.zeros(shape, dtype=bool)
    out = None
    for c in g["conds"]:
        m = cond_mask(ind, c)
        out = m.copy() if out is None else ((out | m) if g["op"] == "or" else (out & m))
    return out


# ────────────────────────────── 回测 ──────────────────────────────
def backtest(pnl, ind, spec, start, end, params, trace=None, pick="amount", seed=7):
    f = pnl.cf
    shape = f["close"].shape
    buy_m = group_mask(ind, spec["buy"], shape)
    sell_m = group_mask(ind, spec["sell"], shape)
    too_short = pnl.n_bars < 30          # 生成代码里 run() 会跳过不足 30 根K线的标的
    buy_m[too_short] = False
    sell_m[too_short] = False

    o, c = f["open"], f["close"]
    amt = f["amount"]
    with np.errstate(invalid="ignore"):
        untradable = (o == c) & (c == f["high"]) & (f["high"] == f["low"]) & (np.abs(f["chg"]) > 9.5)
    bar, has = pnl.bar, pnl.has
    N = pnl.N
    rows_all = np.arange(N)

    i0, i1 = pnl.col_range(start, end)
    cash = float(params["init_cash"])
    fee_r = params["fee_permil"] / 1000.0
    slip_r = params["slip_permil"] / 1000.0
    budget_cap = params["init_cash"] * params["pos_pct"] / 100.0
    max_hold = int(params["max_hold"])
    min_bars = int(params.get("min_bars", 1))
    tp, sl, mb = params["take_profit"], params["stop_loss"], params["max_bars"]

    rng = np.random.default_rng(seed)
    pos = {}
    trades, equity = [], []
    pend_sell = []
    pend_rows = np.empty(0, dtype=np.int64)
    pend_amt = np.empty(0)

    for t in range(i0, i1):
        day = int(pnl.cal[t])
        col = bar[:, t]
        alive = col >= 0

        # 1) 昨日卖出委托 → 今日开盘成交
        for si, reason in pend_sell:
            p = pos.get(si)
            i = int(col[si])
            if p is None or i < 0 or untradable[si, i]:
                continue
            px = o[si, i] * (1 - slip_r)
            gross = px * p["shares"]
            fee = max(gross * fee_r, 5.0)
            cash += gross - fee
            cost = p["price"] * p["shares"] + p["fee"]
            gain = gross - fee - cost
            trades.append({"code": pnl.codes[si], "name": pnl.names[si], "buy_date": p["date"],
                           "buy_price": round(float(p["price"]), 3), "sell_date": day,
                           "sell_price": round(float(px), 3), "shares": int(p["shares"]),
                           "pnl": round(float(gain), 2), "ret": round(float(gain / cost), 6) if cost else 0.0,
                           "bars": int(i - p["bar_i"]), "reason": reason})
            pos.pop(si, None)
        pend_sell = []

        # 2) 昨日买入委托 → 按成交额优先成交
        if len(pend_rows):
            if pick == "random":                              # 敏感性对照：候选随机挑选
                order = rng.permutation(len(pend_rows))
            else:                                             # 默认：金额降序，并列时代码（=行号）降序
                order = np.lexsort((-pend_rows, -pend_amt))
            for k in order:
                si = int(pend_rows[k])
                if len(pos) >= max_hold or si in pos:
                    continue
                i = int(col[si])
                if i < 0 or untradable[si, i]:
                    continue
                px = o[si, i] * (1 + slip_r)
                shares = int(min(cash, budget_cap) / px / 100) * 100
                if shares < 100:
                    continue
                gross = px * shares
                fee = max(gross * fee_r, 5.0)
                if gross + fee > cash:
                    continue
                cash -= gross + fee
                pos[si] = {"bar_i": i, "date": day, "price": float(px), "shares": shares, "fee": fee}
            pend_rows = np.empty(0, dtype=np.int64)
            pend_amt = np.empty(0)

        # 3) 收盘后产生次日委托
        for si, p in pos.items():
            i = int(col[si])
            if i < 1 or (i - p["bar_i"]) < min_bars:
                continue
            ret = float(c[si, i]) / p["price"] - 1.0
            if tp and ret >= tp:
                pend_sell.append((si, "止盈"))
            elif sl and ret <= -sl:
                pend_sell.append((si, "止损"))
            elif mb and (i - p["bar_i"]) >= mb:
                pend_sell.append((si, "到期"))
            elif sell_m[si, i]:
                pend_sell.append((si, "信号"))
        cand = alive & (col >= 1)
        if cand.any():
            r = rows_all[cand]
            ci = col[r]
            sig = buy_m[r, ci]
            r = r[sig]
            if len(r):
                keep = np.array([si not in pos for si in r], dtype=bool)
                r = r[keep]
                pend_rows = r.astype(np.int64)
                pend_amt = amt[r, col[r]]

        # 4) 当日权益
        mv = 0.0
        for si, p in pos.items():
            i = int(col[si])
            mv += (float(c[si, i]) if i >= 0 else p["price"]) * p["shares"]
        equity.append([day, round(cash + mv, 2)])
        if trace is not None:
            trace.append((day, sorted(pnl.codes[int(x)] for x in pend_rows),
                          sorted(pnl.codes[si] for si in pos),
                          round(cash, 2), sorted(pnl.codes[si] for si, _r in pend_sell)))

    # 区间结束平仓
    for si, p in list(pos.items()):
        cols = np.nonzero(has[si, :i1])[0]
        if not len(cols):
            continue
        i = int(bar[si, cols[-1]])
        px = float(c[si, i])
        gross = px * p["shares"]
        fee = max(gross * fee_r, 5.0)
        cash += gross - fee
        cost = p["price"] * p["shares"] + p["fee"]
        gain = gross - fee - cost
        trades.append({"code": pnl.codes[si], "name": pnl.names[si], "buy_date": p["date"],
                       "buy_price": round(float(p["price"]), 3), "sell_date": int(pnl.dates[si, i]),
                       "sell_price": round(float(px), 3), "shares": int(p["shares"]), "pnl": round(float(gain), 2),
                       "ret": round(float(gain / cost), 6) if cost else 0.0,
                       "bars": int(i - p["bar_i"]), "reason": "区间结束平仓"})
        pos.pop(si)

    trades.sort(key=lambda x: (x["buy_date"], x["code"]))
    return {"trades": trades, "equity": equity, "init_cash": params["init_cash"],
            "final_cash": round(cash, 2), "start": start, "end": end, "engine": "numpy"}


TRACE = []


def backtest_traced(pnl, ind, spec, start, end, params):
    """调试用：复用 backtest，但逐日记录候选/持仓/现金，便于与生成代码逐日比对"""
    global TRACE
    TRACE = []
    return backtest(pnl, ind, spec, start, end, params, trace=TRACE)


# ─────────────────── 信号级等权回测（不受仓位/现金约束） ───────────────────
def signal_backtest(pnl, ind, spec, start, end, params, universe=None):
    """把策略的每一个买入信号都成交，等权、无持仓数量上限，用来衡量策略本身的胜率与单笔收益。

    与资金约束版共用同一套信号与退出规则（收盘信号 → 次日开盘成交、T+1、止盈/止损/
    到期/信号退出、一字板不成交），区别只是不做仓位与现金竞争，因此不受「候选多于
    仓位时先挑谁」这一人为规则的影响。同时按「当日所有在持仓位等权」合成一条日收益
    曲线，用于计算年化、回撤与夏普。
    """
    f = pnl.cf
    shape = f["close"].shape
    buy_m = group_mask(ind, spec["buy"], shape)
    sell_m = group_mask(ind, spec["sell"], shape)
    too_short = pnl.n_bars < 30
    buy_m[too_short] = False
    sell_m[too_short] = False
    if universe is not None:                 # 只在指定标的池里开仓（持仓退出规则不变）
        buy_m[~np.asarray(universe, dtype=bool)] = False

    o, c = f["open"], f["close"]
    with np.errstate(invalid="ignore"):
        untradable = (o == c) & (c == f["high"]) & (f["high"] == f["low"]) & (np.abs(f["chg"]) > 9.5)
    i0, i1 = pnl.col_range(start, end)
    fee_r = params["fee_permil"] / 1000.0
    slip_r = params["slip_permil"] / 1000.0
    tp, sl, mb = params["take_profit"], params["stop_loss"], params["max_bars"]
    min_bars = int(params.get("min_bars", 1))
    # retry_days=0：次日买不到（一字板）就放弃该信号；>0：最多向后顺延这么多个交易日
    retry_days = int(params.get("retry_days", 0))

    T = pnl.T
    day_sum = np.zeros(T)        # 当日所有在持仓位的收益之和
    day_cnt = np.zeros(T)        # 当日在持仓位数
    trades = []
    missed = 0                   # 因一字板买不到而放弃的信号数
    col_of = pnl.bar             # [N, T]
    for si in range(pnl.N):
        n = int(pnl.n_bars[si])
        if n < 30:
            continue
        cols = np.nonzero(pnl.has[si])[0]           # 该标的有K线的交易日列
        if not len(cols):
            continue
        bar2col = np.empty(n, dtype=np.int64)
        bar2col[col_of[si, cols]] = cols
        lo = int(np.searchsorted(bar2col, i0, "left"))
        hi = int(np.searchsorted(bar2col, i1, "left"))
        if hi - lo < 3:
            continue
        bm, sm = buy_m[si], sell_m[si]
        oo, cc = o[si], c[si]
        ut = untradable[si]
        j = lo
        while j < hi - 1:
            if not bm[j] or j < 1:
                j += 1
                continue
            e = j + 1                                # 次日开盘成交
            waited = 0
            while e < hi and ut[e] and waited < retry_days:   # 一字板顺延
                e += 1
                waited += 1
            if e >= hi:
                break
            if ut[e]:                                # 仍买不到 → 放弃该信号
                missed += 1
                j += 1
                continue
            price = oo[e] * (1 + slip_r)
            k = e
            reason, xi = None, None
            while k + 1 < hi:
                if (k - e) >= min_bars:
                    ret = cc[k] / price - 1.0
                    if tp and ret >= tp:
                        reason = "止盈"
                    elif sl and ret <= -sl:
                        reason = "止损"
                    elif mb and (k - e) >= mb:
                        reason = "到期"
                    elif sm[k]:
                        reason = "信号"
                    if reason:
                        xi = k + 1
                        while xi < hi and ut[xi]:
                            xi += 1
                        break
                k += 1
            if reason is None:
                xi, reason = hi - 1, "区间结束平仓"
            if xi is None or xi >= hi:
                xi, reason = hi - 1, "区间结束平仓"
            exit_px = oo[xi] * (1 - slip_r) if reason != "区间结束平仓" else cc[xi]
            gross = exit_px / price - 1.0
            net = (1 + gross) * (1 - fee_r) ** 2 - 1.0
            trades.append({"code": pnl.codes[si], "name": pnl.names[si],
                           "buy_date": int(pnl.dates[si, e]), "buy_price": round(float(price), 3),
                           "sell_date": int(pnl.dates[si, xi]), "sell_price": round(float(exit_px), 3),
                           "ret": round(float(net), 6), "gross_ret": round(float(gross), 6),
                           "bars": int(xi - e), "reason": reason})
            # 摊到每日等权收益：建仓日用收盘/买入价，中间用收盘/前收盘，平仓日用卖出价/前收盘
            for b in range(e, xi + 1):
                t = int(bar2col[b])
                if b == e:
                    r = cc[b] / price - 1.0 - fee_r
                elif b == xi:
                    r = exit_px / cc[b - 1] - 1.0 - fee_r
                else:
                    r = cc[b] / cc[b - 1] - 1.0
                day_sum[t] += r
                day_cnt[t] += 1
            j = xi                                   # 同一标的不重叠持仓
        # 循环结束
    daily = np.where(day_cnt > 0, day_sum / np.maximum(day_cnt, 1), 0.0)[i0:i1]
    eq = np.cumprod(1 + daily) * 1.0
    return {"trades": trades, "daily": daily, "equity_curve": eq, "missed": missed,
            "dates": pnl.cal[i0:i1], "avg_positions": float(day_cnt[i0:i1].mean())}


def signal_metrics(r):
    t = r["trades"]
    rets = np.array([x["ret"] for x in t]) if t else np.array([])
    eq = r["equity_curve"]
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    mdd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    d = r["daily"]
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    total = float(eq[-1] - 1) if len(eq) else 0.0
    days = len(d)
    wins = rets[rets > 0]
    loss = rets[rets <= 0]
    # 与「并发持仓数」无关的收益口径：单笔几何平均 + 按平均持有期折算的单仓复利年化
    geo = float(np.expm1(np.mean(np.log1p(np.maximum(rets, -0.999))))) if len(rets) else 0.0
    avg_bars = float(np.mean([x["bars"] for x in t])) if t else 0.0
    turns = 244.0 / avg_bars if avg_bars > 0 else 0.0
    ann_single = (1 + geo) ** turns - 1 if geo > -1 else -1.0
    return {
        "trades": len(t), "wins": int((rets > 0).sum()), "losses": int((rets <= 0).sum()),
        "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "geo_ret": geo, "turns_per_year": turns, "annual_single": float(ann_single),
        "avg_ret": float(rets.mean()) if len(rets) else 0.0,
        "median_ret": float(np.median(rets)) if len(rets) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(loss.mean()) if len(loss) else 0.0,
        "profit_factor": float(wins.sum() / -loss.sum()) if len(loss) and loss.sum() < 0 else float("inf"),
        "expectancy": float(rets.mean()) if len(rets) else 0.0,
        "total_return": total,
        "annual_return": (1 + total) ** (244 / days) - 1 if days > 5 else total,
        "max_drawdown": mdd,
        "sharpe": float(np.mean(d) / sd * np.sqrt(244)) if sd > 0 else 0.0,
        "avg_bars": float(np.mean([x["bars"] for x in t])) if t else 0.0,
        "avg_positions": r["avg_positions"],
        "best": float(rets.max()) if len(rets) else 0.0,
        "worst": float(rets.min()) if len(rets) else 0.0,
        "stocks_traded": len({x["code"] for x in t}),
        "missed_signals": r.get("missed", 0),
    }


# ────────────────────────────── 统计 ──────────────────────────────
def metrics(r):
    t = r["trades"]
    eq = np.array([e[1] for e in r["equity"]], dtype=np.float64) if r["equity"] \
        else np.array([float(r["init_cash"])])
    wins = [x for x in t if x["pnl"] > 0]
    loss = [x for x in t if x["pnl"] <= 0]
    gp = sum(x["pnl"] for x in wins)
    gl = -sum(x["pnl"] for x in loss)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    total = float(eq[-1] / r["init_cash"] - 1)
    days = len(eq)
    return {
        "trades": len(t), "wins": len(wins), "losses": len(loss),
        "win_rate": (len(wins) / len(t)) if t else 0.0,
        "total_return": total,
        "annual_return": (1 + total) ** (244 / days) - 1 if days > 5 else total,
        "max_drawdown": mdd,
        "sharpe": float(np.mean(rets) / sd * np.sqrt(244)) if sd > 0 else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "avg_ret": float(np.mean([x["ret"] for x in t])) if t else 0.0,
        "avg_win": float(np.mean([x["ret"] for x in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([x["ret"] for x in loss])) if loss else 0.0,
        "avg_bars": float(np.mean([x["bars"] for x in t])) if t else 0.0,
        "best": max((x["ret"] for x in t), default=0.0),
        "worst": min((x["ret"] for x in t), default=0.0),
        "final_equity": float(eq[-1]), "bars_traded": days,
        "total_pnl": round(sum(x["pnl"] for x in t), 2),
    }
