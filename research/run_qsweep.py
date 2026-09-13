"""P1-3 q-sweep (E04/E05): Top5 q full sweep.

Sweeps q in {0, 0.1, 0.2, 0.3, 0.4, 0.5} on the locked Top5 basket
(ETC/TRX/ATOM/APT/KAS, equal 20%, E10 FORMULA, aster perp 2x,
fund 0.0005, base fee 0.0004 / fee2x 0.0008).

Engine mirrors research/run_aa.py leg_series + strategy_manager/y1b_basket.py:
quantile_mask_long(q) on long leg only + cooldown + stops + vol_scale
(vt None -> 1.0) + roll1. Per-q report: FULL/H2 sharpe, turnover,
long/short entries + net-PnL share, bull/bear split (BTC close vs 200-bar
MA on the same 4h grid), fee2x net-sharpe curve + knee.

Offline read-only: only reads data/data_15m_3y/*.csv, writes
results/qsweep.json (+ logs/qsweep.log). No orders, no env switches.
"""
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import (
    FORMULA,
    LOCKED_ATOM,
    LOCKED_APT,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
    FEE2X,
)

QS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BASKET_SPECS = {
    "ETC": LOCKED_ETC,
    "TRX": LOCKED_TRX,
    "ATOM": LOCKED_ATOM,
    "APT": LOCKED_APT,
    "KAS": LOCKED_KAS,
}
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
W = 1.0 / len(COINS)
BARS_PER_YEAR = 2190.0
MA_WIN = 200
H2_LEN = 493

OUT = pathlib.Path("results/qsweep.json")
LOG = pathlib.Path("logs/qsweep.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load_15m(coin):
    rows = list(csv.DictReader(open("data/data_15m_3y/%s.csv" % coin)))
    out = []
    for r in rows:
        out.append((int(r["timestamp"]), float(r["open"]), float(r["high"]),
                    float(r["low"]), float(r["close"]), float(r["volume"])))
    return out


def common_4h(coins):
    """Timestamp-intersected 4h bars; returns (bars_map, closes_map)."""
    raw = {c: load_15m(c) for c in coins}
    start = max(r[0][0] for r in raw.values())
    end = min(r[-1][0] for r in raw.values())
    bars, closes = {}, {}
    for c in coins:
        rows = [r for r in raw[c] if start <= r[0] <= end]
        n4 = len(rows) // 16
        b4 = []
        for i in range(n4):
            blk = rows[i * 16:(i + 1) * 16]
            b4.append((blk[0][1], max(r[2] for r in blk),
                       min(r[3] for r in blk), blk[-1][4],
                       sum(r[5] for r in blk)))
        bars[c] = b4
        closes[c] = [b[3] for b in b4]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
        closes[c] = closes[c][:n]
    return bars, closes


def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]),
           "high": torch.tensor([[b[1] for b in bars]]),
           "low": torch.tensor([[b[2] for b in bars]]),
           "close": torch.tensor([[b[3] for b in bars]]),
           "volume": torch.tensor([[b[4] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3]
            for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def quantile_mask_long(sig, q):
    if q is None or float(q) <= 0.0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_pnl(raw, rets_t, sig, spec, fee, fund, q):
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BARS_PER_YEAR,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
    signal = torch.sigmoid(sig)
    is_safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = quantile_mask_long(sig, q)
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    scale = bt._vol_scale(rets_t)
    lp, sp = lp * scale, sp * scale
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    lturn = (lp - lp.roll(1, dims=1)).abs()
    sturn = (sp - sp.roll(1, dims=1)).abs()
    rate = bt.base_fee
    gross = (lp - sp) * rets_t * bt.leverage
    tx = turn * rate
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    gross_l = (gross - gross * 0.0)[0].tolist()
    lp_l = lp[0].tolist()
    sp_l = sp[0].tolist()
    lt_l = lturn[0].tolist()
    st_l = sturn[0].tolist()
    rt_l = rets_t[0].tolist()
    # side net decomposition: net == long_net + short_net per bar
    long_net, short_net = [], []
    for t in range(len(net)):
        ln = lp_l[t] * rt_l[t] * LEV - lt_l[t] * rate * LEV - lp_l[t] * fund * LEV
        sn = -sp_l[t] * rt_l[t] * LEV - st_l[t] * rate * LEV + sp_l[t] * fund * LEV
        long_net.append(ln)
        short_net.append(sn)
    pos = (lp - sp)[0].tolist()
    turn_l = turn[0].tolist()
    entries_l, entries_s = 0, 0
    prev = 0.0
    for v in pos:
        cur = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if cur != 0.0 and prev == 0.0:
            if cur > 0:
                entries_l += 1
            else:
                entries_s += 1
        prev = cur
    return {"net": net, "turn": turn_l, "long_net": long_net,
            "short_net": short_net, "entries_l": entries_l,
            "entries_s": entries_s}


def sharpe(ser):
    n = len(ser)
    if n < 2:
        return 0.0
    mean = sum(ser) / n
    var = sum((x - mean) ** 2 for x in ser) / (n - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(BARS_PER_YEAR)


def seg_stats(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    mean = sum(s) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = mean / math.sqrt(var) * math.sqrt(BARS_PER_YEAR) if var > 0 else 0.0
    cum = sum(s)
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in s:
        cs += x
        peak = max(peak, cs)
        mdd = max(mdd, peak - cs)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BARS_PER_YEAR, 4) if n else 0.0,
            "mdd": round(mdd, 4), "cum": round(cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def side_split(long_net, short_net, entries_l, entries_s, a, b):
    ln = sum(long_net[a:b])
    sn = sum(short_net[a:b])
    tot = ln + sn
    if abs(tot) < 1e-12:
        shr_l, shr_s = 0.0, 0.0
    else:
        shr_l = ln / abs(tot) if tot >= 0 else -ln / abs(tot)
        shr_s = sn / abs(tot) if tot >= 0 else -sn / abs(tot)
        shr_l = round(shr_l, 4)
        shr_s = round(shr_s, 4)
    return {"long_entries": entries_l, "short_entries": entries_s,
            "long_pnl": round(ln, 4), "short_pnl": round(sn, 4),
            "long_share": shr_l, "short_share": shr_s}


def mask_sharpe(net, mask):
    s = [net[t] for t in range(len(net)) if mask[t]]
    return round(sharpe(s), 3), len(s)


def find_knee(xs, ys):
    n = len(xs)
    if n < 3:
        return xs[0], 0
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-12:
        return xs[0], 0
    best_i, best_d = 0, -1e18
    for i in range(n):
        d = abs(dy * xs[i] - dx * ys[i] + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_d, best_i = d, i
    return xs[best_i], best_i


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("qsweep start\n")
    bars, closes = common_4h(COINS + ["BTC"])
    n = len(bars["ETC"])
    log("common 4h bars n=%d %s" % (n, {c: len(bars[c]) for c in bars}))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    btc = closes["BTC"]
    ma = []
    for t in range(n):
        w = btc[max(0, t - MA_WIN + 1):t + 1]
        ma.append(sum(w) / len(w))
    bull = [1 if btc[t] >= ma[t] else 0 for t in range(n)]
    n_bull = sum(bull)
    log("bull bars=%d/%d (%.3f) BTC close>=MA%d" % (n_bull, n, n_bull / n, MA_WIN))

    h2a = n - H2_LEN
    rows = []
    for q in QS:
        legs = {}
        for c in COINS:
            raw, rt, sg = mats[c]
            legs[c] = leg_pnl(raw, rt, sg, BASKET_SPECS[c], FEE, FUND, q)
        net = [sum(legs[c]["net"][t] * W for c in COINS) for t in range(n)]
        turn = [sum(legs[c]["turn"][t] * W for c in COINS) for t in range(n)]
        lnet = [sum(legs[c]["long_net"][t] * W for c in COINS) for t in range(n)]
        snet = [sum(legs[c]["short_net"][t] * W for c in COINS) for t in range(n)]
        el = sum(legs[c]["entries_l"] for c in COINS)
        es = sum(legs[c]["entries_s"] for c in COINS)

        full = seg_stats(net, turn, 0, n)
        h2 = seg_stats(net, turn, h2a, n)

        legs2 = {}
        for c in COINS:
            raw, rt, sg = mats[c]
            legs2[c] = leg_pnl(raw, rt, sg, BASKET_SPECS[c], FEE2X, FUND, q)
        net2 = [sum(legs2[c]["net"][t] * W for c in COINS) for t in range(n)]
        turn2 = [sum(legs2[c]["turn"][t] * W for c in COINS) for t in range(n)]
        full2 = seg_stats(net2, turn2, 0, n)
        h22 = seg_stats(net2, turn2, h2a, n)

        bull_sh, bull_n = mask_sharpe(net, bull)
        bear_mask = [1 - b for b in bull]
        bear_sh, bear_n = mask_sharpe(net, bear_mask)

        row = {"q": q, "FULL": full, "H2": h2,
               "FULL_fee2x": full2, "H2_fee2x": h22,
               "attribution_full": side_split(lnet, snet, el, es, 0, n),
               "attribution_h2": side_split(lnet, snet, el, es, h2a, n),
               "bull": {"sharpe": bull_sh, "n": bull_n},
               "bear": {"sharpe": bear_sh, "n": bear_n}}
        rows.append(row)
        log("q=%.1f FULL sh=%.3f ann=%.3f dd=%.4f to=%.4f | H2 sh=%.3f to=%.4f "
            "| FULL2x=%.3f H22x=%.3f | L/S=%d/%d pnl=%.3f/%.3f | bull=%.3f bear=%.3f"
            % (q, full["sharpe"], full["ann"], full["mdd"], full["turnover"],
               h2["sharpe"], h2["turnover"], full2["sharpe"], h22["sharpe"],
               el, es, row["attribution_full"]["long_pnl"],
               row["attribution_full"]["short_pnl"], bull_sh, bear_sh))

    curve = [r["FULL_fee2x"]["sharpe"] for r in rows]
    knee_q, knee_i = find_knee(QS, curve)
    r03 = next(r for r in rows if abs(r["q"] - 0.3) < 1e-9)
    others = [r["FULL_fee2x"]["sharpe"] for r in rows if abs(r["q"] - 0.3) > 1e-9]
    mean_others = sum(others) / len(others) if others else 0.0
    peak_gap = (r03["FULL_fee2x"]["sharpe"] - mean_others) / max(abs(mean_others), 1e-9)
    isolated = bool(peak_gap > 0.15 and all(
        r03["FULL_fee2x"]["sharpe"] > r["FULL_fee2x"]["sharpe"] for r in rows
        if abs(r["q"] - 0.3) > 1e-9))
    base_curve = [r["FULL"]["sharpe"] for r in rows]
    base03 = r03["FULL"]["sharpe"]
    base_others = [v for qv, v in zip(QS, base_curve) if abs(qv - 0.3) > 1e-9]
    base_gap = (base03 - sum(base_others) / len(base_others)) / max(
        abs(sum(base_others) / len(base_others)), 1e-9)
    if isolated:
        decision = "ADOPT_KNEE_q%.1f" % knee_q
        note = ("q0.3 isolated peak (fee2x gap %+.1f%%); switch to knee q%.1f "
                "+ config note" % (peak_gap * 100, knee_q))
    else:
        decision = "KEEP_q0.3"
        note = ("q0.3 not isolated (fee2x gap %+.1f%%, base gap %+.1f%%); "
                "within-15%% robust, keep + record" % (peak_gap * 100, base_gap * 100))
    log("knee q=%.1f curve=%s decision=%s (%s)" % (knee_q, curve, decision, note))

    res = {"config": {
        "engine": "mirror run_aa.py leg_series + quantile_mask_long(q, long-only) "
                  "+ cooldown + stops + vol_scale(vt None -> 1.0) + roll1",
        "formula": list(FORMULA),
        "basket": {c: dict(BASKET_SPECS[c]) for c in COINS},
        "weights": {c: W for c in COINS},
        "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "fee2x": FEE2X,
        "qs": list(QS), "grid_bars": n, "h2_len": H2_LEN,
        "bull_bear": "BTC 4h close >= trailing %d-bar MA on common grid" % MA_WIN,
        "knee": "max perpendicular distance from chord on (q, FULL fee2x sharpe)",
        "attribution": "per-leg long_net/short_net split (fee+funding allocated "
                       "by side turnover/position); entries count 0->side flips",
        "note": "E10 FORMULA untouched; live chain untouched; offline read-only"},
        "rows": rows,
        "fee2x_curve": [{"q": q, "FULL_fee2x_sharpe": v} for q, v in zip(QS, curve)],
        "base_curve": [{"q": q, "FULL_sharpe": v} for q, v in zip(QS, base_curve)],
        "knee": {"q": knee_q, "idx": knee_i, "curve": curve},
        "q03_check": {"FULL_fee2x": r03["FULL_fee2x"]["sharpe"],
                      "FULL": r03["FULL"]["sharpe"],
                      "mean_others_fee2x": round(mean_others, 3),
                      "gap_pct": round(peak_gap * 100, 1),
                      "base_gap_pct": round(base_gap * 100, 1),
                      "isolated_peak": isolated},
        "decision": decision, "decision_note": note}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s decision=%s" % (OUT, decision))


if __name__ == "__main__":
    main()
