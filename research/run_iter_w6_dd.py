"""W6 15m drawdown anatomy (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05), so per the global exit rule
this step's conclusion is "PENDING" and MUST NOT be used as evidence for
any change. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/15m/*.csv only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), quantile q0.3 long-mask, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

15m NATIVE grid (no 4h aggregation): bars are raw 15m rows (~35040, 1y).
4h-bar params scale x16 inside the engine: cooldown_bars=cd*16,
time_stop=ts*16, vol_window=vw*16. BPY=35040.

Engine mirrors research/run_weight_modes.py leg_net exactly:
quantile q0.3 long-only mask + cooldown + stops + vol_scale(vt None->1.0)
+ roll1 execution lag; gross/fee/funding accounting at lev 2x.

Anatomy: top-3 non-overlapping drawdown windows per coin (standalone leg
net) + top-3 non-overlapping windows of the equal-weight portfolio net.
Per window: depth (peak-to-trough in cum-PnL units), length bars,
recovery bars (bars from trough to first re-touch of peak equity;
None when never recovered), window cum, per-coin contribution (weighted
leg sums over [peak,trough]), underlying vol (ann, per coin) and
pairwise correlation (mean + 5x5) of close-to-close returns in window.

Outputs: results/iter_W6_dd.json (+ logs/iter_w6_dd.log).
Incremental dump after each unit: partial JSON survives interrupts.

Smoke mode (for tests): ITER_W6_SMOKE=1 shrinks to the first 3000 bars
and top-1 window per series. OUT/LOG overridable via
ITER_W6_OUT / ITER_W6_LOG.
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
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(LOCKED_ETC),
    "TRX": dict(LOCKED_TRX),
    "ATOM": dict(LOCKED_ATOM),
    "APT": dict(LOCKED_APT),
    "KAS": dict(LOCKED_KAS),
}
W = {c: 0.2 for c in COINS}
BPY = 35040.0
SCALE = 16  # 4h-bar params -> 15m bars
K = 3

OUT = pathlib.Path(os.getenv("ITER_W6_OUT", "results/iter_W6_dd.json"))
LOG = pathlib.Path(os.getenv("ITER_W6_LOG", "logs/iter_w6_dd.log"))

SMOKE = os.getenv("ITER_W6_SMOKE") == "1"
SMOKE_N = 3000
SMOKE_K = 1


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, OUT)


def load15m(coin):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common15m(coins):
    """Timestamp-intersected native 15m bars (no aggregation)."""
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars, closes, stamps = {}, {}, {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [b[3] for b in bars[c]]
        stamps[c] = [r[0] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
        closes[c] = closes[c][:n]
        stamps[c] = stamps[c][:n]
    return bars, closes, stamps


def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]),
           "high": torch.tensor([[b[1] for b in bars]]),
           "low": torch.tensor([[b[2] for b in bars]]),
           "close": torch.tensor([[b[3] for b in bars]]),
           "volume": torch.tensor([[b[4] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3]
            for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net(raw, rt, sig, spec, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 15m-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY,
                      stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist()
def equity(flat):
    out, cs = [], 0.0
    for x in flat:
        cs += x
        out.append(cs)
    return out


def topk_windows(flat, k, recovery=True):
    """Top-k non-overlapping drawdown windows (peak->trough), deepest first.

    Scan equity peak-to-trough; record each (peak, trough) pair, then
    restart the scan after the trough so windows never overlap. Recovery
    is bars from trough to first re-touch of peak equity (None when the
    peak is never recovered within the series).
    """
    n = len(flat)
    eq = equity(flat)
    wins = []
    start = 0
    while start < n and len(wins) < k:
        pk = -1e18
        best = None  # (depth, peak_ix, trough_ix)
        pk_ix = start
        for i in range(start, n):
            if eq[i] > pk:
                pk = eq[i]
                pk_ix = i
            dd = pk - eq[i]
            if best is None or dd > best[0]:
                best = (dd, pk_ix, i)
        if best is None or best[0] <= 0:
            break
        depth, a, b = best
        rec = None
        if recovery:
            for j in range(b + 1, n):
                if eq[j] >= eq[a]:
                    rec = j - b
                    break
        wins.append({"peak": a, "trough": b, "depth": depth, "recovery": rec})
        start = b + 1
    return wins


def win_stats(net, legs, weights, closes, stamps, win):
    """Per-window anatomy: depth/length/recovery/cum/contrib + vol/corr."""
    a, b = win["peak"], win["trough"]
    depth = win["depth"]
    length = b - a
    # window PnL excludes the peak bar itself so cum == eq[trough]-eq[peak]
    # == -depth and per-coin contributions sum to cum (up to rounding).
    wsum = {c: sum(legs[c][t] * weights[c] for t in range(a + 1, b + 1)) for c in COINS}
    cum = sum(net[t] for t in range(a + 1, b + 1))
    an = {}
    for c in COINS:
        sub = closes[c][a:b + 1]
        rr = [(sub[i + 1] - sub[i]) / sub[i] for i in range(len(sub) - 1)]
        if len(rr) >= 2:
            m = sum(rr) / len(rr)
            v = sum((x - m) ** 2 for x in rr) / (len(rr) - 1)
            sd = math.sqrt(max(v, 0.0))
        else:
            sd = 0.0
        an[c] = round(sd * math.sqrt(BPY), 4)
    rets = {}
    for c in COINS:
        sub = closes[c][a:b + 1]
        rets[c] = [(sub[i + 1] - sub[i]) / sub[i] for i in range(len(sub) - 1)]
    mat = {}
    keys = list(COINS)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ri, rj = rets[keys[i]], rets[keys[j]]
            if len(ri) >= 2 and len(rj) >= 2:
                mi = sum(ri) / len(ri)
                mj = sum(rj) / len(rj)
                vi = sum((x - mi) ** 2 for x in ri) / max(len(ri) - 1, 1)
                vj = sum((x - mj) ** 2 for x in rj) / max(len(rj) - 1, 1)
                si, sj = math.sqrt(vi), math.sqrt(vj)
                if si > 0 and sj > 0:
                    cov = sum((x - mi) * (y - mj) for x, y in zip(ri, rj)) / (len(ri) - 1)
                    rho = max(-1.0, min(1.0, cov / (si * sj)))
                else:
                    rho = None
            else:
                rho = None
            mat["%s:%s" % (keys[i], keys[j])] = None if rho is None else round(rho, 4)
    off = [v for kk, v in mat.items() if kk.split(":")[0] != kk.split(":")[1] and v is not None]
    mean_corr = round(sum(off) / len(off), 4) if off else None
    return {"peak": a, "trough": b, "peak_ts": stamps[a], "trough_ts": stamps[b],
            "depth": round(depth, 4), "length_bars": length,
            "recovery_bars": win["recovery"],
            "cum": round(cum, 4),
            "contrib": {c: round(wsum[c], 4) for c in COINS},
            "vol_ann": an, "mean_corr": mean_corr, "corr": mat}


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_w6_dd start\n")
    k = SMOKE_K if SMOKE else K
    bars, closes, stamps = common15m(COINS)
    n = len(bars[COINS[0]])
    if SMOKE:
        n = min(n, SMOKE_N)
        bars = {c: bars[c][:n] for c in COINS}
        closes = {c: closes[c][:n] for c in COINS}
        stamps = {c: stamps[c][:n] for c in COINS}
    log("common 15m n=%d k=%d coins=%s smoke=%s" % (n, k, COINS, SMOKE))
    assert n > 2000, "15m grid too short: %d" % n
    if not SMOKE:
        assert n > 30000, "15m grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built (E10 FORMULA, locked)")

    res = {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
            "formula": list(FORMULA),
            "basket": {c: dict(BASE_SPECS[c]) for c in COINS},
            "weights": dict(W),
            "coins": list(COINS),
            "topk": k,
            "smoke_n": SMOKE_N if SMOKE else None,
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "bpy": BPY,
            "scale": SCALE,
            "grid": "15m",
            "grid_bars": n,
            "smoke": SMOKE,
            "note": "top-3 dd windows per coin (standalone) + portfolio; depth/length/recovery + per-coin contribution + window vol/corr. E10 FORMULA untouched; live chain untouched; offline read-only.",
        },
        "status": "partial",
        "percoin": {},
        "portfolio": {},
    }
    dump(res)

    legs, turns = {}, {}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs[c], turns[c] = leg_net(raw, rt, sg, BASE_SPECS[c], FEE, FUND)
        log("%s leg built" % c)

    for c in COINS:
        wser = [legs[c][t] * W[c] for t in range(n)]
        wins = topk_windows(wser, k)
        legs_eff = {cc: (legs[cc] if cc == c else [0.0] * n) for cc in COINS}
        rows = [win_stats(wser, legs_eff, W, closes, stamps[c], w) for w in wins]
        res["percoin"][c] = rows
        dump(res)
        log("%s top%d depths=%s" % (c, len(rows), [r["depth"] for r in rows]))

    net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    pwins = topk_windows(net, k)
    res["portfolio"] = {
        "net": [win_stats(net, legs, W, closes, stamps["ETC"], w) for w in pwins],
    }
    cs, pk, md = 0.0, -1e18, 0.0
    for x in net:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    m = sum(net) / n
    v = sum((x - m) ** 2 for x in net) / max(n - 1, 1)
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    res["portfolio"]["FULL"] = {"sharpe": round(sh, 3), "mdd": round(md, 4),
                                "cum": round(cs, 4), "n": n,
                                "turnover": round(sum(turn) / n, 6)}
    dump(res)
    log("portfolio top%d depths=%s FULL sh=%.3f mdd=%.4f" % (
        len(pwins), [r["depth"] for r in res["portfolio"]["net"]],
        res["portfolio"]["FULL"]["sharpe"], res["portfolio"]["FULL"]["mdd"]))

    res["verdict"] = "PENDING"
    res["decision"] = "PENDING"
    res["conclusion"] = ("PENDING (P0-3 FAIL): W6 15m drawdown anatomy "
                         "diagnostic only; top-3 windows + contribution + "
                         "vol/corr are shelf value, no adoption, no live "
                         "change, live untouched.")
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"portfolio_FULL": res["portfolio"]["FULL"],
                      "portfolio_depths": [r["depth"] for r in res["portfolio"]["net"]],
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
