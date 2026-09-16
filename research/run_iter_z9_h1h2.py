"""Z9 15m H1/H2 stability (Top5). Split-half per-coin sharpe + rank corr + FULL.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA via
StackVM+FeatureEngineer; MemeBacktest venue=aster lev2 short_enabled
fund0.0005; quantile q0.3 long-only + cooldown + stops + vol_scale
(vt None->1.0) + roll1). 15m native: reads data/data_1y/15m directly
(no x16 aggregation); cd/ts/vw x16 (4h-bar units -> 15m-bar units);
BPY=35040. Equal 0.2 weights. Offline read-only; verdict PENDING
(P0-3 FAIL), no adoption, live untouched.
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
    FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX,
    LEV, FUND, FEE,
)

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(LOCKED_ETC),
    "TRX": dict(LOCKED_TRX),
    "ATOM": dict(LOCKED_ATOM),
    "APT": dict(LOCKED_APT),
    "KAS": dict(LOCKED_KAS),
}
BPY = 35040.0
SCALE = 16
WEIGHT = 0.2
OUT = pathlib.Path("results/iter_Z9_h1h2.json")
LOG = pathlib.Path("logs/iter_Z9_h1h2.log")


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def scaled_spec(base):
    s = dict(base)
    for k in ("cd", "ts", "vw"):
        if s.get(k) is not None:
            s[k] = int(s[k]) * SCALE
    return s


SPECS = {c: scaled_spec(b) for c, b in BASE_SPECS.items()}


def load15(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [
        (int(r["timestamp"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["volume"]))
        for r in rows
    ]


def common_grid(coins):
    raw = {c: load15(c) for c in coins}
    ts0 = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts0 &= set(r[0] for r in raw[c])
    ts_sorted = sorted(ts0)
    bars = {}
    for c in coins:
        m = {r[0]: r for r in raw[c]}
        bars[c] = [(m[t][1], m[t][2], m[t][3], m[t][4], m[t][5]) for t in ts_sorted]
    return ts_sorted, bars


def build_sig(bars):
    n = len(bars)
    raw = {
        "open": torch.tensor([[b[0] for b in bars]]),
        "high": torch.tensor([[b[1] for b in bars]]),
        "low": torch.tensor([[b[2] for b in bars]]),
        "close": torch.tensor([[b[3] for b in bars]]),
        "volume": torch.tensor([[b[4] for b in bars]]),
        "liquidity": torch.full((1, n), 1e7),
        "fdv": torch.full((1, n), 1e8),
    }
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net(raw, rt, sig, spec, fee, fund):
    bt = MemeBacktest(
        venue="aster", leverage=LEV, short_enabled=True,
        funding_override=fund, fee_override=fee,
        long_th=spec["lth"], short_th=spec["sth"],
        cooldown_bars=spec["cd"], bars_per_year=BPY,
        stop_loss=spec["sl"], time_stop=spec["ts"],
        vol_target=spec["vt"], vol_window=spec["vw"],
    )
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
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), (lp - sp)[0].tolist()


def sharpe(s):
    n = len(s)
    if n < 2:
        return 0.0
    m = sum(s) / n
    v = sum((x - m) ** 2 for x in s) / (n - 1)
    return m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    return {
        "sharpe": round(sh, 3),
        "ann": round(sum(s) / n * BPY, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(sum(s), 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }


def rank_desc(keys_vals):
    order = sorted(keys_vals, key=lambda kv: (-kv[1], kv[0]))
    return {k: i + 1 for i, (k, _) in enumerate(order)}


def spearman(rank_a, rank_b, keys):
    n = len(keys)
    if n < 2:
        return 0.0
    xa = [rank_a[k] for k in keys]
    xb = [rank_b[k] for k in keys]
    ma = sum(xa) / n
    mb = sum(xb) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(xa, xb))
    va = sum((a - ma) ** 2 for a in xa)
    vb = sum((b - mb) ** 2 for b in xb)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z9_h1h2 start\n")
    ts, bars = common_grid(COINS)
    n = len(ts)
    h = n // 2
    log("common 15m n=%d h1=[0,%d) h2=[%d,%d)" % (n, h, h, n))
    legs = {}
    turns = {}
    for c in COINS:
        raw, rt, sig = build_sig(bars[c])
        legs[c], turns[c], _ = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
    per_coin = {}
    for c in COINS:
        per_coin[c] = {
            "FULL": seg(legs[c], turns[c], 0, n),
            "H1": seg(legs[c], turns[c], 0, h),
            "H2": seg(legs[c], turns[c], h, n),
        }
    h1_sh = {c: per_coin[c]["H1"]["sharpe"] for c in COINS}
    h2_sh = {c: per_coin[c]["H2"]["sharpe"] for c in COINS}
    full_sh = {c: per_coin[c]["FULL"]["sharpe"] for c in COINS}
    rank_h1 = rank_desc(list(h1_sh.items()))
    rank_h2 = rank_desc(list(h2_sh.items()))
    rank_full = rank_desc(list(full_sh.items()))
    rho = spearman(rank_h1, rank_h2, COINS)
    net = [sum(legs[c][t] * WEIGHT for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * WEIGHT for c in COINS) for t in range(n)]
    basket = {
        "FULL": seg(net, turn, 0, n),
        "H1": seg(net, turn, 0, h),
        "H2": seg(net, turn, h, n),
    }
    n_pos_h1 = sum(1 for c in COINS if h1_sh[c] > 0)
    n_pos_h2 = sum(1 for c in COINS if h2_sh[c] > 0)
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => Z9 verdict PENDING; "
            "split-half stability read-only, no adoption, live untouched.")
    res = {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + roll1",
            "formula": list(FORMULA),
            "basket": {c: dict(SPECS[c]) for c in COINS},
            "base_specs_4h": {c: dict(BASE_SPECS[c]) for c in COINS},
            "scale_4h_to_15m": SCALE,
            "weights": {c: WEIGHT for c in COINS},
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "bpy": BPY,
            "data_dir": "data/data_1y/15m",
            "grid_bars": n,
            "h1": [0, h],
            "h2": [h, n],
            "note": "E10 FORMULA untouched; offline read-only",
        },
        "per_coin": per_coin,
        "ranks": {"H1": rank_h1, "H2": rank_h2, "FULL": rank_full},
        "rank_corr_H1_vs_H2": round(rho, 4),
        "stability": {
            "n_pos_H1": n_pos_h1,
            "n_pos_H2": n_pos_h2,
            "sign_match": round(sum(1 for c in COINS if (h1_sh[c] > 0) == (h2_sh[c] > 0)) / len(COINS), 4),
        },
        "basket": basket,
        "verdict": verdict,
        "decision": decision,
        "decision_note": note,
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("per-coin FULL=%s" % ({c: per_coin[c]["FULL"]["sharpe"] for c in COINS},))
    log("per-coin H1=%s H2=%s rho=%.4f" % (h1_sh, h2_sh, rho))
    log("basket FULL sh=%.3f H1 sh=%.3f H2 sh=%.3f" % (
        basket["FULL"]["sharpe"], basket["H1"]["sharpe"], basket["H2"]["sharpe"]))
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
