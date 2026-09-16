"""H102 1h SMA-trend filter (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows,
cols timestamp,open,high,low,close,volume,quote_volume,trades).
Equal 0.2 weights.

SMA-trend filter semantics: causal SMA(N) of closes per coin on native
1h bars. SMA[t] = mean(close[t-N+1:t+1]); first N-1 bars -> neutral
(both legs closed). LONG leg allowed only where close[t] > SMA[t];
SHORT leg only where close[t] < SMA[t]. Gate applied to lp/sp right
after the quantile mask and BEFORE cooldown+stops (legs pre-cooldown),
so FULL == LONG + SHORT exactly. Sweep N in {48, 96, 192} vs
no-filter base.

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (fraction of bars with a leg open; long/short split) +
per-coin single-leg sensitivity (sharpe/turnover/trades/coverage per
coin per N) + vs_base deltas.

IMPORTANT: results/iter_H102_sma.json is dumped after EACH unit
(base + 3 N cells; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H102_sma.json (+ logs/iter_H102_sma.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H102_SMOKE=1 shrinks to N {96} on coins
{ETC,TRX} only. ITER_H102_OUT / ITER_H102_LOG override output paths (tests
use temp files so the committed FULL artifact is not clobbered).
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

from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.config import (
    FORMULA,
    LOCKED_APT,
    LOCKED_ATOM,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
NS = [48, 96, 192]
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H102_OUT", "results/iter_H102_sma.json"))
LOG = pathlib.Path(os.getenv("ITER_H102_LOG", "logs/iter_H102_sma.log"))

SMOKE = os.getenv("ITER_H102_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_NS = [96]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
    return bars, n


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


def sma_gates(bars_c, n_win):
    """Causal SMA(N) trend gates on close; first N-1 bars -> neutral (0,0).

    Returns (long_ok, short_ok) float lists: long_ok[t] = 1 iff
    close[t] > SMA_N[t]; short_ok[t] = 1 iff close[t] < SMA_N[t].
    """
    cl = [b[3] for b in bars_c]
    n = len(bars_c)
    lok = [0.0] * n
    sok = [0.0] * n
    if n_win <= 0 or n < n_win:
        return lok, sok
    run = sum(cl[:n_win])
    for t in range(n_win - 1, n):
        if t > n_win - 1:
            run += cl[t] - cl[t - n_win]
        sma = run / n_win
        if cl[t] > sma:
            lok[t] = 1.0
        elif cl[t] < sma:
            sok[t] = 1.0
    return lok, sok


def leg_pre(mat, spec, gate):
    """Locked-engine pre-roll net positions + returns (1h native).

    gate=None -> ungated base. Otherwise (lok, sok) float lists applied
    to lp/sp right after the quantile mask, BEFORE cooldown+stops
    (legs pre-cooldown).
    """
    raw, rt, sg = mat
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY, stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
    s = torch.sigmoid(sg)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (s > bt.long_th).float() * safe
    sp = (s < bt.short_th).float() * safe
    mask = qmask(sg, spec["q"])
    if mask is not None:
        lp = lp * mask
    if gate is not None:
        lok, sok = gate
        lp = lp * torch.tensor([lok])
        sp = sp * torch.tensor([sok])
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    return (lp - sp)[0].tolist(), rt[0].tolist()


def count_trades(pos):
    n = 0
    prev = 0.0
    for v in pos:
        cur = 1.0 if abs(v) > 0.5 else 0.0
        if cur > 0.5 and prev < 0.5:
            n += 1
        prev = cur
    return n


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0
    pk0 = -1e18
    md = 0.0
    for x in s:
        cs += x
        pk0 = max(pk0, cs)
        md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def eval_unit(mats, gates, n, n_win, coins, weights):
    """SMA leg gate (n_win=None -> no filter) + roll1 + accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    covs = []
    for c in coins:
        if n_win is None:
            pre, rt = leg_pre(mats[c], SPECS[c], None)
            cov_l = cov_s = cov = 1.0
        else:
            gate = gates[c]
            pre, rt = leg_pre(mats[c], SPECS[c], gate)
            lok, sok = gate
            cov_l = sum(lok) / n if n else 0.0
            cov_s = sum(sok) / n if n else 0.0
            cov = sum(1 for a, b in zip(lok, sok) if a > 0.5 or b > 0.5) / n if n else 0.0
        rolled = [0.0] + pre[:-1]
        dp = [abs(rolled[t] - (rolled[t - 1] if t else 0.0)) for t in range(n)]
        net_c = [rolled[t] * rt[t] * LEV - dp[t] * FEE * LEV
                 - rolled[t] * FUND * LEV for t in range(n)]
        legs[c] = (rolled, net_c, rt)
        turns[c] = dp
        tr = count_trades(rolled)
        tot_trades += tr
        st = seg(net_c, dp, 0, n)
        st["trades"] = tr
        st["coverage"] = round(cov, 6)
        st["coverage_long"] = round(cov_l, 6)
        st["coverage_short"] = round(cov_s, 6)
        per_coin[c] = st
        covs.append(cov)
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    full["coverage_all"] = round(sum(covs) / len(covs), 6) if covs else 0.0
    return full, per_coin


def dump(rows, units_done, n, coins, weights, ns, complete, base_full=None):
    if base_full is not None:
        for r in rows:
            if r["unit"] != "base" and "vs_base" not in r:
                f = r["FULL"]
                r["vs_base"] = {
                    "d_sharpe": round(f["sharpe"] - base_full["sharpe"], 3),
                    "d_cum": round(f["cum"] - base_full["cum"], 4),
                    "d_turnover": round(f["turnover"] - base_full["turnover"], 6),
                    "d_trades": f["trades"] - base_full["trades"],
                }
    out = {
        "config": {
            "engine": "mirror run_weight_modes.py leg_net + quantile q0.3 "
                      "long-only + cooldown + stops + vol_scale(vt None->1.0) "
                      "+ roll1; static equal 0.2 Top5; 1h native cd/ts/vw x4",
            "formula": list(FORMULA),
            "basket": {c: {"lth": SPECS[c]["lth"], "sth": SPECS[c]["sth"],
                           "cd": SPECS[c]["cd"], "sl": SPECS[c]["sl"],
                           "ts": SPECS[c]["ts"], "q": SPECS[c]["q"]}
                       for c in COINS},
            "weights": dict(weights),
            "filter": "SMA(N) trend: LONG leg only where close>SMA_N "
                      "(causal, full-window; first N-1 bars neutral), SHORT "
                      "leg only where close<SMA_N; gates on legs pre-cooldown "
                      "(post quantile mask)",
            "sma_Ns": list(ns),
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "smoke": SMOKE,
            "grid": "1h",
            "grid_bars": n,
            "bpy": BPY,
            "scale": SCALE,
        },
        "rows": rows,
        "units_done": list(units_done),
        "complete": complete,
        "verdict": "PENDING",
        "decision": "NO_ADOPTION",
        "decision_note": "待定(P0-3 FAIL): SMA-trend过滤只做诊断,不采用,不改live, "
                         "live basket与阈值保持locked原状。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; SMA-trend "
                       "filter diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H102_sma start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    ns = list(SMOKE_NS) if SMOKE else list(NS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s Ns=%s scale=x%d"
        % (n, coins, ns, SCALE))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    gates_cache = {}
    for nw in ns:
        gates_cache[nw] = {c: sma_gates(bars[c], nw) for c in coins}
        for c in coins:
            lok, sok = gates_cache[nw][c]
            cov = sum(1 for a, b in zip(lok, sok) if a > 0.5 or b > 0.5) / n
            log("%s SMA(%d) coverage=%.4f long=%.4f short=%.4f"
                % (c, nw, cov, sum(lok) / n, sum(sok) / n))

    rows = []
    units_done = []

    def run_unit(label, n_win, base_full=None):
        gates = None if n_win is None else gates_cache[n_win]
        full, per_coin = eval_unit(mats, gates, n, n_win, coins, weights)
        row = {"unit": label, "N": n_win, "FULL": full,
               "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        bf = base_full if base_full is not None else (full if n_win is None else None)
        dump(rows, units_done, n, coins, weights, ns, complete=False,
             base_full=bf if n_win is not None else None)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d cov=%.4f"
            % (label, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["coverage_all"]))

    run_unit("base", None)
    base_full = rows[0]["FULL"]
    for nw in ns:
        run_unit("sma_%d" % nw, nw, base_full=base_full)
    dump(rows, units_done, n, coins, weights, ns, complete=True,
         base_full=base_full)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
