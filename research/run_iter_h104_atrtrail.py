"""H104 1h ATR-trailing filter (Top5) -- DIAGNOSTIC ONLY.

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

ATR-trailing semantics: per-coin Wilder ATR(48) on native 1h
highs/lows/closes (seed first 48 bars -> stop inactive). EXIT ONLY:
a ratcheted trailing stop trails each open position --
long trail = max(high[t] - mult*ATR[t]) ratcheted up while long,
short trail = min(low[t] + mult*ATR[t]) ratcheted down while short;
a close beyond the trail forces flat (entries/flips pass through,
trail resets on flat). Gate is applied to pre-roll positions (after
cooldown+stops+vol_scale), before roll1. Sweep mult in {1.5, 2.0, 3.0}
vs no-filter base (v2 grid).

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (fraction of bars with no forced exit) + breaches + per-coin
single-leg sensitivity (sharpe/turnover/trades/coverage/breaches
per coin per mult).

IMPORTANT: results/iter_H104_atrtrail.json is dumped after EACH unit
(base + 3 mult cells; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H104_atrtrail.json (+ logs/iter_H104_atrtrail.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H104_SMOKE=1 shrinks to mult {2.0} on coins
{ETC,TRX} only. ITER_H104_OUT / ITER_H104_LOG override output paths (tests
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
ATR_WINDOW = 48
MULTS = [1.5, 2.0, 3.0]
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H104_OUT", "results/iter_H104_atrtrail.json"))
LOG = pathlib.Path(os.getenv("ITER_H104_LOG", "logs/iter_H104_atrtrail.log"))

SMOKE = os.getenv("ITER_H104_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_MULTS = [2.0]


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
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
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


def atr_wilder(bars_c, window):
    """Wilder ATR(window) on 1h high/low/close; first `window` bars -> None."""
    hi = [b[1] for b in bars_c]
    lo = [b[2] for b in bars_c]
    cl = [b[3] for b in bars_c]
    n = len(bars_c)
    tr = [0.0] * n
    for t in range(n):
        if t == 0:
            tr[t] = hi[t] - lo[t]
        else:
            tr[t] = max(hi[t] - lo[t], abs(hi[t] - cl[t - 1]),
                        abs(lo[t] - cl[t - 1]))
    atr = [None] * n
    if n > window:
        seed = sum(tr[:window]) / window
        atr[window - 1] = seed
        for t in range(window, n):
            atr[t] = (atr[t - 1] * (window - 1) + tr[t]) / window
    return atr


def leg_base(bars_c, spec):
    """Locked-engine pre-roll positions + returns (1h native)."""
    raw, rt, sg = build_sig(bars_c)
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
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    return (lp - sp)[0].tolist(), rt[0].tolist()


def apply_atr_trail(pre, bars_c, atr, mult):
    """EXIT-ONLY ATR trailing stop on pre-roll positions.

    Long trail ratchets up: max(high[t] - mult*ATR[t]); a close below the
    trail forces flat. Short side mirrored with low[t] + mult*ATR[t].
    Entries/flips pass through; trail resets on flat. ATR None (warmup)
    -> stop inactive (pass). Returns (gated, ok, breaches) where ok[t] is
    False only on a forced-exit bar.
    """
    hi = [b[1] for b in bars_c]
    lo = [b[2] for b in bars_c]
    cl = [b[3] for b in bars_c]
    n = len(pre)
    g = [0.0] * n
    ok = [True] * n
    breaches = 0
    trail = None
    prev = 0.0
    for t in range(n):
        cur = pre[t]
        a = atr[t]
        if cur == 0.0:
            g[t] = 0.0
            trail = None
        elif a is None:
            g[t] = cur
            trail = None
        elif cur > 0:
            lvl = hi[t] - mult * a
            trail = lvl if (prev <= 0.0 or trail is None) else max(trail, lvl)
            if cl[t] < trail:
                g[t] = 0.0
                trail = None
                ok[t] = False
                breaches += 1
            else:
                g[t] = cur
        else:
            lvl = lo[t] + mult * a
            trail = lvl if (prev >= 0.0 or trail is None) else min(trail, lvl)
            if cl[t] > trail:
                g[t] = 0.0
                trail = None
                ok[t] = False
                breaches += 1
            else:
                g[t] = cur
        prev = g[t]
    return g, ok, breaches


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


def eval_unit(base, oks, n, mult, coins, weights, bars, atrs):
    """Apply ATR-trailing exit gate (mult=None -> no filter) + roll1 + accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    for c in coins:
        pre, rt = base[c]
        if mult is None:
            p = list(pre)
            cov = 1.0
            br = 0
            ok = [True] * n
        else:
            p, ok, br = apply_atr_trail(pre, bars[c], atrs[c], mult)
            cov = sum(1 for v in ok if v) / n if n else 0.0
        rolled = [0.0] + p[:-1]
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
        st["breaches"] = br
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    if mult is None:
        cov_all = 1.0
        tot_br = 0
    else:
        cov_all = sum(1 for t in range(n)
                      if all(oks[c][t] for c in coins)) / n if n else 0.0
        tot_br = sum(per_coin[c]["breaches"] for c in coins)
    full["coverage_all"] = round(cov_all, 6)
    full["breaches"] = tot_br
    return full, per_coin


def dump(rows, units_done, n, coins, weights, mults, complete):
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
            "filter": "ATR-trailing exit-only: Wilder ATR(48) on 1h "
                      "high/low/close (seed first 48 bars -> inactive); "
                      "long trail ratchets max(high-mult*ATR), exit on "
                      "close < trail; short mirrored; entries/flips pass, "
                      "trail resets on flat",
            "atr_window": ATR_WINDOW,
            "atr_mults": list(mults),
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
        "decision_note": "待定(P0-3 FAIL): ATR-trailing退出只做诊断,不采用,不改live, "
                         "live basket与阈值保持locked原状。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; ATR-trailing "
                       "exit filter diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H104_atrtrail start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    mults = list(SMOKE_MULTS) if SMOKE else list(MULTS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s mults=%s scale=x%d atr_win=%d"
        % (n, coins, mults, SCALE, ATR_WINDOW))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    atrs = {c: atr_wilder(bars[c], ATR_WINDOW) for c in coins}
    for c in coins:
        act = sum(1 for v in atrs[c] if v is not None) / n
        log("%s ATR(%d) active=%.4f" % (c, ATR_WINDOW, act))

    rows = []
    units_done = []

    def run_unit(label, mult):
        oks = None
        if mult is not None:
            oks = {}
            for c in coins:
                _, ok, br = apply_atr_trail(base[c][0], bars[c],
                                            atrs[c], mult)
                oks[c] = ok
                cov = sum(1 for v in ok if v) / n
                log("%s ATRtrail(x%s) coverage=%.4f breaches=%d"
                    % (c, mult, cov, br))
        full, per_coin = eval_unit(base, oks, n, mult, coins, weights,
                                   bars, atrs)
        row = {"unit": label, "mult": mult, "FULL": full,
               "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        dump(rows, units_done, n, coins, weights, mults, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d cov=%.4f br=%d"
            % (label, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["coverage_all"],
               full["breaches"]))

    run_unit("base", None)
    for m in mults:
        run_unit("atrtrail_x%s" % m, m)
    dump(rows, units_done, n, coins, weights, mults, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
