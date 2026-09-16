"""W5 15m turnover audit (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT
be used as demo-listing evidence. No adoption, no live change, live chain
untouched. Offline read-only: reads data/data_1y/15m/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005): quantile q0.3 long-only mask +
cooldown + stops + vol_scale(vt None->1.0) + roll1. 15m NATIVE grid
(data/data_1y/15m, ~35040 bars): cd/ts/vw x16, BPY=35040. Equal 0.2
weights. Top5 locked specs ETC(0.88/0.12/cd18/None/ts24)
TRX(0.85/0.12/cd6/0.05/ts24) ATOM(0.85/0.15/cd6/0.05/ts24)
APT(0.88/0.12/cd18/None/ts24) KAS(0.88/0.12/cd6/None/ts24), q0.3.

Units:
  A. per-coin base audit: entries / flips / exits + hold-bars distribution
     (nonzero-run lengths: count/mean/median/p90/max + buckets) + FULL/H2
     sharpe/mdd/turnover on the executed (post-cooldown/stops/vol/roll1)
     signed position.
  B. turnover-vs-trades scatter: per-coin {turnover, entries, flips,
     trades=entries+flips} points at base config.
  C. min-hold effect: uniform cd override {0,2,4} (4h-bar units, engine
     x16 internally) over all coins; per cell FULL sharpe/mdd/cum/
     turnover/trades + H2 sharpe + deltas vs base.

Incremental dump: results JSON is rewritten after each unit (partial=true
until the final write), so a killed run still leaves a partial artifact.

Smoke (for tests): ITER_W5_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars,
min-hold {0,2}. OUT/LOG overridable via ITER_W5_OUT / ITER_W5_LOG.
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
MINHOLD_GRID = [0, 2, 4]
BPY = 35040.0
SCALE = 16  # 4h-bar params -> 15m bars

OUT = pathlib.Path(os.getenv("ITER_W5_OUT", "results/iter_W5_turn.json"))
LOG = pathlib.Path(os.getenv("ITER_W5_LOG", "logs/iter_w5_turn.log"))

SMOKE = os.getenv("ITER_W5_SMOKE") == "1"
SMOKE_MINHOLD = [0, 2]
SMOKE_COINS = ["ETC", "TRX"]

THRESH = 0.5
HOLD_BUCKETS = ["1", "2_4", "5_8", "9_16", "17_32", "33_64", "65p"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


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
    bars, closes = {}, {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [b[3] for b in bars[c]]
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


def leg_net(raw, rt, sig, lth, sth, cd, sl, ts, vt, vw, q, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 15m-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=lth, short_th=sth,
                      cooldown_bars=cd * SCALE, bars_per_year=BPY,
                      stop_loss=sl, time_stop=ts * SCALE,
                      vol_target=vt, vol_window=vw * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, q)
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
    pos = lp - sp
    gross = pos * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = pos * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), pos[0].tolist()


def sgn(v, thresh=THRESH):
    if v > thresh:
        return 1
    if v < -thresh:
        return -1
    return 0


def hold_runs(pos, thresh=THRESH):
    """Run lengths split on flat bars AND on direct long<->short flips."""
    runs = []
    cur_len = 0
    cur_sign = 0
    for v in pos:
        s = sgn(v, thresh)
        if s == 0:
            if cur_len > 0:
                runs.append(cur_len)
                cur_len, cur_sign = 0, 0
        elif s != cur_sign:
            if cur_len > 0:
                runs.append(cur_len)
            cur_len, cur_sign = 1, s
        else:
            cur_len += 1
    if cur_len > 0:
        runs.append(cur_len)
    return runs


def bucketize(runs):
    b = {k: 0 for k in HOLD_BUCKETS}
    for L in runs:
        if L <= 1:
            b["1"] += 1
        elif L <= 4:
            b["2_4"] += 1
        elif L <= 8:
            b["5_8"] += 1
        elif L <= 16:
            b["9_16"] += 1
        elif L <= 32:
            b["17_32"] += 1
        elif L <= 64:
            b["33_64"] += 1
        else:
            b["65p"] += 1
    return b


def trade_stats(pos, thresh=THRESH):
    """entries (flat->nonzero), flips (nonzero sign change), exits, holds."""
    entries = flips = exits = 0
    prev = 0
    for v in pos:
        s = sgn(v, thresh)
        if s != 0 and prev == 0:
            entries += 1
        if s != 0 and prev != 0 and s != prev:
            flips += 1
        if s == 0 and prev != 0:
            exits += 1
        prev = s
    runs = hold_runs(pos, thresh)
    if runs:
        srt = sorted(runs)
        mean = sum(runs) / len(runs)
        median = float(srt[len(srt) // 2])
        p90 = float(srt[min(len(srt) - 1, int(len(srt) * 0.9))])
        mx, mn = max(runs), min(runs)
    else:
        mean = median = p90 = 0.0
        mx = mn = 0
    return {
        "entries": entries,
        "flips": flips,
        "exits": exits,
        "trades": entries + flips,
        "holds": {
            "count": len(runs),
            "mean": round(mean, 2),
            "median": median,
            "p90": p90,
            "max": mx,
            "min": mn,
            "buckets": bucketize(runs),
        },
    }


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs, pk, md = 0.0, -1e18, 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def portfolio(legs, turns, coins, n):
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    return net, turn


def dump(state):
    OUT.write_text(json.dumps(state, indent=1))


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_w5_turn start\n")
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    mh_grid = list(SMOKE_MINHOLD) if SMOKE else list(MINHOLD_GRID)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, _closes = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 15m n=%d h2a=%d coins=%s minhold=%s smoke=%s" % (n, h2a, coins, mh_grid, SMOKE))
    assert n > 2000, "15m grid too short: %d" % n
    if not SMOKE:
        assert n > 30000, "15m grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")

    base_cfg = {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "weights": dict(w),
        "coins": list(coins),
        "minhold_grid": list(mh_grid),
        "minhold_unit": "4h-bar, engine x16 internally",
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "bpy": BPY,
        "scale": SCALE,
        "grid": "15m",
        "grid_bars": n,
        "h2_start": h2a,
        "smoke": SMOKE,
        "note": "turnover audit only: entries/flips/hold-bars + turnover-vs-trades scatter + min-hold effect. E10 FORMULA untouched; live chain untouched; offline read-only.",
    }

    def run_legs(cd_map):
        legs, turns, poss = {}, {}, {}
        for c in coins:
            spec = BASE_SPECS[c]
            raw, rt, sg = mats[c]
            legs[c], turns[c], poss[c] = leg_net(
                raw, rt, sg, spec["lth"], spec["sth"], cd_map[c],
                spec["sl"], spec["ts"], spec["vt"], spec["vw"],
                spec["q"], FEE, FUND)
        return legs, turns, poss

    state = {"config": base_cfg, "partial": True,
             "verdict": "PENDING", "decision": "PENDING"}

    # ---- Unit A: per-coin base audit ----
    base_cd = {c: BASE_SPECS[c]["cd"] for c in coins}
    legs, turns, poss = run_legs(base_cd)
    net, turn = portfolio(legs, turns, coins, n)
    base_FULL = seg(net, turn, 0, n)
    base_FULL["trades"] = sum(trade_stats(poss[c][0:n])["trades"] for c in coins)
    base_H2 = seg(net, turn, h2a, n)
    base_per_coin = {}
    for c in coins:
        st = trade_stats(poss[c][0:n])
        fh = seg(legs[c], turns[c], 0, n)
        hh = seg(legs[c], turns[c], h2a, n)
        base_per_coin[c] = {"FULL": fh, "H2": hh, **st}
        log("A %s entries=%d flips=%d exits=%d holds=%d mean=%.1f p90=%.0f max=%d sh=%.3f to=%.6f" % (
            c, st["entries"], st["flips"], st["exits"], st["holds"]["count"],
            st["holds"]["mean"], st["holds"]["p90"], st["holds"]["max"],
            fh["sharpe"], fh["turnover"]))
    state["base_FULL"] = base_FULL
    state["base_H2"] = base_H2
    state["base_H2_sharpe"] = base_H2["sharpe"]
    state["base_per_coin"] = base_per_coin
    dump(state)
    log("unit A dumped (partial)")

    # ---- Unit B: turnover-vs-trades scatter (base config) ----
    scatter = [{"coin": c,
                "turnover": base_per_coin[c]["FULL"]["turnover"],
                "entries": base_per_coin[c]["entries"],
                "flips": base_per_coin[c]["flips"],
                "trades": base_per_coin[c]["trades"]} for c in coins]
    state["scatter"] = scatter
    dump(state)
    log("unit B dumped (partial): %s" % (
        " ".join("%s(to=%.6f,tr=%d)" % (p["coin"], p["turnover"], p["trades"]) for p in scatter)))

    # ---- Unit C: min-hold effect (uniform cd override) ----
    minhold_rows = []
    for mh in mh_grid:
        cd_map = {c: mh for c in coins}
        l2, t2, p2 = run_legs(cd_map)
        nn, tt = portfolio(l2, t2, coins, n)
        full = seg(nn, tt, 0, n)
        full["trades"] = sum(trade_stats(p2[c][0:n])["trades"] for c in coins)
        h2 = seg(nn, tt, h2a, n)
        row = {"min_hold": mh, "FULL": full, "H2_sharpe": h2["sharpe"],
               "H2_trades": sum(trade_stats(p2[c][h2a:n])["trades"] for c in coins),
               "d_sharpe_vs_base": round(full["sharpe"] - base_FULL["sharpe"], 3),
               "d_turnover_vs_base": round(full["turnover"] - base_FULL["turnover"], 6),
               "turnover_cut_vs_base": round(
                   (base_FULL["turnover"] - full["turnover"]) / base_FULL["turnover"], 4)
               if base_FULL["turnover"] > 0 else 0.0}
        minhold_rows.append(row)
        state["minhold_rows"] = minhold_rows
        dump(state)
        log("C mh=%d sh=%.3f mdd=%.4f cum=%.4f to=%.6f tr=%d H2=%.3f dsh=%+.3f cut=%.1f%% (partial)" % (
            mh, full["sharpe"], full["mdd"], full["cum"], full["turnover"],
            full["trades"], h2["sharpe"], row["d_sharpe_vs_base"],
            row["turnover_cut_vs_base"] * 100))

    mh0 = next(r for r in minhold_rows if r["min_hold"] == 0)
    note = ("min-hold 0 vs base: turnover x%.2f, sharpe %+.3f; "
            "diagnostic only, no adoption." % (
                (mh0["FULL"]["turnover"] / base_FULL["turnover"]
                 if base_FULL["turnover"] > 0 else 0.0),
                mh0["d_sharpe_vs_base"]))
    log(note)
    state["minhold_note"] = note
    state["partial"] = False
    state["conclusion"] = ("PENDING (P0-3 FAIL): W5 15m turnover audit diagnostic "
                           "only; entries/flips/hold-bars + scatter + min-hold "
                           "effects are shelf value, no adoption, no live change, "
                           "live untouched.")
    dump(state)
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_FULL, "base_H2_sharpe": base_H2["sharpe"],
                      "scatter": scatter, "minhold_rows": minhold_rows,
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
