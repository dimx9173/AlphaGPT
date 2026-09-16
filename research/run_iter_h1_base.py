"""H1 1h baseline anchor (Top5) -- equal 0.2 FULL + fee2x + H2 + per-coin legs.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native: reads
data/data_1y/1h directly (no aggregation); cd/ts/vw x4 (4h-bar units
-> 1h-bar units); BPY=8760. Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Output: results/iter_H1_base.json with equal-0.2 FULL
sharpe/mdd/final_x/turnover/trades + fee2x FULL + H2 (last 730 bars) +
per-coin legs. This anchors all H-rounds.

trades = opened directional trades = entries (flat->non-zero) + flips
(long<->short direct switch); exits reported separately. Fee does not
change positions, so fee2x trades/turnover equal base by construction.

Offline read-only; verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Incremental dump: results JSON rewritten after each unit (partial survives).
Smoke (for tests): ITER_H1_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
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
    LEV, FUND, FEE, FEE2X,
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
BPY = 8760.0
SCALE = 4
WEIGHT = 0.2
H2_LEN = 730
OUT = pathlib.Path(os.getenv("ITER_H1_OUT", "results/iter_H1_base.json"))
LOG = pathlib.Path(os.getenv("ITER_H1_LOG", "logs/iter_H1_base.log"))
SMOKE = os.getenv("ITER_H1_SMOKE") == "1"


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


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [
        (int(r["timestamp"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["volume"]))
        for r in rows
    ]


def common_grid(coins):
    raw = {c: load1h(c) for c in coins}
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


def count_trades(pos, a, b):
    """Opened-directional-trade counts over pos[a:b].

    entries: flat->non-zero; flips: long<->short direct switch;
    exits: non-zero->flat; trades = entries + flips.
    Slice-aware: previous position is pos[a-1] (or 0.0 at grid start).
    """
    entries = flips = exits = 0
    prev = pos[a - 1] if a > 0 else 0.0
    for t in range(a, b):
        cur = pos[t]
        if cur != 0.0 and prev == 0.0:
            entries += 1
        elif cur == 0.0 and prev != 0.0:
            exits += 1
        elif cur != 0.0 and prev != 0.0 and (cur > 0) != (prev > 0):
            flips += 1
        prev = cur
    return {"trades": entries + flips, "entries": entries,
            "flips": flips, "exits": exits}


def seg(net, turn, pos, a, b):
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
    cum = sum(s)
    out = {
        "sharpe": round(sh, 3),
        "ann": round(cum / n * BPY, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(cum, 4),
        "final_x": round(1.0 + cum, 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }
    out.update(count_trades(pos, a, b))
    return out


def dump(state):
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def base_config(coins, n, h2a):
    return {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins},
        "scale_4h_to_1h": SCALE,
        "weights": {c: WEIGHT for c in coins},
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "fee2x": FEE2X,
        "grid": "1h",
        "grid_bars": n,
        "h2_len": H2_LEN,
        "h2_start": h2a,
        "bpy": BPY,
        "data_dir": "data/data_1y/1h",
        "coins": list(coins),
        "smoke": SMOKE,
        "trades_def": "trades = entries (flat->non-zero) + flips (long<->short); exits reported separately; fee does not change positions so fee2x trades == base",
        "note": "H1 1h baseline anchor for all H-rounds; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H1_base start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    h2a = n - H2_LEN
    log("common 1h n=%d h2a=%d coins=%s smoke=%s scale=x%d" % (n, h2a, coins, SMOKE, SCALE))
    assert n > 2000, n
    assert 0 < H2_LEN < n, (H2_LEN, n)
    cfg = base_config(coins, n, h2a)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    legs, turns, poss = {}, {}, {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs[c], turns[c], poss[c] = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg,
              "legs_done": sorted(legs)})
        log("leg %s done" % c)
    per_coin = {c: {"FULL": seg(legs[c], turns[c], poss[c], 0, n),
                    "H2": seg(legs[c], turns[c], poss[c], h2a, n)} for c in coins}
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    pos = [sum(poss[c][t] * w for c in coins) for t in range(n)]
    basket = {"FULL": seg(net, turn, pos, 0, n),
              "H2": seg(net, turn, pos, h2a, n)}
    # basket trades = sum of per-coin leg trades (parallel legs, not netted)
    for k in ("FULL", "H2"):
        for f in ("trades", "entries", "flips", "exits"):
            basket[k][f] = sum(per_coin[c][k][f] for c in coins)
    f, h = basket["FULL"], basket["H2"]
    log("basket FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f to=%.6f trades=%d | H2 sh=%.3f mdd=%.4f trades=%d"
        % (f["sharpe"], f["mdd"], f["cum"], f["final_x"], f["turnover"], f["trades"], h["sharpe"], h["mdd"], h["trades"]))
    dump({"status": "partial", "stage": "basket", "config": cfg,
          "per_coin": per_coin, "basket": basket})
    legs2, turns2, poss2 = {}, {}, {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs2[c], turns2[c], poss2[c] = leg_net(raw, rt, sig, SPECS[c], FEE2X, FUND)
        dump({"status": "partial", "stage": "fee2x_leg_%s" % c, "config": cfg,
              "per_coin": per_coin, "basket": basket,
              "fee2x_legs_done": sorted(legs2)})
    per_coin2 = {c: {"FULL": seg(legs2[c], turns2[c], poss2[c], 0, n),
                     "H2": seg(legs2[c], turns2[c], poss2[c], h2a, n)} for c in coins}
    net2 = [sum(legs2[c][t] * w for c in coins) for t in range(n)]
    turn2 = [sum(turns2[c][t] * w for c in coins) for t in range(n)]
    pos2 = [sum(poss2[c][t] * w for c in coins) for t in range(n)]
    basket2 = {"FULL": seg(net2, turn2, pos2, 0, n),
               "H2": seg(net2, turn2, pos2, h2a, n)}
    for k in ("FULL", "H2"):
        for fld in ("trades", "entries", "flips", "exits"):
            basket2[k][fld] = sum(per_coin2[c][k][fld] for c in coins)
    fee2x = {"basket": {
        "FULL": basket2["FULL"], "H2": basket2["H2"],
        "gap_FULL_sharpe": round(basket2["FULL"]["sharpe"] - basket["FULL"]["sharpe"], 3)},
        "per_coin": per_coin2}
    log("fee2x FULL sh=%.3f gap=%+.3f final_x=%.4f | H2 sh=%.3f"
        % (basket2["FULL"]["sharpe"], fee2x["basket"]["gap_FULL_sharpe"],
           basket2["FULL"]["final_x"], basket2["H2"]["sharpe"]))
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H1 verdict PENDING; "
            "1h baseline anchor only, no adoption, live untouched.")
    res = {"status": "final", "config": cfg, "per_coin": per_coin,
           "basket": basket, "fee2x": fee2x,
           "verdict": verdict, "decision": decision, "decision_note": note,
           "conclusion": ("H1 1h Top5 equal-0.2 anchor FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f trades=%d | "
                          "fee2x FULL sh=%.3f (gap %+.3f) | H2(last %d) sh=%.3f. PENDING; no live change."
                          % (f["sharpe"], f["mdd"], f["final_x"], f["turnover"], f["trades"],
                             basket2["FULL"]["sharpe"], fee2x["basket"]["gap_FULL_sharpe"],
                             H2_LEN, h["sharpe"]))}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
