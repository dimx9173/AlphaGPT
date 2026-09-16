"""H39 1h session split (Top5) -- Asia/EU/US 8h per-coin sharpe/trades + session-only arms.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native: reads
data/data_1y/1h directly (no aggregation); cd/ts/vw x4 (4h-bar units
-> 1h-bar units); BPY=8760. Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Sessions by bar UTC hour (8h blocks): asia 00-07, eu 08-15, us 16-23.
Per-coin session slices: subset stats of the base leg net series over
session bars only (sharpe/ann annualised with BPY_SESS=2920 = 8760/3;
mdd over the session-only equity curve; trades = masked counts where
the transition bar t falls in the session, so session trades sum to
FULL trades). Session-only arms: per-coin positions gated to one
session (zero outside), leg economics recomputed with fee+funding on
the gated positions (boundary exit/re-entry costs included), equal
0.2 basket mix per arm.

trades = opened directional trades = entries (flat->non-zero) + flips
(long<->short direct switch); exits reported separately.

Output: results/iter_H39_session.json with per-coin FULL + sessions +
3 session-only arms.

Offline read-only; verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Incremental dump: results JSON rewritten after each unit (partial survives).
Smoke (for tests): ITER_H39_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
"""
import csv
import datetime
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
BPY_SESS = 2920.0
SCALE = 4
WEIGHT = 0.2
SESSIONS = {
    "asia": frozenset(range(0, 8)),
    "eu": frozenset(range(8, 16)),
    "us": frozenset(range(16, 24)),
}
OUT = pathlib.Path(os.getenv("ITER_H39_OUT", "results/iter_H39_session.json"))
LOG = pathlib.Path(os.getenv("ITER_H39_LOG", "logs/iter_H39_session.log"))
SMOKE = os.getenv("ITER_H39_SMOKE") == "1"


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


def session_of_hour(h):
    for name, hours in SESSIONS.items():
        if h in hours:
            return name
    raise ValueError("hour out of range: %r" % h)


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


def count_trades_masked(pos, idxset):
    """Trade counts attributed to bars in idxset.

    Transition at bar t (vs t-1, or flat before grid start) is counted
    iff t is in idxset, so per-session counts sum to the FULL count.
    """
    entries = flips = exits = 0
    prev = 0.0
    for t in range(len(pos)):
        cur = pos[t]
        kind = None
        if cur != 0.0 and prev == 0.0:
            kind = "entries"
        elif cur == 0.0 and prev != 0.0:
            kind = "exits"
        elif cur != 0.0 and prev != 0.0 and (cur > 0) != (prev > 0):
            kind = "flips"
        if kind is not None and t in idxset:
            if kind == "entries":
                entries += 1
            elif kind == "flips":
                flips += 1
            else:
                exits += 1
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


def seg_subset(net, turn, pos, idx):
    """Session-slice stats over subset bars idx (session-only equity curve).

    Annualised with BPY_SESS=2920 (one third of the 1h grid year).
    """
    idxs = set(idx)
    s = [net[i] for i in idx]
    t = [turn[i] for i in idx]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY_SESS) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    out = {
        "sharpe": round(sh, 3),
        "ann": round(cum / n * BPY_SESS, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(cum, 4),
        "final_x": round(1.0 + cum, 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }
    out.update(count_trades_masked(pos, idxs))
    return out


def gated_leg(rt, pos, idxset, fee, fund):
    """Session-only arm leg: positions gated to idxset, economics recomputed.

    Boundary exit/re-entry costs are included via gated turnover.
    """
    n = len(pos)
    g = [pos[t] if t in idxset else 0.0 for t in range(n)]
    net = []
    turn = []
    prev = 0.0
    for t in range(n):
        d = abs(g[t] - prev)
        turn.append(d)
        net.append(g[t] * rt[t] * LEV - d * fee * LEV - g[t] * fund * LEV)
        prev = g[t]
    return net, turn, g


def dump(state):
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def base_config(coins, n, sess_idx):
    return {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins},
        "scale_4h_to_1h": SCALE,
        "weights": {c: WEIGHT for c in coins},
        "sessions": {k: sorted(v) for k, v in SESSIONS.items()},
        "session_hours": "asia 00-07 / eu 08-15 / us 16-23 UTC (8h blocks)",
        "session_bars": {k: len(v) for k, v in sess_idx.items()},
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "fee2x": FEE2X,
        "grid": "1h",
        "grid_bars": n,
        "bpy": BPY,
        "bpy_session": BPY_SESS,
        "data_dir": "data/data_1y/1h",
        "coins": list(coins),
        "smoke": SMOKE,
        "trades_def": "trades = entries (flat->non-zero) + flips (long<->short); exits reported separately; session-slice trades attributed by transition bar so sessions sum to FULL",
        "arms_def": "session-only arm = per-coin positions gated to session hours (zero outside), leg economics recomputed with fee+funding, equal-weight basket mix",
        "note": "H39 1h Asia/EU/US session split; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H39_session start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    hours = [datetime.datetime.utcfromtimestamp(t / 1000).hour for t in ts]
    sess_idx = {k: [i for i, h in enumerate(hours) if h in v]
                for k, v in SESSIONS.items()}
    log("common 1h n=%d coins=%s smoke=%s scale=x%d sess=%s"
        % (n, coins, SMOKE, SCALE, {k: len(v) for k, v in sess_idx.items()}))
    assert n > 2000, n
    assert sum(len(v) for v in sess_idx.values()) == n
    assert all(len(v) > 100 for v in sess_idx.values())
    cfg = base_config(coins, n, sess_idx)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    legs, turns, poss, rts = {}, {}, {}, {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs[c], turns[c], poss[c] = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
        rts[c] = rt[0].tolist()
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg,
              "legs_done": sorted(legs)})
        log("leg %s done" % c)
    per_coin = {}
    for c in coins:
        pc = {"FULL": seg(legs[c], turns[c], poss[c], 0, n)}
        for k in ("asia", "eu", "us"):
            pc[k] = seg_subset(legs[c], turns[c], poss[c], sess_idx[k])
        per_coin[c] = pc
        dump({"status": "partial", "stage": "sess_%s" % c, "config": cfg,
              "per_coin_done": sorted(per_coin)})
        f = pc["FULL"]
        log("coin %s FULL sh=%.3f trades=%d | asia sh=%.3f t=%d | eu sh=%.3f t=%d | us sh=%.3f t=%d"
            % (c, f["sharpe"], f["trades"],
               pc["asia"]["sharpe"], pc["asia"]["trades"],
               pc["eu"]["sharpe"], pc["eu"]["trades"],
               pc["us"]["sharpe"], pc["us"]["trades"]))
    w = 1.0 / len(coins)
    arms = {}
    for k in ("asia", "eu", "us"):
        idxs = set(sess_idx[k])
        glegs, gturns, gposs = {}, {}, {}
        for c in coins:
            glegs[c], gturns[c], gposs[c] = gated_leg(rts[c], poss[c], idxs, FEE, FUND)
        pcg = {c: seg(glegs[c], gturns[c], gposs[c], 0, n) for c in coins}
        net = [sum(glegs[c][t] * w for c in coins) for t in range(n)]
        turn = [sum(gturns[c][t] * w for c in coins) for t in range(n)]
        pos = [sum(gposs[c][t] * w for c in coins) for t in range(n)]
        basket = seg(net, turn, pos, 0, n)
        for fld in ("trades", "entries", "flips", "exits"):
            basket[fld] = sum(pcg[c][fld] for c in coins)
        arms[k] = {"basket": basket, "per_coin": pcg}
        dump({"status": "partial", "stage": "arm_%s" % k, "config": cfg,
              "per_coin": per_coin, "arms_done": sorted(arms)})
        log("arm %s-only basket sh=%.3f mdd=%.4f final_x=%.4f to=%.6f trades=%d"
            % (k, basket["sharpe"], basket["mdd"], basket["final_x"],
               basket["turnover"], basket["trades"]))
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H39 verdict PENDING; "
            "session split descriptive only, no adoption, live untouched.")
    best = max(arms.items(), key=lambda kv: kv[1]["basket"]["sharpe"])
    res = {"status": "final", "config": cfg, "per_coin": per_coin,
           "arms": arms,
           "verdict": verdict, "decision": decision, "decision_note": note,
           "conclusion": ("H39 1h Top5 session split FULL-grid session-only arms: "
                          "asia sh=%.3f eu sh=%.3f us sh=%.3f (best %s). "
                          "PENDING; no live change."
                          % (arms["asia"]["basket"]["sharpe"],
                             arms["eu"]["basket"]["sharpe"],
                             arms["us"]["basket"]["sharpe"], best[0]))}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
