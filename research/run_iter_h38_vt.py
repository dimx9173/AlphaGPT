"""H38 1h vol-target sweep (Top5): vt {None,0.003,0.006,0.012,0.024} FULL.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native:
cd/ts/vw x4 (4h-bar units -> 1h-bar units); BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows). Equal 0.2 weights.
Top5 locked specs ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Output: results/iter_H38_vt.json, verdict PENDING (P0-3 FAIL),
no adoption, live untouched.
Incremental dump: results JSON rewritten after each unit (partial survives).
Smoke (for tests): ITER_H38_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
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
VT_GRID = [None, 0.003, 0.006, 0.012, 0.024]
OUT = pathlib.Path(os.getenv("ITER_H38_OUT", "results/iter_H38_vt.json"))
LOG = pathlib.Path(os.getenv("ITER_H38_LOG", "logs/iter_H38_vt.log"))
SMOKE = os.getenv("ITER_H38_SMOKE") == "1"


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


def vt_key(vt):
    return "None" if vt is None else str(vt)


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
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net(raw, rt, sig, spec, fee, fund, vt="__spec__"):
    bt = MemeBacktest(
        venue="aster", leverage=LEV, short_enabled=True,
        funding_override=fund, fee_override=fee,
        long_th=spec["lth"], short_th=spec["sth"],
        cooldown_bars=spec["cd"], bars_per_year=BPY,
        stop_loss=spec["sl"], time_stop=spec["ts"],
        vol_target=spec["vt"] if vt == "__spec__" else vt,
        vol_window=spec["vw"],
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
    cum = sum(s)
    return {
        "sharpe": round(sh, 3),
        "ann": round(cum / n * BPY, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(cum, 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }


def dump(state):
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def base_config(coins, n):
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
        "grid": "1h",
        "grid_bars": n,
        "bpy": BPY,
        "data_dir": "data/data_1y/1h",
        "coins": list(coins),
        "smoke": SMOKE,
        "vt_grid": [vt_key(v) for v in VT_GRID],
        "vt_note": "vol_target passed to MemeBacktest (clamp 0.2-2.0 on trailing price-vol, vol_window=scaled vw); None->scale 1.0",
        "note": "H38 1h vol-target sweep; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H38_vt start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    log("common 1h n=%d coins=%s smoke=%s scale=x%d vt_grid=%s" % (n, coins, SMOKE, SCALE, [vt_key(v) for v in VT_GRID]))
    assert n > 2000, n
    cfg = base_config(coins, n)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    sigs = {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        sigs[c] = (raw, rt, sig)
    dump({"status": "partial", "stage": "signals", "config": cfg,
          "coins_done": sorted(sigs)})
    rows = {}
    for vt in VT_GRID:
        key = vt_key(vt)
        legs, turns = {}, {}
        for c in coins:
            raw, rt, sig = sigs[c]
            legs[c], turns[c], _ = leg_net(raw, rt, sig, SPECS[c], FEE, FUND, vt=vt)
        per_coin = {c: seg(legs[c], turns[c], 0, n) for c in coins}
        net = [sum(legs[c][t] * WEIGHT for c in coins) for t in range(n)]
        turn = [sum(turns[c][t] * WEIGHT for c in coins) for t in range(n)]
        basket = seg(net, turn, 0, n)
        rows[key] = {"vt": vt, "per_coin": per_coin, "basket": basket}
        dump({"status": "partial", "stage": "vt_%s" % key, "config": cfg, "rows": rows})
        log("vt=%s basket sh=%.3f mdd=%.4f to=%.6f" % (key, basket["sharpe"], basket["mdd"], basket["turnover"]))
    base = rows["None"]["basket"]
    for key, row in rows.items():
        b = row["basket"]
        row["gap_vs_none_sharpe"] = round(b["sharpe"] - base["sharpe"], 3)
        row["gap_vs_none_mdd"] = round(b["mdd"] - base["mdd"], 4)
        row["gap_vs_none_turnover"] = round(b["turnover"] - base["turnover"], 6)
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H38 verdict PENDING; "
            "vol-target sweep read-only, no adoption, live untouched.")
    res = {"status": "final", "config": cfg, "rows": rows,
           "base_vt": "None", "verdict": verdict, "decision": decision,
           "decision_note": note,
           "conclusion": "H38 PENDING (P0-3 FAIL): 1h vol-target sweep recorded; no live change."}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
