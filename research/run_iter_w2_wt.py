
"""W2 15m weight sweep fine (Top5).

Engine mirror of research/run_iter_y1_weights.py / run_weight_modes leg_net
(ONLY engine reference): E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] via
StackVM+FeatureEngineer; MemeBacktest venue=aster lev2 short_enabled
fund0.0005 fee0.0004/fee2x0.0008; quantile q0.3 long-only + cooldown +
stops + vol_scale(vt None->1.0) + roll1.
Top5 locked specs (4h units, scaled x16): ETC(0.88/0.12/cd18/None/ts24)
TRX(0.85/0.12/cd6/0.05/ts24) ATOM(0.85/0.15/cd6/0.05/ts24)
APT(0.88/0.12/cd18/None/ts24) KAS(0.88/0.12/cd6/None/ts24) q0.3.
15m native: data/data_1y/15m/{COIN}.csv (~35040 rows, cols
timestamp,open,high,low,close,volume,quote_volume,trades). No aggregation.
cd/ts/vw x16, BPY=35040.

Sweep A (ETC weight): ETC in {0.1,0.15,0.2,0.25,0.3}, rest equal-split
((1-wETC)/4 each for TRX/ATOM/APT/KAS). Static weights, lev 1.0.
Sweep B (KAS cap contrast): KAS capped at {0.1, 0.2}: weight map equal 0.2
except KAS=min(0.2,cap) with residual/slack equal-split to other four
(cap0.2 -> pure equal; cap0.1 -> KAS 0.1, others 0.225).
Per cell: FULL sharpe/mdd/ann/cum/final_x/n/turnover (+trades) + H2
sharpe/trades + fee2x FULL sharpe + 12fold median.
Incremental: results/iter_W2_wt.json dumped after EACH unit (partial survives).
P0-3 FAIL => verdict PENDING, decision KEEP_equal, no adoption, live untouched.

Smoke mode (tests): ITER_W2_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars,
ETC weights {0.1,0.2}, no KAS-cap arm. OUT/LOG via ITER_W2_OUT/ITER_W2_LOG.
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

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
LOCKED_4H = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
             "APT": LOCKED_APT, "KAS": LOCKED_KAS}
SCALE = 16
BPY = 35040.0
ETC_GRID = [0.1, 0.15, 0.2, 0.25, 0.3]
KAS_CAPS = [0.1, 0.2]

OUT = pathlib.Path(os.getenv("ITER_W2_OUT", "results/iter_W2_wt.json"))
LOG = pathlib.Path(os.getenv("ITER_W2_LOG", "logs/iter_w2_wt.log"))

SMOKE = os.getenv("ITER_W2_SMOKE") == "1"
SMOKE_GRID = [0.1, 0.2]
SMOKE_COINS = ["ETC", "TRX"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def scaled_spec(spec):
    out = dict(spec)
    for k in ("cd", "ts", "vw"):
        if out.get(k) is not None:
            out[k] = int(out[k]) * SCALE
    return out


SPECS = {c: scaled_spec(LOCKED_4H[c]) for c in COINS}


def load15(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"])) for r in rows]


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
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BPY,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
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


def etc_weights(w_etc, coins):
    rest = (1.0 - w_etc) / (len(coins) - 1)
    return {c: (w_etc if c == "ETC" else rest) for c in coins}


def kas_cap_weights(cap, coins):
    w = {c: 0.2 for c in coins}
    if "KAS" not in coins:
        return w
    w["KAS"] = min(0.2, cap)
    others = [c for c in coins if c != "KAS"]
    share = (1.0 - w["KAS"]) / len(others)
    for c in others:
        w[c] = share
    return w


def count_entries(pos):
    n = 0
    for t, v in enumerate(pos):
        if abs(v) > 0.5 and (t == 0 or abs(pos[t - 1]) <= 0.5):
            n += 1
    return n


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
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def fold12_median(net, turn):
    n = len(net)
    fn = n // 12
    ss = []
    for i in range(12):
        a = i * fn
        b = a + fn if i < 11 else n
        ss.append(seg(net, turn, a, b)["sharpe"])
    srt = sorted(ss)
    return {"sharpes": [round(x, 3) for x in ss],
            "median": round(srt[len(srt) // 2], 3),
            "mean": round(sum(ss) / len(ss), 3)}


def dump(etc_rows, kas_rows, n, h2a, final=False):
    arms = {}
    for r in etc_rows:
        arms[r["id"]] = r
    for r in kas_rows:
        arms[r["id"]] = r
    res = {"config": {"engine": "mirror research/run_iter_y1_weights.py leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
                      "formula": list(FORMULA),
                      "locked_4h": {c: dict(LOCKED_4H[c]) for c in COINS},
                      "specs_15m": {c: dict(SPECS[c]) for c in COINS},
                      "scale_4h_to_15m": SCALE, "grid": "15m", "grid_bars": n,
                      "bpy": BPY, "venue": "aster", "lev": LEV, "fund": FUND,
                      "fee": FEE, "fee2x": FEE2X,
                      "etc_grid": list(ETC_GRID), "kas_caps": list(KAS_CAPS),
                      "smoke": SMOKE,
                      "note": "E10 FORMULA untouched; live default equal; offline read-only"},
           "etc_sweep": etc_rows,
           "kas_cap": kas_rows,
           "arms": arms,
           "verdict": "PENDING",
           "decision": "KEEP_equal",
           "decision_note": ("P0-3 permutation FAIL (per-coin p>=0.05) => W2 verdict PENDING; "
                             "contrast only, default stays equal regardless of sweep outcome. "
                             "No adoption, live untouched."),
           "compare": {}}
    if final:
        base = next((r for r in etc_rows if abs(r["w_etc"] - 0.2) < 1e-9), None)
        if base is not None:
            comp = {}
            for r in etc_rows + kas_rows:
                if r["id"] == base["id"]:
                    continue
                f, b = r["FULL"], base["FULL"]
                comp[r["id"]] = {"beats_equal_on_both": bool(f["sharpe"] > b["sharpe"] and f["mdd"] < b["mdd"]),
                                 "d_sharpe": round(f["sharpe"] - b["sharpe"], 3),
                                 "d_mdd": round(f["mdd"] - b["mdd"], 4)}
            res["compare"] = comp
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("dumped %s etc=%d kas=%d final=%s" % (OUT, len(etc_rows), len(kas_rows), final))


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_w2_wt start\n")
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    grid = list(SMOKE_GRID) if SMOKE else list(ETC_GRID)
    bars = {c: load15(c) for c in coins}
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("15m native n=%d h2a=%d coins=%s etc_grid=%s smoke=%s" % (n, h2a, coins, grid, SMOKE))
    assert n > 2000, "15m grid too short: %d" % n
    if not SMOKE:
        assert n > 30000, "15m grid too short: %d" % n
    specs = {c: scaled_spec(LOCKED_4H[c]) if c in LOCKED_4H else scaled_spec(LOCKED_ETC) for c in coins}
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")
    legs, turns, poss = {}, {}, {}
    legs2 = {}
    for c in coins:
        raw, rt, sg = mats[c]
        legs[c], turns[c], poss[c] = leg_net(raw, rt, sg, specs[c], FEE, FUND)
        legs2[c], _, _ = leg_net(raw, rt, sg, specs[c], FEE2X, FUND)

    def eval_weights(w):
        net = [sum(legs[c][t] * w[c] for c in coins) for t in range(n)]
        turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
        net2 = [sum(legs2[c][t] * w[c] for c in coins) for t in range(n)]
        full = seg(net, turn, 0, n)
        full["trades"] = sum(count_entries(poss[c][0:n]) for c in coins)
        h2 = seg(net, turn, h2a, n)
        f2 = seg(net2, turn, 0, n)
        f12 = fold12_median(net, turn)
        return full, h2["sharpe"], sum(count_entries(poss[c][h2a:n]) for c in coins), f2["sharpe"], f12

    etc_rows, kas_rows = [], []
    for w_etc in grid:
        w = etc_weights(w_etc, coins)
        full, h2sh, h2tr, f2sh, f12 = eval_weights(w)
        row = {"id": "etc_%.2f" % w_etc, "kind": "etc_weight", "w_etc": w_etc,
               "weights": {c: round(w[c], 4) for c in coins},
               "FULL": full, "H2_sharpe": h2sh, "H2_trades": h2tr,
               "fee2x_FULL_sharpe": f2sh, "fold12": f12}
        etc_rows.append(row)
        log("ETC w=%.2f FULL sh=%.3f mdd=%.4f fx=%.4f to=%.6f tr=%d H2=%.3f fee2x=%.3f med12=%.3f w=%s" % (
            w_etc, full["sharpe"], full["mdd"], full["final_x"], full["turnover"],
            full["trades"], h2sh, f2sh, f12["median"], row["weights"]))
        dump(etc_rows, kas_rows, n, h2a, final=False)
    if not SMOKE:
        for cap in KAS_CAPS:
            w = kas_cap_weights(cap, COINS)
            full, h2sh, h2tr, f2sh, f12 = eval_weights(w)
            row = {"id": "kas_cap_%.1f" % cap, "kind": "kas_cap", "kas_cap": cap,
                   "weights": {c: round(w[c], 4) for c in coins},
                   "FULL": full, "H2_sharpe": h2sh, "H2_trades": h2tr,
                   "fee2x_FULL_sharpe": f2sh, "fold12": f12}
            kas_rows.append(row)
            log("KAS cap=%.1f FULL sh=%.3f mdd=%.4f fx=%.4f to=%.6f tr=%d H2=%.3f fee2x=%.3f med12=%.3f w=%s" % (
                cap, full["sharpe"], full["mdd"], full["final_x"], full["turnover"],
                full["trades"], h2sh, f2sh, f12["median"], row["weights"]))
            dump(etc_rows, kas_rows, n, h2a, final=False)
    dump(etc_rows, kas_rows, n, h2a, final=True)
    log("wrote %s verdict=PENDING decision=KEEP_equal" % OUT)


if __name__ == "__main__":
    main()
