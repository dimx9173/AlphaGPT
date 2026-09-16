"""Y5 15m coin scan (28 coins) -- L0/L1 FUNNEL ONLY, DIAGNOSTIC.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING (\u5f85\u5b9a)" and MUST
NOT be used as demo-listing evidence. No adoption, no live change, live
chain untouched. Offline read-only: reads data/data_1y/15m/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net (via Y7 15m-native):
E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer;
MemeBacktest venue=aster lev2 short_enabled fund0.0005; quantile q0.3
long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1.

Per-coin single-leg scan (no weighting; weights N/A for single legs):
  L0 eligibility (no backtest): n>=17280 bars (half-year 15m), missing<1%.
  L1 gate: 3 templates (ETC-style lth0.88/sth0.12/slNone/ts24/vtNone/vw12/
    q0.3 x cd{6,12,18}); best = max base-fee FULL sharpe;
    pass = final_x>1 & sharpe>0 & fee2x FULL sharpe>0
    (fee2x = full-segment recompute with fee 0.0008).

15m native: cd/ts/vw x16, BPY=35040. Each coin runs on its OWN native grid
(no timestamp intersection). Signal built ONCE per coin, reused by all 3
templates x 2 fee cells (6 leg evals).

Incremental writes: results/iter_Y5_scan.json is dumped after EACH coin
(partial=true) so partial progress survives interruption; final dump has
partial=false. Focus six BTC/ETH/SOL/BNB/LINK/LTC get per-template detail.

Output: results/iter_Y5_scan.json (+ logs/iter_y5_scan.log).
Verdict PENDING; no coins added.

Smoke mode (for tests): ITER_Y5_SMOKE=1 shrinks to coins {BTC,ETH} full
grid (2 coins). ITER_Y5_OUT / ITER_Y5_LOG override paths (tests use temp
files so the committed FULL artifact is not clobbered).
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
from strategy_manager.config import FORMULA, LEV, FUND, FEE, FEE2X

assert LEV == 2.0, "LEV lock broken: %r" % LEV

UNIVERSE = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
            "DOGE", "DOT", "ETC", "ETH", "HBAR", "ICP", "KAS", "LINK",
            "LTC", "NEAR", "PEPE", "POL", "RENDER", "SHIB", "SOL", "SUI",
            "TRX", "UNI", "XLM", "XRP"]
FOCUS6 = ["BTC", "ETH", "SOL", "BNB", "LINK", "LTC"]
TEMPLATES = [
    {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24,
     "vt": None, "vw": 12, "q": 0.3},
    {"lth": 0.88, "sth": 0.12, "cd": 12, "sl": None, "ts": 24,
     "vt": None, "vw": 12, "q": 0.3},
    {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24,
     "vt": None, "vw": 12, "q": 0.3},
]
BPY = 35040.0
SCALE = 16
L0_MIN_BARS = 17280
L0_MAX_MISSING = 0.01
WATCH_FEE2X = 0.8
WATCH_SHARPE = 1.0
OUT = pathlib.Path(os.getenv("ITER_Y5_OUT", "results/iter_Y5_scan.json"))
LOG = pathlib.Path(os.getenv("ITER_Y5_LOG", "logs/iter_y5_scan.log"))
SMOKE = os.getenv("ITER_Y5_SMOKE") == "1"
SMOKE_COINS = ["BTC", "ETH"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load15m(coin):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def l0_check(coin):
    rows = load15m(coin)
    n = len(rows)
    if n >= 2:
        span = rows[-1][0] - rows[0][0]
        exp = span // 900000 + 1
        miss = 1.0 - n / exp if exp > 0 else 1.0
    else:
        miss = 1.0
    ok = n >= L0_MIN_BARS and miss < L0_MAX_MISSING
    return {"n": n, "missing": round(miss, 6), "pass": bool(ok)}, rows


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


def sharpe_of(net):
    n = len(net)
    if n < 10:
        return 0.0
    m = sum(net) / n
    v = sum((x - m) ** 2 for x in net) / (n - 1)
    return m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0


def scan_coin(coin, rows):
    bars = [(r[1], r[2], r[3], r[4], r[5]) for r in rows]
    raw, rt, sig = build_sig(bars)
    n = len(bars)
    tmpl_rows = []
    for spec in TEMPLATES:
        net, turn = leg_net(raw, rt, sig, spec, FEE, FUND)
        full = seg(net, turn, 0, n)
        net2x, _ = leg_net(raw, rt, sig, spec, FEE2X, FUND)
        sh2x = round(sharpe_of(net2x), 3)
        tmpl_rows.append({"template": dict(spec), "FULL": full,
                          "fee2x_sharpe": sh2x})
    best = max(tmpl_rows, key=lambda r: r["FULL"]["sharpe"])
    full = best["FULL"]
    l1 = full["final_x"] > 1 and full["sharpe"] > 0 and best["fee2x_sharpe"] > 0
    return tmpl_rows, best, bool(l1)


def dump(out_path, coins, rows, focus, partial):
    by = {r["coin"]: r for r in rows}
    universe = list(SMOKE_COINS) if SMOKE else list(UNIVERSE)
    add, watch, reject = [], [], []
    for r in rows:
        if r["L1_pass"]:
            add.append(r["coin"])
        elif (r.get("fee2x") or 0) >= WATCH_FEE2X or (r.get("sharpe") or 0) >= WATCH_SHARPE:
            watch.append(r["coin"])
        else:
            reject.append(r["coin"])
    done = [r["coin"] for r in rows]
    missing = [c for c in coins if c not in done]
    res = {
        "iter": "Y5",
        "scope": "28-coin 15m-native L0/L1 funnel (single-leg, no weighting; no L2 adoption)",
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; single-leg per coin; 15m native cd/ts/vw x16",
            "formula": list(FORMULA),
            "templates": [dict(t) for t in TEMPLATES],
            "weights": "N/A (single-leg scan; no basket weighting)",
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "fee2x": FEE2X,
            "grid": "15m",
            "bpy": BPY,
            "scale": SCALE,
            "l0_min_bars": L0_MIN_BARS,
            "l0_max_missing": L0_MAX_MISSING,
            "data": "data/data_1y/15m",
            "smoke": SMOKE,
            "note": "L0/L1 funnel only, read-only; live chain untouched",
        },
        "universe": universe,
        "focus6": {c: by[c] for c in FOCUS6 if c in by},
        "rows": rows,
        "addable_pending": sorted(add),
        "watch": sorted(watch),
        "reject": sorted(reject),
        "partial": partial,
        "done": len(rows),
        "missing": missing,
        "verdict": "PENDING",
        "decision": "NO_COINS_ADDED",
        "conclusion": "PENDING (\u5f85\u5b9a, P0-3 FAIL); no coins added",
        "offline": True,
    }
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Y5_scan start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(FOCUS6) + [c for c in UNIVERSE if c not in FOCUS6]
    log("universe n=%d smoke=%s order=%s" % (len(coins), SMOKE, coins))
    rows, focus = [], {}
    for i, coin in enumerate(coins):
        l0, raw_rows = l0_check(coin)
        if not l0["pass"]:
            row = {"coin": coin, "L0_pass": False, "L0": l0,
                   "L1_pass": False, "final_x": 0.0, "sharpe": 0.0,
                   "fee2x": 0.0, "best_template": None,
                   "verdict": "REJECT_L0"}
            log("[%d/%d] %s L0 FAIL n=%d missing=%.4f" % (i + 1, len(coins), coin, l0["n"], l0["missing"]))
        else:
            tmpl_rows, best, l1 = scan_coin(coin, raw_rows)
            full = best["FULL"]
            row = {"coin": coin, "L0_pass": True, "L0": l0,
                   "L1_pass": l1, "final_x": full["final_x"],
                   "sharpe": full["sharpe"], "fee2x": best["fee2x_sharpe"],
                   "best_template": dict(best["template"]),
                   "FULL": full,
                   "verdict": "PASS_ALL_PENDING" if l1 else "REJECT_L1"}
            if coin in FOCUS6:
                row["templates"] = tmpl_rows
            log("[%d/%d] %s L1 %s final_x=%.3f sharpe=%.3f fee2x=%.3f cd=%s" % (
                i + 1, len(coins), coin, "PASS" if l1 else "fail",
                full["final_x"], full["sharpe"], best["fee2x_sharpe"],
                best["template"]["cd"]))
        rows.append(row)
        if coin in FOCUS6:
            focus[coin] = row
        dump(OUT, coins, rows, focus, partial=True)
    res = dump(OUT, coins, rows, focus, partial=False)
    log("wrote %s done=%d add=%s watch=%s reject=%d" % (
        OUT, res["done"], res["addable_pending"], res["watch"], len(res["reject"])))
    if not SMOKE:
        assert res["done"] == 28 and not res["missing"], (res["done"], res["missing"])


if __name__ == "__main__":
    main()
