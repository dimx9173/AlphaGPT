"""H13 realized-vs-assumed slippage (1h native, Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, live untouched.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows:
timestamp,open,high,low,close,volume,quote_volume,trades).

Realized leg: parse results/y1b_hourly.jsonl demo rows, match each
executed fill (results[].fill) to its plan price (plans[].price) by
symbol, and compute side-adjusted adverse slippage in bp:

    buy  (want>0): (fill - plan) / plan * 1e4
    sell (want<0): (plan - fill) / plan * 1e4   (shorts sell first)

Positive = adverse (paid worse than quoted). Distribution over fills:
n, mean, median, stdev, min, max, q25, q75, q90, share adverse,
share |slip| > 5bp, plus per-coin breakdown.

Cost-drag leg: portfolio slip drag per bar = turnover * slip * LEV
(H5 symmetric accounting: tx = |dpos| * (fee+slip) * LEV), annualized
as turnover * slip * LEV * BPY. Compare assumed 5bp vs realized mean
(and conservative q90). Turnover comes from the mirrored 1h engine.

Sample-size guard: fewer than MIN_FILLS (=10) realized fills keeps the
comparison DESCRIPTIVE ONLY (verdict PENDING regardless).

IMPORTANT: results/iter_H13_slipreal.json is dumped after EACH unit
(legs, realized parse, compare), so a killed run still leaves partial
rows behind (partial survives).

Outputs: results/iter_H13_slipreal.json (+ logs/iter_H13_slipreal.log).
Offline read-only: reads data/data_1y/1h/*.csv + results/y1b_hourly.jsonl
only. No orders, no broker imports.

Smoke mode (for tests): ITER_H13_SMOKE=1 shrinks to coins {ETC,TRX},
first 3000 bars (realized parse still reads the full jsonl: it is tiny).
ITER_H13_OUT / ITER_H13_LOG override output paths (tests use a temp
file so the committed artifact is not clobbered).
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
    FEE,
    FUND,
    LEV,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 8760.0
SCALE = 4
ASSUMED_SLIP_BP = 5.0
MIN_FILLS = 10
FILLS_SRC = "results/y1b_hourly.jsonl"

OUT = pathlib.Path(os.getenv("ITER_H13_OUT", "results/iter_H13_slipreal.json"))
LOG = pathlib.Path(os.getenv("ITER_H13_LOG", "logs/iter_H13_slipreal.log"))

SMOKE = os.getenv("ITER_H13_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_BARS = 3000


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins, cap=None):
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
    if cap is not None and n > cap:
        n = cap
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


def leg_net(raw, rt, sig, spec, fee=FEE, fund=FUND):
    """Mirrored locked engine leg (quantile q0.3 long-only + cooldown +
    stops + vol_scale(vt None->1.0) + roll1). Returns (net, turn, pos)."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY, stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
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
    return ((gross - tx - fnd)[0].tolist(), turn[0].tolist(),
            (lp - sp)[0].tolist())


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def adverse_bp(want, plan, fill):
    """Side-adjusted adverse slippage in bp. Positive = paid worse than
    the plan quote. Buys: (fill-plan)/plan; sells/shorts: (plan-fill)/plan."""
    if plan is None or fill is None:
        return None
    try:
        plan = float(plan)
        fill = float(fill)
        want = float(want)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(plan) or not math.isfinite(fill):
        return None
    if plan <= 0 or want == 0:
        return None
    if want > 0:
        return (fill - plan) / plan * 1e4
    return (plan - fill) / plan * 1e4


def parse_fills(path=FILLS_SRC):
    """Match executed demo fills to plan prices by symbol.

    Returns (fills, skipped): fills = [{ts, symbol, coin, side_want,
    plan, fill, slip_bp}]; skipped counts rows that could not be matched.
    """
    fills = []
    skipped = {"rows_no_plans": 0, "results_skipped": 0,
               "no_matching_plan": 0, "bad_price": 0}
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return fills, dict(skipped, missing_file=True)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            skipped["rows_no_plans"] += 1
            continue
        plans = row.get("plans") or []
        pmap = {}
        for p in plans:
            try:
                pmap[p.get("symbol")] = p
            except AttributeError:
                continue
        if not pmap:
            skipped["rows_no_plans"] += 1
            continue
        for x in (row.get("results") or []):
            if not isinstance(x, dict) or x.get("skipped"):
                skipped["results_skipped"] += 1
                continue
            if x.get("fill") is None or not x.get("ok"):
                skipped["results_skipped"] += 1
                continue
            p = pmap.get(x.get("symbol"))
            if p is None:
                skipped["no_matching_plan"] += 1
                continue
            sb = adverse_bp(p.get("want"), p.get("price"), x.get("fill"))
            if sb is None:
                skipped["bad_price"] += 1
                continue
            fills.append({"ts": row.get("ts"), "symbol": x.get("symbol"),
                          "coin": p.get("coin"), "side_want": p.get("want"),
                          "plan": float(p.get("price")),
                          "fill": float(x.get("fill")),
                          "slip_bp": round(sb, 4)})
    return fills, skipped


def quantile(vals, p):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return float(s[0])
    k = (n - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarize(vals):
    """Slippage distribution summary (bp units in, bp units out)."""
    v = [float(x) for x in vals]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0,
                "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0,
                "q90": 0.0, "share_adverse": 0.0, "share_abs_gt5": 0.0}
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "median": round(quantile(v, 0.5), 4),
            "stdev": round(sd, 4), "min": round(min(v), 4),
            "max": round(max(v), 4), "q25": round(quantile(v, 0.25), 4),
            "q75": round(quantile(v, 0.75), 4),
            "q90": round(quantile(v, 0.90), 4),
            "share_adverse": round(sum(1 for x in v if x > 0) / n, 4),
            "share_abs_gt5": round(sum(1 for x in v if abs(x) > 5) / n, 4)}


def slip_drag_ann(turnover, slip_frac, lev=LEV, bpy=BPY):
    """Annualized portfolio slip drag (cum units): turnover * slip * LEV * BPY."""
    return float(turnover) * float(slip_frac) * float(lev) * float(bpy)


def base_config(coins, n):
    return {
        "engine": "mirror run_weight_modes.py leg_net + quantile q0.3 "
                  "long-only + cooldown + stops + vol_scale(vt None->1.0) "
                  "+ roll1; static equal 0.2 Top5; 1h native cd/ts/vw x4",
        "formula": list(FORMULA),
        "basket": {c: {"lth": SPECS[c]["lth"], "sth": SPECS[c]["sth"],
                       "cd": SPECS[c]["cd"], "sl": SPECS[c]["sl"],
                       "ts": SPECS[c]["ts"], "q": SPECS[c]["q"]}
                   for c in coins},
        "weights": {c: W[c] for c in coins},
        "venue": "aster",
        "lev": LEV,
        "fee": FEE,
        "fund": FUND,
        "assumed_slip_bp": ASSUMED_SLIP_BP,
        "assumed_slip": ASSUMED_SLIP_BP * 1e-4,
        "fills_src": FILLS_SRC,
        "min_fills": MIN_FILLS,
        "slip_def": "adverse bp vs plan quote: buy (fill-plan)/plan*1e4, "
                    "sell/short (plan-fill)/plan*1e4; positive = worse",
        "drag_def": "slip drag/yr (cum units) = turnover * slip * LEV * BPY "
                    "(H5 symmetric accounting)",
        "smoke": SMOKE,
        "grid": "1h",
        "grid_bars": n,
        "bpy": BPY,
        "scale": SCALE,
        "coins": list(coins),
        "note": "E10 FORMULA untouched; offline read-only; P0-3 FAIL => PENDING",
    }


def dump(doc):
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H13_slipreal start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    bars, n = common1h(coins, cap=SMOKE_BARS if SMOKE else None)
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d" % (n, coins, SMOKE, SCALE))
    assert n > 2000, "grid too short: %d" % n
    cfg = base_config(coins, n)
    dump({"status": "partial", "stage": "loaded", "config": cfg,
          "verdict": "PENDING"})

    legs, turns = {}, {}
    for c in coins:
        raw, rt, sg = build_sig(bars[c])
        legs[c], turns[c], _ = leg_net(raw, rt, sg, SPECS[c])
        log("leg %s cum=%.4f" % (c, sum(legs[c])))
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg,
              "legs_done": sorted(legs), "verdict": "PENDING"})

    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    basket = seg(net, turn, 0, n)
    per_coin = {c: seg(legs[c], turns[c], 0, n) for c in coins}
    log("basket turnover=%.6f cum=%.4f" % (basket["turnover"], basket["cum"]))
    dump({"status": "partial", "stage": "engine", "config": cfg,
          "basket": basket, "per_coin": per_coin, "verdict": "PENDING"})

    fills, skipped = parse_fills()
    slips = [f["slip_bp"] for f in fills]
    dist = summarize(slips)
    by_coin = {}
    for c in sorted({f["coin"] for f in fills if f.get("coin")}):
        by_coin[str(c)] = summarize([f["slip_bp"] for f in fills
                                     if f.get("coin") == c])
    log("fills n=%d mean=%s median=%s skipped=%s"
        % (dist["n"], dist["mean"], dist["median"], skipped))
    dump({"status": "partial", "stage": "realized", "config": cfg,
          "basket": basket, "per_coin": per_coin,
          "realized": {"n_fills": len(fills), "fills": fills,
                       "dist_bp": dist, "by_coin_bp": by_coin,
                       "skipped": skipped},
          "verdict": "PENDING"})

    assumed = ASSUMED_SLIP_BP * 1e-4
    real_mean = (dist["mean"] * 1e-4) if dist["n"] else 0.0
    real_q90 = (dist["q90"] * 1e-4) if dist["n"] else 0.0
    to = basket["turnover"]
    drag_assumed = slip_drag_ann(to, assumed)
    drag_mean = slip_drag_ann(to, real_mean)
    drag_q90 = slip_drag_ann(to, real_q90)
    fee_drag = slip_drag_ann(to, FEE)
    cmp = {"turnover": to,
           "assumed_bp": ASSUMED_SLIP_BP,
           "realized_mean_bp": dist["mean"],
           "realized_q90_bp": dist["q90"],
           "drag_assumed_yr": round(drag_assumed, 4),
           "drag_realized_mean_yr": round(drag_mean, 4),
           "drag_realized_q90_yr": round(drag_q90, 4),
           "fee_drag_yr": round(fee_drag, 4),
           "drag_ratio_mean_over_assumed": (
               round(drag_mean / drag_assumed, 4) if drag_assumed else 0.0),
           "conservative_covers_q90": bool(drag_assumed >= drag_q90),
           "n_fills": dist["n"],
           "sample_adequate": bool(dist["n"] >= MIN_FILLS)}
    log("drag assumed=%.4f realized_mean=%.4f q90=%.4f fee=%.4f" %
        (drag_assumed, drag_mean, drag_q90, fee_drag))

    verdict = "PENDING"
    decision = "NO_ADOPTION"
    note = ("P0-3 permutation FAIL => H13 verdict PENDING; realized sample "
            "n=%d (< %d => descriptive only); no adoption, live untouched."
            % (dist["n"], MIN_FILLS))
    res = {"status": "final", "config": cfg, "basket": basket,
           "per_coin": per_coin,
           "realized": {"n_fills": len(fills), "fills": fills,
                        "dist_bp": dist, "by_coin_bp": by_coin,
                        "skipped": skipped},
           "compare": cmp,
           "verdict": verdict, "decision": decision,
           "decision_note": note,
           "conclusion": ("H13 1h Top5 slip-real: assumed 5bp drag/yr=%.4f vs "
                          "realized mean %.2fbp drag/yr=%.4f (q90 %.2fbp "
                          "drag/yr=%.4f); fills n=%d. PENDING; no live change."
                          % (drag_assumed, dist["mean"], drag_mean,
                             dist["q90"], drag_q90, dist["n"]))}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
