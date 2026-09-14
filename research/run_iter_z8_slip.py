"""Z8 15m slip sensitivity (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change (FEE/FUND/LEV untouched).

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

15m native: cd/ts/vw x16, BPY=35040. Data data/data_1y/15m/{COIN}.csv
(~35040 rows: timestamp,open,high,low,close,volume,quote_volume,trades).

Sweep: slip in {0,2,5,10,20}bp at fixed fee=FEE fund=FUND ->
FULL sharpe slope (+ann/mdd/cum/final_x/n/turnover). slip fraction =
slip_bp * 1e-4; effective per-turn cost = fee + slip. Signed-funding
accounting mirrors leg_net (fnd = pos*fund*LEV); turnover invariant
across slip cells (positions precomputed once, slip cells are pure
arithmetic).

IMPORTANT: results/iter_Z8_slip.json is dumped after EACH evaluation
(incremental; "status": "partial" until the final write), so a killed run
still leaves partial rows behind.

Outputs: results/iter_Z8_slip.json (+ logs/iter_Z8_slip.log).
Offline read-only: reads data/data_1y/15m/*.csv only. No orders.

Smoke mode (for tests): ITER_Z8_SMOKE=1 shrinks to coins {ETC,TRX} and
the first 3000 bars. ITER_Z8_OUT / ITER_Z8_LOG override output paths
(tests use a temp file so the committed artifact is not clobbered).
"""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE

assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert abs(FUND - 0.0005) < 1e-12, "FUND lock broken: %r" % FUND
assert abs(FEE - 0.0004) < 1e-12, "FEE lock broken: %r" % FEE

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
SLIP_BP_GRID = [0, 2, 5, 10, 20]
FEE_FIXED = FEE
FUND_FIXED = FUND
LEV_USE = 2.0
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path(os.getenv("ITER_Z8_OUT", "results/iter_Z8_slip.json"))
LOG = pathlib.Path(os.getenv("ITER_Z8_LOG", "logs/iter_Z8_slip.log"))
SMOKE = os.getenv("ITER_Z8_SMOKE") == "1"

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common15m(coins):
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
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
    raw = {"open": torch.tensor([[b[0] for b in bars]]), "high": torch.tensor([[b[1] for b in bars]]), "low": torch.tensor([[b[2] for b in bars]]), "close": torch.tensor([[b[3] for b in bars]]), "volume": torch.tensor([[b[4] for b in bars]]), "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
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

def base_series(raw, rt, sig, spec):
    """Executed position/turnover/gross series; fee/fund/slip independent."""
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=0.0, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    pos = lp - sp
    gross = pos * rt * bt.leverage
    return pos[0].tolist(), turn[0].tolist(), gross[0].tolist()

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def lin_slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den

def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z8_slip start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, n = common15m(coins)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d fee=%.4f fund=%.4f" % (n, coins, SMOKE, SCALE, FEE_FIXED, FUND_FIXED))
    assert n > 2000, n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    legs = {}
    for c in coins:
        raw, rt, sg = mats[c]
        legs[c] = base_series(raw, rt, sg, SPECS[c])
    log("base position series precomputed")
    G = [sum(legs[c][2][t] * w[c] for c in coins) for t in range(n)]
    T = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    P = [sum(legs[c][0][t] * w[c] for c in coins) for t in range(n)]

    def eval_slip(slip_bp):
        slip = slip_bp * 1e-4
        eff = FEE_FIXED + slip
        net = [G[t] - T[t] * eff * LEV_USE - P[t] * FUND_FIXED * LEV_USE for t in range(n)]
        return seg(net, T, 0, n), slip, eff

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 15m native cd/ts/vw x16; positions precomputed once, slip cells arithmetic", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "slip_bp_grid": list(SLIP_BP_GRID), "fee": FEE_FIXED, "fund": FUND_FIXED, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "smoke": SMOKE, "note": "slip sweep only, read-only; live FEE/FUND/LEV untouched"},
        "status": "partial", "curve": {"rows": []},
        "verdict": "PENDING"}
    dump(res)

    for bp in SLIP_BP_GRID:
        st, slip, eff = eval_slip(bp)
        res["curve"]["rows"].append({"slip_bp": bp, "slip": round(slip, 5), "fee": FEE_FIXED, "fund": FUND_FIXED, "eff_fee": round(eff, 5), **st})
        dump(res)
        log("slip %dbp FULL sh=%.3f final_x=%.4f mdd=%.4f cum=%.4f to=%.6f" % (bp, st["sharpe"], st["final_x"], st["mdd"], st["cum"], st["turnover"]))
    rows = res["curve"]["rows"]
    mono_down = all(rows[i + 1]["sharpe"] <= rows[i]["sharpe"] + 1e-9 for i in range(len(rows) - 1))
    to_flat = len({r["turnover"] for r in rows}) == 1
    sh = [r["sharpe"] for r in rows]
    fx = [r["final_x"] for r in rows]
    slope = lin_slope([float(r["slip_bp"]) for r in rows], sh)
    res["curve"]["monotone_down"] = bool(mono_down)
    res["curve"]["turnover_flat"] = bool(to_flat)
    res["curve"]["sharpe_slope_per_bp"] = round(slope, 5)
    res["curve"]["sharpe_drop_0_to_20bp"] = round(sh[0] - sh[-1], 3)
    res["curve"]["final_x_monotone_down"] = bool(all(fx[i + 1] <= fx[i] + 1e-9 for i in range(len(fx) - 1)))
    dump(res)

    res["verdict"] = "PENDING"
    res["verdict_note"] = "PENDING (\u5f85\u5b9a): P0-3 permutation FAILED, no adoption, no live change; slip sweep is diagnostic only."
    res["decision"] = "NO_CHANGE"
    res["decision_note"] = "\u5f85\u5b9a,\u4e0d\u52a8FEE/FUND/LEV\u300215m\u539f\u751f\u683c\u5b50\u3002"
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s (diagnostic only)" % (OUT, res["verdict"]))

if __name__ == "__main__":
    main()
