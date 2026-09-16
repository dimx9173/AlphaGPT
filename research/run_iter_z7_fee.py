"""Z7 fee sensitivity (15m native, Top5) -- DIAGNOSTIC ONLY.

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

Sweep: fee in {0.0002, 0.0004, 0.0008, 0.0016} at fixed fund 0.0005 ->
FULL sharpe/final_x (+ann/mdd/cum/n/turnover). Break-even fee via upward
bracket-doubling from grid hi + 20-iteration bisection on failure
threshold sharpe<0.05. Signed-funding accounting mirrors leg_net
(fnd = pos*fund*LEV); turnover invariant across fee cells (positions
precomputed once, fee cells are pure arithmetic).

IMPORTANT: results/iter_Z7_fee.json is dumped after EACH evaluation
(incremental; "status": "partial" until the final write), so a killed run
still leaves partial rows behind.

Outputs: results/iter_Z7_fee.json (+ logs/iter_Z7_fee.log).
Offline read-only: reads data/data_1y/15m/*.csv only. No orders.

Smoke mode (for tests): ITER_Z7_SMOKE=1 shrinks to coins {ETC,TRX} and
the first 3000 bars. ITER_Z7_OUT / ITER_Z7_LOG override output paths
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

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
FEE_GRID = [0.0002, 0.0004, 0.0008, 0.0016]
FUND_FIXED = 0.0005
assert abs(FUND_FIXED - FUND) < 1e-12, "FUND lock broken: %r vs %r" % (FUND_FIXED, FUND)
FAIL_SHARPE = 0.05
LEV_USE = 2.0
BPY = 35040.0
SCALE = 16
BISECT_ITERS = 20
OUT = pathlib.Path(os.getenv("ITER_Z7_OUT", "results/iter_Z7_fee.json"))
LOG = pathlib.Path(os.getenv("ITER_Z7_LOG", "logs/iter_Z7_fee.log"))
SMOKE = os.getenv("ITER_Z7_SMOKE") == "1"

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

def base_series(raw, rt, sig, spec):
    """Executed position/turnover/gross series; fee independent."""
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

def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z7_fee start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, n = common15m(coins)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d fund=%.4f" % (n, coins, SMOKE, SCALE, FUND_FIXED))
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

    def eval_fee(fee):
        net = [G[t] - T[t] * fee * LEV_USE - P[t] * FUND_FIXED * LEV_USE for t in range(n)]
        return seg(net, T, 0, n)

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 15m native cd/ts/vw x16; positions precomputed once, fee cells arithmetic", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "fee_grid": list(FEE_GRID), "fund": FUND_FIXED, "fail_sharpe": FAIL_SHARPE, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "smoke": SMOKE, "note": "fee sweep only, read-only; live FEE/FUND/LEV untouched"},
        "status": "partial", "curve": {"rows": []}, "breakeven": {"bisection_rows": []},
        "verdict": "PENDING"}
    dump(res)

    for fee in FEE_GRID:
        st = eval_fee(fee)
        res["curve"]["rows"].append({"fee": fee, **st})
        dump(res)
        log("fee %.4f FULL sh=%.3f final_x=%.4f mdd=%.4f cum=%.4f to=%.6f" % (fee, st["sharpe"], st["final_x"], st["mdd"], st["cum"], st["turnover"]))
    rows = res["curve"]["rows"]
    mono_down = all(rows[i + 1]["sharpe"] <= rows[i]["sharpe"] + 1e-9 for i in range(len(rows) - 1))
    to_flat = len({r["turnover"] for r in rows}) == 1
    res["curve"]["monotone_down"] = bool(mono_down)
    res["curve"]["turnover_flat"] = bool(to_flat)
    dump(res)

    lo, hi = FEE_GRID[-1], FEE_GRID[-1] * 2.0
    st_lo = eval_fee(lo)
    res["breakeven"]["bisection_rows"].append({"fee": lo, **st_lo})
    dump(res)
    k = 0
    while not (eval_fee(hi)["sharpe"] < FAIL_SHARPE <= st_lo["sharpe"]) and k < 30:
        # if lo already fails, step lo down toward grid; else double hi
        if st_lo["sharpe"] < FAIL_SHARPE:
            hi = lo
            lo = lo / 2.0
            st_lo = eval_fee(lo)
            res["breakeven"]["bisection_rows"].append({"fee": lo, **st_lo})
            dump(res)
        else:
            hi *= 2.0
        k += 1
    st_hi = eval_fee(hi)
    assert st_lo["sharpe"] >= FAIL_SHARPE > st_hi["sharpe"], "bracket failed: lo=%.5f sh=%.3f hi=%.5f sh=%.3f" % (lo, st_lo["sharpe"], hi, st_hi["sharpe"])
    log("bracket lo=%.5f sh=%.3f hi=%.5f sh=%.3f thresh=%.2f" % (lo, st_lo["sharpe"], hi, st_hi["sharpe"], FAIL_SHARPE))
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        st_m = eval_fee(mid)
        res["breakeven"]["bisection_rows"].append({"fee": mid, **st_m})
        dump(res)
        if st_m["sharpe"] > FAIL_SHARPE:
            lo = mid
            st_lo = st_m
        else:
            hi = mid
    be_fee = 0.5 * (lo + hi)
    st_z = eval_fee(be_fee)
    log("break-even fee=%.6f sharpe=%.3f (width=%.2e)" % (be_fee, st_z["sharpe"], hi - lo))
    res["breakeven"]["bracket_lo"] = lo
    res["breakeven"]["bracket_hi"] = hi
    res["breakeven"]["threshold"] = FAIL_SHARPE
    res["breakeven"]["breakeven_fee"] = round(be_fee, 6)
    res["breakeven"]["breakeven_check"] = st_z
    res["breakeven"]["margin_from_grid_hi"] = round(be_fee - FEE_GRID[-1], 6)
    dump(res)

    res["verdict"] = "PENDING"
    res["verdict_note"] = "PENDING (\u5f85\u5b9a): P0-3 permutation FAILED, no adoption, no live change; fee sweep is diagnostic only."
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s (diagnostic only)" % (OUT, res["verdict"]))

if __name__ == "__main__":
    main()
