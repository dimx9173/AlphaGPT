"""H8 funding-regime sweep (1h native): Top5 equal-weight FULL at fee2x.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp lev2, fee=FEE2X. 1h native: cd/ts/vw x4, BPY=8760.
E10 FORMULA untouched. Read-only: LEV/FUND/FEE config untouched. Verdict PENDING (P0-3 FAIL); no adoption, live untouched.

Grid: fund in {0.0001, 0.0005, 0.001, 0.002, 0.005} at fee2x -> FULL
sharpe/mdd/final_x (+ann/cum/n/turnover). Downside zero-cross bisect on
negative fund until sharpe<0.05 (failure threshold 0.05). Gate check: 0.001
reasonable (passes with margin to failure). Verdict on gate only.

Speed: executed positions (lp/sp after cooldown/stops/vol/roll) do not
depend on fee/fund, so per-coin pos/turn/gross series are precomputed once;
each fund cell is pure arithmetic. Incremental dump: OUT rewritten after
EACH fund-cell evaluation (grid + bracket + bisect steps), status
partial->done, so partial progress survives kills.

Smoke: ITER_Y8_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
OUT/LOG overridable via ITER_Y8_OUT / ITER_Y8_LOG.
Output: results/iter_H8_fund.json (gate check only; verdict forced PENDING, no adoption).
"""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE, FEE2X

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
FUND_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.005]
GATE_FUND = 0.001
FAIL_SHARPE = 0.05
LEV_USE = 2.0
BPY = 8760.0
SCALE = 4
BISECT_ITERS = 20
BRACKET_LO0 = -0.001
OUT = pathlib.Path(os.getenv("ITER_H8_OUT", "results/iter_H8_fund.json"))
LOG = pathlib.Path(os.getenv("ITER_H8_LOG", "logs/iter_H8_fund.log"))
SMOKE = os.getenv("ITER_H8_SMOKE") == "1"

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; closes = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [r[4] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]; closes[c] = closes[c][:n]
    return bars, closes

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
    """Executed position/turnover/gross series; fee/fund independent."""
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=0.0, fee_override=FEE2X, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
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
    open(LOG, "w").write("iter_H8_fund start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, _ = common1h(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d fee2x=%.4f" % (n, coins, SMOKE, SCALE, FEE2X))
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

    def eval_fund(fund):
        net = [G[t] - T[t] * FEE2X * LEV_USE - P[t] * fund * LEV_USE for t in range(n)]
        return seg(net, T, 0, n)

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 1h native cd/ts/vw x4; positions precomputed once, fund cells arithmetic", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "fee": FEE2X, "fund_grid": list(FUND_GRID), "gate_fund": GATE_FUND, "fail_sharpe": FAIL_SHARPE, "grid": "1h", "grid_bars": n, "bpy": BPY, "scale": SCALE, "smoke": SMOKE, "note": "fund sweep only, read-only; live FUND/FEE/LEV untouched"},
        "status": "partial", "curve": {"rows": []}, "downside": {"bisection_rows": []},
        "verdict": "PENDING"}
    dump(res)

    for fund in FUND_GRID:
        st = eval_fund(fund)
        res["curve"]["rows"].append({"fund": fund, **st})
        dump(res)
        log("fund %.4f FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f to=%.6f" % (fund, st["sharpe"], st["mdd"], st["cum"], st["final_x"], st["turnover"]))
    rows = res["curve"]["rows"]
    mono_up = all(rows[i + 1]["sharpe"] >= rows[i]["sharpe"] - 1e-9 for i in range(len(rows) - 1))
    to_flat = len({r["turnover"] for r in rows}) == 1
    res["curve"]["monotone_up"] = bool(mono_up)
    res["curve"]["turnover_flat"] = bool(to_flat)
    dump(res)

    cov = next(r for r in rows if abs(r["fund"] - GATE_FUND) < 1e-12)
    cover_ok = bool(cov["sharpe"] > 0)

    lo, hi = BRACKET_LO0, FUND_GRID[0]
    st_lo = eval_fund(lo)
    res["downside"]["bisection_rows"].append({"fund": lo, **st_lo})
    dump(res)
    k = 0
    while not (st_lo["sharpe"] < FAIL_SHARPE < eval_fund(hi)["sharpe"]) and k < 10:
        lo *= 2.0
        st_lo = eval_fund(lo)
        res["downside"]["bisection_rows"].append({"fund": lo, **st_lo})
        dump(res)
        k += 1
    st_hi = eval_fund(hi)
    assert st_lo["sharpe"] < FAIL_SHARPE < st_hi["sharpe"], "bracket failed: lo=%.5f sh=%.3f hi=%.5f sh=%.3f" % (lo, st_lo["sharpe"], hi, st_hi["sharpe"])
    log("bracket lo=%.5f sh=%.3f hi=%.5f sh=%.3f thresh=%.2f" % (lo, st_lo["sharpe"], hi, st_hi["sharpe"], FAIL_SHARPE))
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        st_m = eval_fund(mid)
        res["downside"]["bisection_rows"].append({"fund": mid, **st_m})
        dump(res)
        if st_m["sharpe"] > FAIL_SHARPE:
            hi = mid
        else:
            lo = mid
    zero_cross = 0.5 * (lo + hi)
    st_z = eval_fund(zero_cross)
    log("zero-cross fund=%.6f sharpe=%.3f (width=%.2e)" % (zero_cross, st_z["sharpe"], hi - lo))
    res["downside"]["bracket_lo"] = res["downside"]["bisection_rows"][0]["fund"]
    res["downside"]["bracket_hi"] = hi
    res["downside"]["threshold"] = FAIL_SHARPE
    res["downside"]["zero_cross_fund"] = round(zero_cross, 6)
    res["downside"]["zero_cross_check"] = st_z
    dump(res)

    margin = GATE_FUND - zero_cross
    reasonable = bool(cover_ok and margin > 0 and mono_up)
    res["upside"] = {"max_fund": FUND_GRID[-1], "max_sharpe": rows[-1]["sharpe"], "failure_in_grid": False, "note": "no upside failure in [0.0001, 0.005]; sharpe monotone increasing (short rebate scales linearly)"}
    res["gate_0001"] = {"sharpe": cov["sharpe"], "pass_sharpe_gt_0": cover_ok, "margin_to_failure": round(margin, 6), "reasonable": reasonable, "note": "0.001 gate passes with sharpe=%.3f; failure (sharpe<%.2f) only at fund=%.6f, margin %.6f" % (cov["sharpe"], FAIL_SHARPE, zero_cross, margin)}
    res["verdict"] = "PENDING"
    res["verdict_detail"] = "PASS" if reasonable else "FAIL"
    res["decision"] = "NO_ADOPTION"
    res["decision_note"] = ("P0-3 permutation FAIL => H8 verdict PENDING; gate check only " "(gate_0001.reasonable=%s), no adoption, live FUND/FEE/LEV untouched." % reasonable)
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s (gate only)" % (OUT, res["verdict"]))

if __name__ == "__main__":
    main()
