"""W7 15m entry-timing delay (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change (FEE/FUND/LEV untouched).

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll(1+D). Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

15m native: cd/ts/vw x16, BPY=35040. Data data/data_1y/15m/{COIN}.csv
(~35040 rows: timestamp,open,high,low,close,volume,quote_volume,trades).

Sweep: execution delay D in {0,1,2,3} bars after signal -> FULL
sharpe/turnover (+ann/mdd/cum/final_x/n) plus per-coin sharpe per delay.
D=0 reproduces leg_net exactly (roll1 + zero-first-bar + wrap-artifact
turnover, mirrored deliberately). Pre-roll lp/sp are precomputed once per
coin (cooldown+stops+vol_scale are delay-independent); delay cells are pure
arithmetic: lp_D = lp_pre.roll(1+D) with first 1+D bars zeroed.

IMPORTANT: results/iter_W7_delay.json is dumped after EACH delay unit
(incremental; "status": "partial" until the final write), so a killed run
still leaves partial rows behind.

Outputs: results/iter_W7_delay.json (+ logs/iter_W7_delay.log).
Offline read-only: reads data/data_1y/15m/*.csv only. No orders.

Smoke mode (for tests): ITER_W7_SMOKE=1 shrinks to coins {ETC,TRX} and
the first 3000 bars. ITER_W7_OUT / ITER_W7_LOG override output paths
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
DELAYS = [0, 1, 2, 3]
FEE_FIXED = FEE
FUND_FIXED = FUND
LEV_USE = 2.0
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path(os.getenv("ITER_W7_OUT", "results/iter_W7_delay.json"))
LOG = pathlib.Path(os.getenv("ITER_W7_LOG", "logs/iter_W7_delay.log"))
SMOKE = os.getenv("ITER_W7_SMOKE") == "1"

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

def pre_series(raw, rt, sig, spec):
    """Pre-roll executed intent (delay-independent): cooldown+stops+vol_scale."""
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=0.0, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    return lp, sp

def apply_delay(lp_pre, sp_pre, rt, delay):
    """Mirror leg_net execution exactly: roll(1+D), zero first 1+D, wrap-artifact turnover."""
    lp = lp_pre.roll(1 + delay, dims=1); lp[:, :1 + delay] = 0
    sp = sp_pre.roll(1 + delay, dims=1); sp[:, :1 + delay] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    pos = lp - sp
    gross = pos * rt * LEV_USE
    return pos[0].tolist(), turn[0].tolist(), gross[0].tolist()

def sharpe_of(s):
    n = len(s)
    if n < 2:
        return 0.0
    m = sum(s) / n
    v = sum((x - m) ** 2 for x in s) / (n - 1)
    return m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0

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
    open(LOG, "w").write("iter_W7_delay start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, n = common15m(coins)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d fee=%.4f fund=%.4f delays=%s" % (n, coins, SMOKE, SCALE, FEE_FIXED, FUND_FIXED, DELAYS))
    assert n > 2000, n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    pres = {}
    for c in coins:
        raw, rt, sg = mats[c]
        pres[c] = (pre_series(raw, rt, sg, SPECS[c]), rt)
    log("pre-roll intent series precomputed")

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll(1+D); static equal Top5; 15m native cd/ts/vw x16; pre-roll intent precomputed once, delay cells arithmetic", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "delays": list(DELAYS), "fee": FEE_FIXED, "fund": FUND_FIXED, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "smoke": SMOKE, "note": "entry-timing delay sweep only, read-only; live FEE/FUND/LEV untouched"},
        "status": "partial", "curve": {"rows": []},
        "verdict": "PENDING"}
    dump(res)

    for d in DELAYS:
        P = {}; T = {}; G = {}
        for c in coins:
            (lp_pre, sp_pre), rt = pres[c]
            P[c], T[c], G[c] = apply_delay(lp_pre, sp_pre, rt, d)
        net = [sum((G[c][t] - T[c][t] * FEE_FIXED * LEV_USE - P[c][t] * FUND_FIXED * LEV_USE) for c in coins) * (1.0 / len(coins)) for t in range(n)]
        turn = [sum(T[c][t] for c in coins) * (1.0 / len(coins)) for t in range(n)]
        st = seg(net, turn, 0, n)
        per_coin = {}
        for c in coins:
            cn = [G[c][t] - T[c][t] * FEE_FIXED * LEV_USE - P[c][t] * FUND_FIXED * LEV_USE for t in range(n)]
            per_coin[c] = round(sharpe_of(cn), 3)
        row = {"delay": d, **st, "per_coin_sharpe": per_coin}
        res["curve"]["rows"].append(row)
        dump(res)
        log("delay %d FULL sh=%.3f final_x=%.4f mdd=%.4f cum=%.4f to=%.6f per=%s" % (d, st["sharpe"], st["final_x"], st["mdd"], st["cum"], st["turnover"], per_coin))

    rows = res["curve"]["rows"]
    sh = [r["sharpe"] for r in rows]
    fx = [r["final_x"] for r in rows]
    mono = all(sh[i + 1] <= sh[i] + 1e-9 for i in range(len(sh) - 1))
    res["curve"]["monotone_down"] = bool(mono)
    res["curve"]["sharpe_slope_per_bar"] = round(lin_slope([float(r["delay"]) for r in rows], sh), 5)
    res["curve"]["sharpe_drop_0_to_3"] = round(sh[0] - sh[-1], 3)
    res["curve"]["final_x_monotone_down"] = bool(all(fx[i + 1] <= fx[i] + 1e-9 for i in range(len(fx) - 1)))
    res["curve"]["turnover_by_delay"] = [r["turnover"] for r in rows]
    dump(res)

    res["verdict"] = "PENDING"
    res["verdict_note"] = "PENDING (\u5f85\u5b9a): P0-3 permutation FAILED, no adoption, no live change; delay sweep is diagnostic only."
    res["decision"] = "NO_CHANGE"
    res["decision_note"] = "\u5f85\u5b9a,\u4e0d\u52a8FEE/FUND/LEV\u300215m\u539f\u751f\u683c\u5b50\u3002"
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s (diagnostic only)" % (OUT, res["verdict"]))

if __name__ == "__main__":
    main()
