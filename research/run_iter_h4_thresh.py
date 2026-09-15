"""H4 thresh robustness (1h native, Top5) -- ROBUSTNESS ONLY, DIAGNOSTIC.

Mirror research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1.

Top5 locked specs: ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3. Equal 0.2 weights.

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows).
Per-coin lth/sth +/-0.02 (20 runs), FULL sharpe delta vs base;
flat iff max|d|<=0.15. No optimum pursuit, no adoption, live untouched.

Output: results/iter_H4_thresh.json (+ logs/iter_h4_thresh.log).
Incremental dump after each unit: partial JSON survives interrupts.
Offline read-only. Verdict PENDING (P0-3 FAIL).

Smoke mode (for tests): ITER_H4_SMOKE=1 shrinks to ETC sth +/-0.02
only (base + 2 runs). ITER_H4_OUT / ITER_H4_LOG override output paths
(tests use a temp file so the committed FULL artifact is not clobbered).
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
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units
DELTA = 0.02
FLAT_TOL = 0.15
PARAMS = ("lth", "sth")
SGNS = (+0.02, -0.02)
OUT = pathlib.Path(os.getenv("ITER_H4_OUT", "results/iter_H4_thresh.json"))
LOG = pathlib.Path(os.getenv("ITER_H4_LOG", "logs/iter_h4_thresh.log"))
SMOKE = os.getenv("ITER_H4_SMOKE") == "1"
SMOKE_COINS = ["ETC"]
SMOKE_PARAMS = ("sth",)

_partial = {}

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def dump():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_partial, indent=1, ensure_ascii=False))

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

def leg_net(raw, rt, sig, spec, fee, fund):
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist()

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def eval_basket(mats, n, specs):
    legs = {}; turns = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs[c], turns[c] = leg_net(raw, rt, sg, specs[c], FEE, FUND)
    net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    return net, turn, seg(net, turn, 0, n)

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H4_thresh start smoke=%s\n" % SMOKE)
    _partial["config"] = {
        "engine": "mirror run_weight_modes.py leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1",
        "formula": list(FORMULA),
        "basket": {c: {"lth": SPECS[c]["lth"], "sth": SPECS[c]["sth"], "cd": SPECS[c]["cd"], "sl": SPECS[c]["sl"], "ts": SPECS[c]["ts"], "q": SPECS[c]["q"]} for c in COINS},
        "weights": {c: W[c] for c in COINS},
        "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE,
        "delta": DELTA, "params": list(SMOKE_PARAMS if SMOKE else PARAMS),
        "flat_tol": FLAT_TOL,
        "flat_rule": "max|d FULL sharpe| <= %.2f => flat(\u7a69\u5065), else sensitive(\u654f\u611f); robustness only, no adoption" % FLAT_TOL,
        "smoke": SMOKE,
        "grid": "1h",
        "bpy": BPY,
        "scale": SCALE,
        "note": "E10 FORMULA untouched; live chain untouched; offline read-only; PENDING (\u5f85\u5b9a), not demo evidence",
    }
    _partial["rows"] = []
    _partial["verdict"] = "PENDING"
    _partial["decision"] = "KEEP_LOCKED"
    dump()
    bars, closes = common1h(COINS)
    n = len(bars["ETC"])
    log("common 1h native n=%d smoke=%s scale=x%d" % (n, SMOKE, SCALE))
    assert n > 2000, "grid too short: %d" % n
    _partial["config"]["grid_bars"] = n
    dump()
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")
    _partial.setdefault("units_done", []).append("signals")
    dump()
    base_specs = {c: dict(SPECS[c]) for c in COINS}
    _, _, base_full = eval_basket(mats, n, base_specs)
    base_sh = base_full["sharpe"]
    _partial["base_FULL"] = base_full
    _partial["units_done"].append("base")
    dump()
    log("base FULL sharpe=%.3f turnover=%.6f mdd=%.4f cum=%.4f" % (base_sh, base_full["turnover"], base_full["mdd"], base_full["cum"]))
    coins = SMOKE_COINS if SMOKE else list(COINS)
    params = SMOKE_PARAMS if SMOKE else PARAMS
    rows = []
    for coin in coins:
        for param in params:
            for sgn in SGNS:
                specs = {c: dict(base_specs[c]) for c in COINS}
                new_v = round(specs[coin][param] + sgn, 4)
                specs[coin][param] = new_v
                _, _, full = eval_basket(mats, n, specs)
                d = round(full["sharpe"] - base_sh, 4)
                rows.append({"coin": coin, "param": param, "delta": sgn, "new_value": new_v, "FULL": full, "d_sharpe": d})
                _partial["units_done"].append("%s_%s_%+.2f" % (coin, param, sgn))
                log("%s %s%+.2f -> %.2f FULL sharpe=%.3f d=%+.4f to=%.6f mdd=%.4f" % (coin, param, sgn, new_v, full["sharpe"], d, full["turnover"], full["mdd"]))
                order = sorted(range(len(rows)), key=lambda i: abs(rows[i]["d_sharpe"]), reverse=True)
                _partial["rows"] = [rows[i] for i in order]
                dump()
    rows = _partial["rows"]
    max_abs = abs(rows[0]["d_sharpe"]) if rows else 0.0
    lth_max = max([abs(r["d_sharpe"]) for r in rows if r["param"] == "lth"] or [0.0])
    sth_max = max([abs(r["d_sharpe"]) for r in rows if r["param"] == "sth"] or [0.0])
    flat = bool(max_abs <= FLAT_TOL)
    flat_verdict = "flat(\u7a69\u5065)" if flat else "sensitive(\u654f\u611f)"
    log("max|d|=%.4f (lth %.4f / sth %.4f) tol=%.2f => %s" % (max_abs, lth_max, sth_max, FLAT_TOL, flat_verdict))
    _partial["flat_check"] = {"flat_verdict": flat_verdict, "flat": flat, "max_abs_d_sharpe": round(max_abs, 4), "lth_max_abs_d": round(lth_max, 4), "sth_max_abs_d": round(sth_max, 4), "tol": FLAT_TOL, "n_runs": len(rows)}
    _partial["verdict"] = "PENDING"
    _partial["decision"] = "KEEP_LOCKED"
    _partial["conclusion"] = "\u5f85\u5b9a: H4 1h per-coin lth/sth +/-0.02 %s, max|d sharpe|=%.4f (tol %.2f); no optimum pursuit, no adoption" % (flat_verdict, max_abs, FLAT_TOL)
    dump()
    log("wrote %s conclusion=%s" % (OUT, _partial["conclusion"]))

if __name__ == "__main__":
    main()
