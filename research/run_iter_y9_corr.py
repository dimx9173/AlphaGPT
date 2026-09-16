"""Y9 15m correlation RESUME (Top5). E10 FORMULA leg nets, rolling-960-bar pairwise corr.
Engine mirrors research/run_weight_modes.py leg_net (StackVM+FeatureEngineer, MemeBacktest
venue=aster lev2 short_enabled fund0.0005, quantile q0.3 long-only + cooldown + stops +
vol_scale(vt None->1.0) + roll1). 15m native: cd/ts/vw x16, BPY=35040. Offline read-only.
Verdict PENDING, no live change. Brake rule default OFF (Y1B_CORR_BRAKE)."""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import (FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC,
    LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE)

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
X = 16  # 4h -> 15m native scale for cd/ts/vw
SPECS = {}
for _c in COINS:
    _b = BASE[_c]
    SPECS[_c] = dict(lth=_b["lth"], sth=_b["sth"], cd=_b["cd"] * X, sl=_b["sl"],
                     ts=_b["ts"] * X, vt=_b["vt"], vw=_b["vw"] * X, q=_b["q"])
W = {c: 0.2 for c in COINS}
BPY = 35040.0
WIN = 960
OUT = pathlib.Path("results/iter_Y9_corr.json")
LOG = pathlib.Path("logs/iter_Y9_corr.log")

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"])) for r in rows]

def common15(coins):
    raw = {c: load15(c) for c in coins}
    ts = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts &= set(r[0] for r in raw[c])
    ts = sorted(ts)
    bars = {}
    for c in coins:
        d = {r[0]: r[1:] for r in raw[c]}
        bars[c] = [(t,) + d[t] for t in ts]
    return bars

def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[1] for b in bars]]), "high": torch.tensor([[b[2] for b in bars]]),
           "low": torch.tensor([[b[3] for b in bars]]), "close": torch.tensor([[b[4] for b in bars]]),
           "volume": torch.tensor([[b[5] for b in bars]]), "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][4] - bars[i][4]) / bars[i][4] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_net(raw, rt, sig, spec):
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE, long_th=spec["lth"],
                      short_th=spec["sth"], cooldown_bars=spec["cd"], bars_per_year=BPY,
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
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist()

def leg_stats(s):
    n = len(s)
    m = sum(s) / n
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1)
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x; pk = max(pk, cs); md = max(md, pk - cs)
    return {"n": n, "cum": round(sum(s), 4), "sharpe": round(sh, 3),
            "ann": round(m * BPY, 4), "mdd": round(md, 4)}

def roll_corr(a, b, w):
    import numpy as np
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = len(a); m = n - w + 1
    ca = np.concatenate([[0.0], np.cumsum(a)]); cb = np.concatenate([[0.0], np.cumsum(b)])
    cab = np.concatenate([[0.0], np.cumsum(a * b)])
    ca2 = np.concatenate([[0.0], np.cumsum(a * a)]); cb2 = np.concatenate([[0.0], np.cumsum(b * b)])
    sx = ca[w:] - ca[:-w]; sy = cb[w:] - cb[:-w]
    sxy = cab[w:] - cab[:-w]; sx2 = ca2[w:] - ca2[:-w]; sy2 = cb2[w:] - cb2[:-w]
    cov = (sxy - sx * sy / w) / w
    vx = (sx2 - sx * sx / w) / w; vy = (sy2 - sy * sy / w) / w
    den = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
    out = np.zeros(m)
    ok = den > 1e-12
    out[ok] = cov[ok] / den[ok]
    return [float(v) for v in out]

def q(vals, p):
    s = sorted(vals); n = len(s)
    if n == 0:
        return 0.0
    k = (n - 1) * p
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("Y9 corr start\n")
    bars = common15(COINS)
    n = len(bars["ETC"])
    log("common 15m n=%d" % n)
    legs = {}
    for c in COINS:
        raw, rt, sg = build_sig(bars[c])
        legs[c] = leg_net(raw, rt, sg, SPECS[c])
        log("%s cum=%.4f" % (c, sum(legs[c])))
    cfg = {"engine": "mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1",
           "formula": list(FORMULA), "coins": COINS,
           "specs_4h": {c: dict(BASE[c]) for c in COINS},
           "specs_15m": {c: dict(SPECS[c]) for c in COINS},
           "scale_15m": X, "weights": dict(W), "grid": "15m", "grid_bars": n,
           "bpy": BPY, "window": WIN, "venue": "aster", "lev": LEV, "fee": FEE, "fund": FUND,
           "data": "data/data_1y/15m/{COIN}.csv",
           "brake_env": "Y1B_CORR_BRAKE", "brake_default": "OFF",
           "note": "E10 FORMULA untouched; equal 0.2 weights (corr scale-invariant); offline read-only"}
    eng = {"config": cfg, "legs": {c: leg_stats(legs[c]) for c in COINS},
           "engine_done": True, "stats": None, "verdict": "PENDING"}
    OUT.write_text(json.dumps(eng, indent=1, ensure_ascii=False))
    log("wrote engine checkpoint %s" % OUT)
    pairs = {}
    for i in range(len(COINS)):
        for j in range(i + 1, len(COINS)):
            key = "%s-%s" % (COINS[i], COINS[j])
            rc = roll_corr(legs[COINS[i]], legs[COINS[j]], WIN)
            pairs[key] = {"max": round(max(rc), 4), "mean": round(sum(rc) / len(rc), 4),
                          "q50": round(q(rc, 0.5), 4), "q95": round(q(rc, 0.95), 4), "n_win": len(rc)}
            log("%s max=%.4f mean=%.4f q50=%.4f q95=%.4f" % (key, pairs[key]["max"], pairs[key]["mean"], pairs[key]["q50"], pairs[key]["q95"]))
    mx = [v["max"] for v in pairs.values()]
    mn = [v["mean"] for v in pairs.values()]
    stats = {"pairs": pairs, "n_pairs": len(pairs), "window": WIN,
             "avg_max": round(sum(mx) / len(mx), 4),
             "share_max_gt_0.6": round(sum(1 for v in mx if v > 0.6) / len(mx), 4),
             "share_mean_gt_0.6": round(sum(1 for v in mn if v > 0.6) / len(mn), 4),
             "share_max_gt_0.7": round(sum(1 for v in mx if v > 0.7) / len(mx), 4),
             "share_max_gt_0.85": round(sum(1 for v in mx if v > 0.85) / len(mx), 4)}
    if stats["share_max_gt_0.85"] > 0:
        div = "CONCENTRATED"
    elif stats["share_max_gt_0.7"] > 0:
        div = "WATCH"
    elif stats["avg_max"] < 0.6:
        div = "DIVERSIFIED"
    else:
        div = "WATCH"
    brake_on = os.getenv("Y1B_CORR_BRAKE", "OFF") not in ("OFF", "", "0", "false", "False")
    res = {"config": cfg, "legs": {c: leg_stats(legs[c]) for c in COINS}, "engine_done": True,
           "stats": stats,
           "diversification": {"verdict": div, "rule": "CONCENTRATED if any max>0.85; WATCH if any max>0.7; DIVERSIFIED if avg_max<0.6"},
           "brake": {"enabled": bool(brake_on), "env": "Y1B_CORR_BRAKE",
                     "rule": "rolling-960 max>0.7 halve joint weight, max>0.85 quarter; default OFF",
                     "halve_triggers": sorted([k for k, v in pairs.items() if v["max"] > 0.7]),
                     "quarter_triggers": sorted([k for k, v in pairs.items() if v["max"] > 0.85])},
           "verdict": "PENDING", "decision": "KEEP_NO_LIVE_CHANGE",
           "decision_note": "Correlation study only; brake default OFF; no live weight change."}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s avg_max=%.4f div=%s verdict=PENDING" % (OUT, stats["avg_max"], div))

if __name__ == "__main__":
    main()
