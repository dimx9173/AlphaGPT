"""H7 lev scan (1h native): Top5 equal-weight FULL sharpe/mdd/final_x/turnover at lev {1,2,3}.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp, fee=FEE fund=FUND. 1h native: cd/ts/vw x4, BPY=8760.
Sweep leverage only; read-only, LEV untouched.
Output: results/iter_H7_lev.json (verdict PENDING (P0-3 FAIL), decision KEEP_LEV_2.0).
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
LEVS = [1, 2, 3]
BPY = 8760.0
SCALE = 4
OUT = pathlib.Path("results/iter_H7_lev.json")
LOG = pathlib.Path("logs/iter_H7_lev.log")

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

def leg_net(raw, rt, sig, spec, fee, fund, lev):
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
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

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H7_lev start\n")
    bars, closes = common1h(COINS)
    n = len(bars["ETC"])
    log("common 1h native n=%d levs=%s scale=x%d" % (n, LEVS, SCALE))
    mats = {c: build_sig(bars[c]) for c in COINS}
    rows = {}
    for lev in LEVS:
        legs = {}; turns = {}
        for c in COINS:
            raw, rt, sg = mats[c]
            legs[c], turns[c] = leg_net(raw, rt, sg, SPECS[c], FEE, FUND, lev)
        net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
        turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
        full = seg(net, turn, 0, n)
        rows[str(lev)] = {"FULL": full}
        log("lev=%d FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f to=%.6f" % (lev, full["sharpe"], full["mdd"], full["cum"], full["final_x"], full["turnover"]))
        OUT.write_text(json.dumps({"config": {"levs": list(LEVS), "grid": "1h", "grid_bars": n, "bpy": BPY, "partial": True}, "rows": dict(rows), "verdict": "PENDING", "decision": "KEEP_LEV_2.0"}, indent=1, ensure_ascii=False))
    sh = {k: rows[k]["FULL"]["sharpe"] for k in rows}
    dd = {k: rows[k]["FULL"]["mdd"] for k in rows}
    fx = {k: rows[k]["FULL"]["final_x"] for k in rows}
    sharpe_spread = round(max(sh.values()) - min(sh.values()), 4)
    dd_uncontrolled_3x = bool(dd["3"] > 1.0 or (dd["2"] > 0 and dd["3"] > 1.5 * dd["2"]))
    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5; 1h native cd/ts/vw x4", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "levs": list(LEVS), "venue": "aster", "lev_locked": LEV, "fund": FUND, "fee": FEE, "grid": "1h", "grid_bars": n, "bpy": BPY, "scale": SCALE, "note": "LEV sweep only, read-only; live LEV untouched (stays 2.0)"},
        "rows": rows,
        "compare": {"sharpe_by_lev": sh, "mdd_by_lev": dd, "final_x_by_lev": fx, "sharpe_spread": sharpe_spread, "sharpe_invariant": bool(sharpe_spread < 0.01), "dd_2v1": round(dd["2"] / dd["1"], 3) if dd["1"] > 0 else None, "dd_3v2": round(dd["3"] / dd["2"], 3) if dd["2"] > 0 else None, "dd_3x_uncontrolled": dd_uncontrolled_3x},
        "verdict": "PENDING",
        "decision": "KEEP_LEV_2.0",
        "decision_note": "P0-3 FAIL\u5f85\u5b9a\uff0c\u4e0d\u52d5LEV\u3002sharpe\u5728{1,2,3}x\u4e0b\u4e0d\u8b8a(\u7dda\u6027\u7e2e\u653e)\uff0c\u6536\u76ca/dd\u540c\u6bd4\u4f3c\u653e\u5927\u30021h\u539f\u751f\u683c\u5b50\u3002"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s decision=%s" % (OUT, res["verdict"], res["decision"]))

if __name__ == "__main__":
    main()
