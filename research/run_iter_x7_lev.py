"""X7 lev scan: Top5 equal-weight FULL sharpe/mdd/final at lev {1,2,3}.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + roll1, static equal 0.2 weights, aster perp,
fee=FEE fund=FUND. Sweep leverage only; read-only, LEV untouched.
Output: results/iter_X7_lev.json (verdict PENDING, decision KEEP_LEV_2.0).
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
BPY = 2190.0
OUT = pathlib.Path("results/iter_X7_lev.json")
LOG = pathlib.Path("logs/iter_X7_lev.log")

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15(c):
    rows = list(csv.DictReader(open("data/data_15m_3y/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common4h(coins):
    raw = {c: load15(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; closes = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        b4 = []
        for i in range(len(rr) // 16):
            blk = rr[i * 16:(i + 1) * 16]
            b4.append((blk[0][1], max(r[2] for r in blk), min(r[3] for r in blk), blk[-1][4], sum(r[5] for r in blk)))
        bars[c] = b4; closes[c] = [b[3] for b in b4]
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

def leg_net(raw, rt, sig, spec, fee, fund, lev):
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=spec["cd"], bars_per_year=BPY, stop_loss=spec["sl"], time_stop=spec["ts"], vol_target=spec["vt"], vol_window=spec["vw"])
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
    cs = pk = md = 0.0; pk0 = -1e18
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_X7_lev start\n")
    bars, closes = common4h(COINS)
    n = len(bars["ETC"])
    log("common 4h n=%d levs=%s" % (n, LEVS))
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
    sh = {k: rows[k]["FULL"]["sharpe"] for k in rows}
    dd = {k: rows[k]["FULL"]["mdd"] for k in rows}
    fx = {k: rows[k]["FULL"]["final_x"] for k in rows}
    sharpe_spread = round(max(sh.values()) - min(sh.values()), 4)
    lev2_optimal = bool(sh["2"] >= sh["1"] and sh["2"] >= sh["3"] and dd["2"] < dd["3"])
    dd_uncontrolled_3x = bool(dd["3"] > 1.0 or dd["3"] > 1.5 * dd["2"])
    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + roll1; static equal 0.2 Top5", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "levs": list(LEVS), "venue": "aster", "lev_locked": LEV, "fund": FUND, "fee": FEE, "grid_bars": n, "note": "LEV sweep only, read-only; live LEV untouched (stays 2.0)"},
        "rows": rows,
        "compare": {"sharpe_by_lev": sh, "mdd_by_lev": dd, "final_x_by_lev": fx, "sharpe_spread": sharpe_spread, "sharpe_invariant": bool(sharpe_spread < 0.01), "dd_2v1": round(dd["2"] / dd["1"], 3) if dd["1"] > 0 else None, "dd_3v2": round(dd["3"] / dd["2"], 3) if dd["2"] > 0 else None, "lev2_optimal_sharpe": lev2_optimal, "dd_3x_uncontrolled": dd_uncontrolled_3x},
        "verdict": "PENDING_KEEP_LEV2",
        "decision": "KEEP_LEV_2.0",
        "decision_note": "\u5f85\u5b9a\uff0c\u4e0d\u52d5LEV\u3002sharpe\u5728{1,2,3}x\u4e0b\u4e0d\u8b8a(\u7dda\u6027\u7e2e\u653e)\uff0c\u6536\u76ca/dd\u540c\u6bd4\u4f3c\u653e\u5927\u3002"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s decision=%s" % (OUT, res["verdict"], res["decision"]))

if __name__ == "__main__":
    main()
