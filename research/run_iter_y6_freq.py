"""Y6 15m frequency experiment (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED => verdict PENDING, no live change.
15m-native decision vs 1h-thinned decision (every 4th bar signal, same engine).

Engine mirrors research/run_weight_modes.py leg_net + quantile q0.3
long-only + cooldown + stops + vol_scale(vt None->1.0). 15m NATIVE grid
(data/data_1y/15m, ~35040 bars): cd/ts/vw x16, BPY=35040. Equal 0.2
weights. Top5 locked specs ETC(0.88/0.12/cd18/None/ts24)
TRX(0.85/0.12/cd6/0.05/ts24) ATOM(0.85/0.15/cd6/0.05/ts24)
APT(0.88/0.12/cd18/None/ts24) KAS(0.88/0.12/cd6/None/ts24), q0.3.

Arms (same desired-position engine, decision cadence differs):
  native_15m : adopt desired position every 15m bar (+1 execution lag).
  thinned_1h : sample-and-hold every 4th bar signal (+1 execution lag),
               mirrors X6 aligned_4h gate semantics on the 15m grid.

Per arm: FULL sharpe/turnover/flips/entries + fee2x gap + H2 (2nd half).
Recommendation: Y1B_DECISION_ALIGN shelved (executor default stays unset).

Offline read-only: reads data/data_1y/15m/*.csv only. No orders, no env switches.
Smoke (for tests): ITER_Y6_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
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
BPY = 35040.0
SCALE = 16
THIN_EVERY = 4
OUT = pathlib.Path(os.getenv("ITER_Y6_OUT", "results/iter_Y6_freq.json"))
LOG = pathlib.Path(os.getenv("ITER_Y6_LOG", "logs/iter_y6_freq.log"))
SMOKE = os.getenv("ITER_Y6_SMOKE") == "1"

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common15m(coins):
    raw = {c: load15m(c) for c in coins}
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

def desired_pos(raw, rt, sig, spec):
    """Desired {1,-1,0} position per 15m bar, NO execution lag (roll applied at gate)."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    pos = (lp - sp)[0].tolist()
    return [1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0) for v in pos]

def apply_gate(d, freq):
    """Decision cadence + 1-bar execution lag. thinned_1h holds every 4th bar signal."""
    n = len(d)
    if freq == "native_15m":
        adopted = list(d)
    elif freq == "thinned_1h":
        adopted = []
        cur = 0.0
        for t in range(n):
            if t % THIN_EVERY == 0:
                cur = d[t]
            adopted.append(cur)
    else:
        raise ValueError(freq)
    return [0.0] + adopted[:-1]

def pnl_from_pos(pos, rets, fee, fund):
    net, turn, flips, entries = [], [], 0, 0
    prev = 0.0
    for t in range(len(pos)):
        pp = pos[t]; r = rets[t]; dp = pp - prev
        turn.append(abs(dp))
        net.append(pp * r * LEV - abs(dp) * fee * LEV - pp * fund * LEV)
        if pp != 0.0 and prev == 0.0:
            entries += 1
        if pp != 0.0 and prev != 0.0 and (pp > 0) != (prev > 0):
            flips += 1
        prev = pp
    return net, turn, flips, entries

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    return {"sharpe": round(sh, 3), "ann": round(sum(s) / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(sum(s), 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def eval_arm(coins, des, rets, n, h2a, fee, fund, freq):
    legs_net, legs_turn, per_coin = [], [], {}
    for c in coins:
        pos = apply_gate(des[c], freq)
        net, turn, flips, entries = pnl_from_pos(pos, rets[c], fee, fund)
        per_coin[c] = {"FULL": seg(net, turn, 0, n), "H2": seg(net, turn, h2a, n), "flips": flips, "entries": entries}
        legs_net.append(net); legs_turn.append(turn)
    w = 1.0 / len(coins)
    net = [sum(legs_net[i][t] * w for i in range(len(coins))) for t in range(n)]
    turn = [sum(legs_turn[i][t] * w for i in range(len(coins))) for t in range(n)]
    return {"FULL": seg(net, turn, 0, n), "H2": seg(net, turn, h2a, n), "flips": sum(v["flips"] for v in per_coin.values()), "entries": sum(v["entries"] for v in per_coin.values()), "per_coin": per_coin}

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Y6_freq start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    bars, _ = common15m(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 15m native n=%d h2a=%d coins=%s smoke=%s scale=x%d" % (n, h2a, coins, SMOKE, SCALE))
    assert n > 2000, n
    assert n > h2a + 100, (n, h2a)
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    des, rets = {}, {}
    for c in coins:
        raw, rt, sg = mats[c]
        des[c] = desired_pos(raw, rt, sg, SPECS[c])
        rets[c] = rt[0].tolist()
    log("desired positions built")
    arms = {}
    for freq in ("native_15m", "thinned_1h"):
        arms[freq] = eval_arm(coins, des, rets, n, h2a, FEE, FUND, freq)
        f, h = arms[freq]["FULL"], arms[freq]["H2"]
        log("%s FULL sh=%.3f dd=%.4f cum=%.4f to=%.6f flips=%d entries=%d | H2 sh=%.3f dd=%.4f" % (freq, f["sharpe"], f["mdd"], f["cum"], f["turnover"], arms[freq]["flips"], arms[freq]["entries"], h["sharpe"], h["mdd"]))
    fee2x = {}
    for freq in ("native_15m", "thinned_1h"):
        r = eval_arm(coins, des, rets, n, h2a, FEE2X, FUND, freq)
        fee2x[freq] = {"FULL_sharpe": r["FULL"]["sharpe"], "H2_sharpe": r["H2"]["sharpe"], "FULL_cum": r["FULL"]["cum"], "gap_FULL_sharpe": round(r["FULL"]["sharpe"] - arms[freq]["FULL"]["sharpe"], 3)}
        log("%s fee2x FULL sh=%.3f gap=%+.3f cum=%.4f | H2 sh=%.3f" % (freq, r["FULL"]["sharpe"], fee2x[freq]["gap_FULL_sharpe"], r["FULL"]["cum"], r["H2"]["sharpe"]))
    fn, ft = arms["native_15m"]["FULL"], arms["thinned_1h"]["FULL"]
    hn, ht = arms["native_15m"]["H2"], arms["thinned_1h"]["H2"]
    to_ratio = (fn["turnover"] / ft["turnover"] if ft["turnover"] > 1e-12 else 0.0)
    d_sharpe = round(fn["sharpe"] - ft["sharpe"], 3)
    d_sharpe_h2 = round(hn["sharpe"] - ht["sharpe"], 3)
    rec, reason = ("0", "native 15m sharpe gain >= +0.2") if d_sharpe >= 0.2 else ("1", "native churn not paid (d=%+.3f, ratio=%.2f)" % (d_sharpe, to_ratio))
    log("compare dFULL=%+.3f dH2=%+.3f ratio=%.2f -> ALIGN=%s (%s)" % (d_sharpe, d_sharpe_h2, to_ratio, rec, reason))
    out = {
        "config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0); static equal 0.2 Top5; 15m native cd/ts/vw x16", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": {c: 0.2 for c in coins}, "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "fee2x": FEE2X, "grid": "15m", "grid_bars": n, "h2_start": h2a, "bpy": BPY, "scale": SCALE, "thin_every": THIN_EVERY, "coins": list(coins), "smoke": SMOKE, "note": "decision-cadence contrast only, read-only; Y1B_DECISION_ALIGN shelved, executor default stays unset"},
        "arms": arms, "fee2x": fee2x,
        "compare": {"d_sharpe_FULL": d_sharpe, "d_sharpe_H2": d_sharpe_h2, "d_cum_FULL": round(fn["cum"] - ft["cum"], 4), "d_turnover_FULL": round(fn["turnover"] - ft["turnover"], 6), "turnover_ratio_native_over_thinned": round(to_ratio, 4), "d_flips": arms["native_15m"]["flips"] - arms["thinned_1h"]["flips"], "d_entries": arms["native_15m"]["entries"] - arms["thinned_1h"]["entries"], "fee2x_gap_native": fee2x["native_15m"]["gap_FULL_sharpe"], "fee2x_gap_thinned": fee2x["thinned_1h"]["gap_FULL_sharpe"]},
        "recommendation": {"Y1B_DECISION_ALIGN": rec, "reason": reason, "status": "shelved (executor default stays unset)"},
        "verdict": "PENDING",
        "decision": "PENDING",
        "conclusion": "Y6 15m replay native vs 1h-thinned dFULL=%+.3f ratio=%.2f ALIGN=%s (%s). PENDING; no live change." % (d_sharpe, to_ratio, rec, reason),
    }
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s" % OUT)

if __name__ == "__main__":
    main()
