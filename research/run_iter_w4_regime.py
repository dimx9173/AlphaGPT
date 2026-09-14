"""W4 15m regime split (Top5): bull/bear via BTC 15m close>=MA960.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp lev2 fee=FEE fund=FUND. 15m native: cd/ts/vw x16, BPY=35040.
E10 FORMULA untouched. Read-only: LEV/FUND/FEE config untouched.

Regime: BTC 15m close[t] >= trailing MA960[t] -> bull else bear
(MA960 partial window during warmup). Per-coin sharpe FULL + bull + bear,
FULL basket (equal 0.2) FULL + bull + bear, basket funding-pos/neg split
(sign of basket funding leg F[t]=mean_c pos_c[t]*FUND*LEV: pay>0 / earn<0).

Verdict PENDING (P0-3 FAIL): no adoption, live untouched.
Incremental dump: OUT rewritten after EACH unit (per-coin legs, basket,
regimes, funding split), status partial->done, so partial survives kills.

Smoke: ITER_W4_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
OUT/LOG overridable via ITER_W4_OUT / ITER_W4_LOG.
Output: results/iter_W4_regime.json
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
BPY = 35040.0
SCALE = 16
MA = 960
LEV_USE = 2.0
OUT = pathlib.Path(os.getenv("ITER_W4_OUT", "results/iter_W4_regime.json"))
LOG = pathlib.Path(os.getenv("ITER_W4_LOG", "logs/iter_W4_regime.log"))
SMOKE = os.getenv("ITER_W4_SMOKE") == "1"

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return {int(r["timestamp"]): (float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows}

def common15m(coins):
    raws = {c: load15m(c) for c in coins + ["BTC"]}
    ts = sorted(set(raws[coins[0]]) & set(raws["BTC"]))
    for c in coins[1:]:
        ts = [t for t in ts if t in raws[c]]
    bars = {c: [raws[c][t] for t in ts] for c in coins}
    closes = {c: [b[3] for b in bars[c]] for c in coins}
    btc = [raws["BTC"][t][3] for t in ts]
    return bars, closes, btc, ts

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

def leg_parts(raw, rt, sig, spec):
    """Executed pos/turn/gross series; fee/fund accounting applied later."""
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
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

def seg_subset(net, turn, idx, n):
    s = [net[i] for i in idx]; t = [turn[i] for i in idx]; m = len(s)
    mean = sum(s) / m if m else 0.0
    v = sum((x - mean) ** 2 for x in s) / max(m - 1, 1) if m > 1 else 0.0
    sh = mean / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / m * BPY, 4) if m else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "n": m, "coverage": round(m / n, 4) if n else 0.0, "turnover": round(sum(t) / m, 6) if m else 0.0}

def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_W4_regime start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, closes, btc, ts = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        closes = {c: closes[c][:n] for c in coins}
        btc = btc[:n]; ts = ts[:n]
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d ma=%d" % (n, coins, SMOKE, SCALE, MA))
    assert n > 2000, n
    bull = [1 if btc[t] >= sum(btc[max(0, t - MA + 1):t + 1]) / len(btc[max(0, t - MA + 1):t + 1]) else 0 for t in range(n)]
    bull_idx = [t for t in range(n) if bull[t]]
    bear_idx = [t for t in range(n) if not bull[t]]
    all_idx = list(range(n))
    log("regime BTC close>=MA%d bull_n=%d (%.3f) bear_n=%d (%.3f)" % (MA, len(bull_idx), len(bull_idx) / n, len(bear_idx), len(bear_idx) / n))
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal weights; 15m native cd/ts/vw x16", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "fee": FEE, "fund": FUND, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "ma_window": MA, "regime": "bull = BTC 15m close[t] >= trailing MA%d[t] (partial warmup window); else bear" % MA, "smoke": SMOKE, "note": "regime split only, read-only; live LEV/FUND/FEE untouched"},
        "status": "partial", "per_coin": {}, "basket": {}, "funding_split": {},
        "regime": {"ma_window": MA, "bull_n": len(bull_idx), "bear_n": len(bear_idx), "bull_coverage": round(len(bull_idx) / n, 4), "bear_coverage": round(len(bear_idx) / n, 4)},
        "verdict": "PENDING"}
    dump(res)

    net_c = {}; turn_c = {}; pos_c = {}
    for c in coins:
        raw, rt, sg = mats[c]
        pos, turn, gross = leg_parts(raw, rt, sg, SPECS[c])
        pos_c[c] = pos; turn_c[c] = turn
        net = [gross[t] - turn[t] * FEE * LEV_USE - pos[t] * FUND * LEV_USE for t in range(n)]
        net_c[c] = net
        res["per_coin"][c] = {"FULL": seg_subset(net, turn, all_idx, n), "bull": seg_subset(net, turn, bull_idx, n), "bear": seg_subset(net, turn, bear_idx, n)}
        dump(res)
        f = res["per_coin"][c]["FULL"]
        log("%s FULL sh=%.3f cum=%.4f to=%.6f | bull sh=%.3f n=%d | bear sh=%.3f n=%d" % (c, f["sharpe"], f["cum"], f["turnover"], res["per_coin"][c]["bull"]["sharpe"], res["per_coin"][c]["bull"]["n"], res["per_coin"][c]["bear"]["sharpe"], res["per_coin"][c]["bear"]["n"]))

    bnet = [sum(net_c[c][t] * w[c] for c in coins) for t in range(n)]
    bturn = [sum(turn_c[c][t] * w[c] for c in coins) for t in range(n)]
    res["basket"] = {"FULL": seg_subset(bnet, bturn, all_idx, n), "bull": seg_subset(bnet, bturn, bull_idx, n), "bear": seg_subset(bnet, bturn, bear_idx, n)}
    dump(res)
    b = res["basket"]
    log("BASKET FULL sh=%.3f cum=%.4f | bull sh=%.3f | bear sh=%.3f" % (b["FULL"]["sharpe"], b["FULL"]["cum"], b["bull"]["sharpe"], b["bear"]["sharpe"]))

    fleg = [sum(pos_c[c][t] * w[c] for c in coins) * FUND * LEV_USE for t in range(n)]
    fpos = [t for t in range(n) if fleg[t] > 0]
    fneg = [t for t in range(n) if fleg[t] < 0]
    fzero = n - len(fpos) - len(fneg)
    res["funding_split"] = {"fund_pos_pay": seg_subset(bnet, bturn, fpos, n), "fund_neg_earn": seg_subset(bnet, bturn, fneg, n), "fund_zero_n": fzero, "note": "split by sign of basket funding leg F[t]=mean_c pos_c[t]*fund*lev: pay>0 / earn<0 (fund=%.4f>0 const)" % FUND}
    dump(res)
    fs = res["funding_split"]
    log("FUND pay sh=%.3f n=%d | earn sh=%.3f n=%d | zero n=%d" % (fs["fund_pos_pay"]["sharpe"], fs["fund_pos_pay"]["n"], fs["fund_neg_earn"]["sharpe"], fs["fund_neg_earn"]["n"], fzero))

    res["verdict"] = "PENDING"
    res["decision"] = "NO_ADOPTION"
    res["decision_note"] = "P0-3 permutation FAIL => W4 verdict PENDING; regime/funding splits are descriptive only, no adoption, live untouched."
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, res["verdict"], res["decision"]))

if __name__ == "__main__":
    main()
