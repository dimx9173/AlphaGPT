# H32 1h OBV filter (Top5) -- DIAGNOSTIC ONLY.
# P0-3 permutation FAILED => verdict PENDING, no adoption, live untouched.
# Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA via StackVM+FeatureEngineer;
# MemeBacktest venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1.
# OBV gate: per-coin legs gated by OBV slope sign (24-bar); FULL sharpe + long-short split + pass rate.
# Top5 locked specs ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24) ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24) KAS(0.88/0.12/cd6/None/ts24) q0.3.
# 1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760. Data data/data_1y/1h/{COIN}.csv (~8760 rows). Equal 0.2 weights.
# OBV window 24 bars on 1h == same 24h lookback as V8 96x15m.
# Incremental dump of results JSON after each unit (partial survives).
# Smoke: ITER_H32_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars. ITER_H32_OUT / ITER_H32_LOG override paths (tests use temp files).
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE
assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 8760.0
SCALE = 4
OBV_WIN = 24
Q = 0.3
OUT = pathlib.Path(os.getenv("ITER_H32_OUT", "results/iter_H32_obv.json"))
LOG = pathlib.Path(os.getenv("ITER_H32_LOG", "logs/iter_H32_obv.log"))
SMOKE = os.getenv("ITER_H32_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_N = 3000
def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; closes = {}; vols = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [r[4] for r in rr]
        vols[c] = [r[5] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]; closes[c] = closes[c][:n]; vols[c] = vols[c][:n]
    return bars, closes, vols

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

def obv_gate(closes, vols, win=OBV_WIN):
    n = len(closes)
    obv = [0.0] * n
    run = 0.0
    for t in range(n):
        if t == 0:
            run = float(vols[0])
        else:
            if closes[t] > closes[t - 1]:
                run += float(vols[t])
            elif closes[t] < closes[t - 1]:
                run -= float(vols[t])
        obv[t] = run
    gate = [1.0] * n
    for t in range(n):
        if t < win:
            gate[t] = 1.0
        else:
            gate[t] = 1.0 if (obv[t] - obv[t - win]) > 0 else 0.0
    return gate, obv

def leg_nets(raw, rt, sig, spec, closes, vols, fee, fund, lev, use_gate):
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    if use_gate:
        gate, _ = obv_gate(closes, vols, OBV_WIN)
        g = torch.tensor([gate])
        lp = lp * g; sp = sp * g
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    lt = (lp - lp.roll(1, dims=1)).abs(); st = (sp - sp.roll(1, dims=1)).abs()
    lv = bt.leverage
    lnet = (lp * rt * lv - lt * bt.base_fee * lv - lp * bt.default_funding_rate * lv)[0].tolist()
    snet = (-sp * rt * lv - st * bt.base_fee * lv + sp * bt.default_funding_rate * lv)[0].tolist()
    fnet = [a + b for a, b in zip(lnet, snet)]
    return lnet, snet, fnet, lt[0].tolist(), st[0].tolist(), lp[0].tolist(), sp[0].tolist()

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

def entries_of(pos):
    n = 0
    prev = 0.0
    for x in pos:
        cur = 1.0 if x > 0.5 else 0.0
        if cur > 0.5 and prev <= 0.5:
            n += 1
        prev = cur
    return n

def dump(obj):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=1, ensure_ascii=False))

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H32_obv start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    w = 1.0 / len(coins)
    bars, closes, vols = common1h(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, SMOKE_N)
        bars = {c: bars[c][:n] for c in coins}
        closes = {c: closes[c][:n] for c in coins}
        vols = {c: vols[c][:n] for c in coins}
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d obv_win=%d q=%.1f" % (n, coins, SMOKE, SCALE, OBV_WIN, Q))
    assert n > 2000, n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    res = {"config": {"engine": "mirror run_weight_modes.py leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; legs gated by OBV slope sign (24-bar)", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "coins": list(coins), "weights": {c: round(w, 4) for c in coins}, "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "q": Q, "grid": "1h", "grid_bars": n, "bpy": BPY, "scale": SCALE, "obv_win": OBV_WIN, "gate_rule": "OBV[t]-OBV[t-24]>0 else flat; warmup first 24 bars pass-through (gate=1)", "smoke": SMOKE, "note": "E10 FORMULA untouched; live chain untouched; offline read-only; PENDING (P0-3 FAIL), not demo evidence"}, "per_coin": {}, "status": "partial"}
    dump(res)
    store = {}
    for c in coins:
        raw, rt, sg = mats[c]
        gate, _ = obv_gate(closes[c], vols[c], OBV_WIN)
        lb, sb, fb, ltb, stb, lpb, spb = leg_nets(raw, rt, sg, SPECS[c], closes[c], vols[c], FEE, FUND, LEV, False)
        lg, sg2, fg, ltg, stg, lpg, spg = leg_nets(raw, rt, sg, SPECS[c], closes[c], vols[c], FEE, FUND, LEV, True)
        tb = [a + b for a, b in zip(ltb, stb)]
        tg = [a + b for a, b in zip(ltg, stg)]
        fB = seg(fb, tb, 0, n); lB = seg(lb, ltb, 0, n); sB = seg(sb, stb, 0, n)
        fG = seg(fg, tg, 0, n); lG = seg(lg, ltg, 0, n); sG = seg(sg2, stg, 0, n)
        cumB = sum(fb); cumG = sum(fg)
        lB["trades"] = entries_of(lpb); sB["trades"] = entries_of(spb)
        lG["trades"] = entries_of(lpg); sG["trades"] = entries_of(spg)
        lB["pnl_share"] = round(sum(lb) / cumB, 4) if abs(cumB) > 1e-12 else 0.0
        sB["pnl_share"] = round(sum(sb) / cumB, 4) if abs(cumB) > 1e-12 else 0.0
        lG["pnl_share"] = round(sum(lg) / cumG, 4) if abs(cumG) > 1e-12 else 0.0
        sG["pnl_share"] = round(sum(sg2) / cumG, 4) if abs(cumG) > 1e-12 else 0.0
        errB = max(abs(f - (a + b)) for f, a, b in zip(fb, lb, sb))
        errG = max(abs(f - (a + b)) for f, a, b in zip(fg, lg, sg2))
        pass_rate = round(sum(gate) / n, 4)
        res["per_coin"][c] = {"baseline": {"FULL": fB, "LONG": lB, "SHORT": sB, "additivity_err": errB}, "gated": {"FULL": fG, "LONG": lG, "SHORT": sG, "additivity_err": errG}, "gate": {"win": OBV_WIN, "pass_bars": int(sum(gate)), "pass_rate": pass_rate, "warmup": OBV_WIN}, "delta": {"d_sharpe_full": round(fG["sharpe"] - fB["sharpe"], 3), "d_cum_full": round(fG["cum"] - fB["cum"], 4), "d_turnover_full": round(fG["turnover"] - fB["turnover"], 6)}}
        store[c] = {"base": (lb, sb, fb, ltb, stb), "gated": (lg, sg2, fg, ltg, stg)}
        dump(res)
        log("%s base sh=%.3f cum=%.4f to=%.6f | gated sh=%.3f cum=%.4f to=%.6f | pass=%.3f | dsh=%+.3f" % (c, fB["sharpe"], fB["cum"], fB["turnover"], fG["sharpe"], fG["cum"], fG["turnover"], pass_rate, fG["sharpe"] - fB["sharpe"]))
    basket = {}
    for name, idx in (("baseline", "base"), ("gated", "gated")):
        bf = [sum(store[c][idx][2][t] * w for c in coins) for t in range(n)]
        bl = [sum(store[c][idx][0][t] * w for c in coins) for t in range(n)]
        bs = [sum(store[c][idx][1][t] * w for c in coins) for t in range(n)]
        bt_ = [sum((store[c][idx][3][t] + store[c][idx][4][t]) * w for c in coins) for t in range(n)]
        blt = [sum(store[c][idx][3][t] * w for c in coins) for t in range(n)]
        bst = [sum(store[c][idx][4][t] * w for c in coins) for t in range(n)]
        f = seg(bf, bt_, 0, n); l = seg(bl, blt, 0, n); s = seg(bs, bst, 0, n)
        tot = sum(bf)
        l["pnl_share"] = round(sum(bl) / tot, 4) if abs(tot) > 1e-12 else 0.0
        s["pnl_share"] = round(sum(bs) / tot, 4) if abs(tot) > 1e-12 else 0.0
        err = max(abs(f_ - (a + b)) for f_, a, b in zip(bf, bl, bs))
        basket[name] = {"FULL": f, "LONG": l, "SHORT": s, "additivity_err": err}
    dsh = round(basket["gated"]["FULL"]["sharpe"] - basket["baseline"]["FULL"]["sharpe"], 3)
    basket["delta"] = {"d_sharpe_full": dsh, "d_cum_full": round(basket["gated"]["FULL"]["cum"] - basket["baseline"]["FULL"]["cum"], 4)}
    res["basket"] = basket
    res["verdict"] = "PENDING"
    res["verdict_detail"] = "PENDING_P03_FAIL"
    res["decision"] = "NO_ADOPTION"
    res["decision_note"] = "P0-3 permutation FAIL => V8 verdict PENDING; OBV-gate attribution only (basket dsh=%+.3f), nothing promoted, live untouched." % dsh
    res["conclusion"] = "PENDING: H32 1h OBV-slope(24) gate Top5; basket base sh=%.3f gated sh=%.3f; no live change" % (basket["baseline"]["FULL"]["sharpe"], basket["gated"]["FULL"]["sharpe"])
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION base=%.3f gated=%.3f dsh=%+.3f" % (OUT, basket["baseline"]["FULL"]["sharpe"], basket["gated"]["FULL"]["sharpe"], dsh))

if __name__ == "__main__":
    main()
