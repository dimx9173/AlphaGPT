"""Z6 long/short attribution (15m native): Top5 per-coin long-only vs short-only.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp lev2 fund=FUND fee=FEE. 15m native: cd/ts/vw x16, BPY=35040.
Joint lp/sp pipeline per coin is split into long/short legs AFTER the shared
cooldown+stops (so FULL == LONG + SHORT exactly). Bull/bear split via BTC 4h
close>=MA200 mapped onto the 15m grid. Read-only, verdict PENDING, no adoption.
Output: results/iter_Z6_ls.json
"""
import bisect
import csv
import json
import math
import os
import pathlib
import sys
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
BPY = 35040.0
SCALE = 16
MA = 200
OUT = pathlib.Path("results/iter_Z6_ls.json")
LOG = pathlib.Path("logs/iter_Z6_ls.log")

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common15m(coins):
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; closes = {}; ts = None
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        if ts is None:
            ts = [r[0] for r in rr]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [r[4] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]; closes[c] = closes[c][:n]
    return bars, closes, ts[:n]

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

def leg_split(raw, rt, sig, spec, fee, fund, lev):
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
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

def active_of(pos):
    return sum(1 for x in pos if x > 0.5)

def mask_stat(s, mask):
    sub = [x for x, f in zip(s, mask) if f]
    n = len(sub)
    cum = sum(sub)
    return {"sharpe": round(sharpe_of(sub), 3), "cum": round(cum, 4), "ann": round(cum / n * BPY, 4) if n else 0.0, "n": n}

def bull_mask_15m(ts15):
    rows = list(csv.DictReader(open("data/data_1y/4h/BTC.csv")))
    ts4 = [int(r["timestamp"]) for r in rows]
    cl4 = [float(r["close"]) for r in rows]
    bull4 = []
    run = 0.0
    for i, c in enumerate(cl4):
        run += c
        if i >= MA:
            run -= cl4[i - MA]
            bull4.append(c >= run / MA)
        else:
            bull4.append(c >= run / (i + 1) if i + 1 >= MA else False)
    mask = []
    for t in ts15:
        j = bisect.bisect_right(ts4, t) - 1
        j = max(0, min(j, len(ts4) - 1))
        mask.append(bool(bull4[j]))
    return mask, {"n4h": len(ts4), "warmup": MA - 1}

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z6_ls start\n")
    bars, closes, ts15 = common15m(COINS)
    n = len(bars["ETC"])
    log("common 15m native n=%d scale=x%d" % (n, SCALE))
    bull = None
    bull, reg = bull_mask_15m(ts15)
    bear = [not b for b in bull]
    log("regime BTC4h n=%d warmup=%d bull=%d bear=%d" % (reg["n4h"], reg["warmup"], sum(bull), sum(bear)))
    mats = {c: build_sig(bars[c]) for c in COINS}
    legs = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV)
        legs[c] = {"long": lnet, "short": snet, "full": fnet, "lt": lt, "st": st, "lp": lp, "sp": sp}
        log("%s cumF=%.4f cumL=%.4f cumS=%.4f tradesL=%d tradesS=%d" % (c, sum(fnet), sum(lnet), sum(snet), entries_of(lp), entries_of(sp)))
    coins = {}
    for c in COINS:
        L = legs[c]
        f = seg(L["full"], [a + b for a, b in zip(L["lt"], L["st"])], 0, n)
        ls = seg(L["long"], L["lt"], 0, n)
        ss = seg(L["short"], L["st"], 0, n)
        cumF = sum(L["full"])
        ls["trades"] = entries_of(L["lp"]); ls["active_bars"] = active_of(L["lp"])
        ss["trades"] = entries_of(L["sp"]); ss["active_bars"] = active_of(L["sp"])
        ls["pnl_share"] = round(sum(L["long"]) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        ss["pnl_share"] = round(sum(L["short"]) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        entry = {"FULL": f, "LONG": ls, "SHORT": ss}
        for name, m in (("bull", bull), ("bear", bear)):
            sub = {}
            for k, key in (("FULL", "full"), ("LONG", "long"), ("SHORT", "short")):
                st_ = mask_stat(L[key], m)
                st_["cum_share"] = round(st_["cum"] / f["cum"], 4) if abs(f["cum"]) > 1e-12 else 0.0
                sub[k] = st_
            entry[name] = sub
        coins[c] = entry
    bfull = [sum(legs[c]["full"][t] * W[c] for c in COINS) for t in range(n)]
    blong = [sum(legs[c]["long"][t] * W[c] for c in COINS) for t in range(n)]
    bshort = [sum(legs[c]["short"][t] * W[c] for c in COINS) for t in range(n)]
    bturn = [sum((legs[c]["lt"][t] + legs[c]["st"][t]) * W[c] for c in COINS) for t in range(n)]
    blt = [sum(legs[c]["lt"][t] * W[c] for c in COINS) for t in range(n)]
    bst = [sum(legs[c]["st"][t] * W[c] for c in COINS) for t in range(n)]
    bf = seg(bfull, bturn, 0, n)
    bl = seg(blong, blt, 0, n)
    bs = seg(bshort, bst, 0, n)
    bl["pnl_share"] = round(sum(blong) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bs["pnl_share"] = round(sum(bshort) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    basket = {"FULL": bf, "LONG": bl, "SHORT": bs}
    for name, m in (("bull", bull), ("bear", bear)):
        sub = {}
        for k, s_ in (("FULL", bfull), ("LONG", blong), ("SHORT", bshort)):
            st_ = mask_stat(s_, m)
            st_["cum_share"] = round(st_["cum"] / bf["cum"], 4) if abs(bf["cum"]) > 1e-12 else 0.0
            sub[k] = st_
        basket[name] = sub
    basket["coin_pnl_share"] = {c: round(sum(legs[c]["full"]) * W[c] / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in COINS}
    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; joint lp/sp split into LONG/SHORT legs post cooldown+stops; static equal 0.2 Top5; 15m native cd/ts/vw x16", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "bpy": BPY, "grid": "15m", "grid_bars": n, "scale": SCALE, "regime": {"src": "data/data_1y/4h/BTC.csv", "rule": "BTC 4h close>=MA200", "ma": MA, "warmup_bars": reg["warmup"], "n4h": reg["n4h"], "bull_n": sum(bull), "bear_n": sum(bear), "bull_share": round(sum(bull) / n, 4)}, "trades_def": "entries on executed (rolled) binary positions; active_bars = bars with pos>0.5", "note": "E10 FORMULA untouched; live default untouched; offline read-only"}, "coins": coins, "basket": basket, "verdict": "PENDING", "decision": "NO_ADOPTION", "decision_note": "P0-3 permutation FAIL => Z6 verdict PENDING; attribution only, nothing promoted, live untouched."}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=PENDING decision=NO_ADOPTION basket sh=%.3f long_share=%.3f short_share=%.3f" % (OUT, bf["sharpe"], bl["pnl_share"], bs["pnl_share"]))

if __name__ == "__main__":
    main()
