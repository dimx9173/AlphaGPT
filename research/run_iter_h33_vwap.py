"""H33 1h VWAP filter (Top5): long only above / short only below 24-bar rolling VWAP.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native:
cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760. Data
data/data_1y/1h/{COIN}.csv. Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

H33 addition: per-coin 24-bar rolling VWAP on typical price (h+l+c)/3
weighted by bar volume, computed on the native 1h grid:
  vwap[t] = sum(typ*vol)/sum(vol) over bars[max(0,t-23):t+1]
(zero-volume window falls back to close[t]). LONG leg kept only where
close[t] > vwap[t]; SHORT leg only where close[t] < vwap[t]; equality
blocks both (measure-zero). Gate applied to lp/sp right after the
quantile mask and BEFORE the joint cooldown+stops, so FULL == LONG +
SHORT exactly.

Baseline (ungated, identical pipeline minus the VWAP gate) reported for
contrast. Offline read-only; verdict PENDING (P0-3 FAIL), no adoption,
live untouched. Incremental dump: results JSON rewritten after each unit
(partial survives). Smoke: ITER_H33_SMOKE=1 -> coins {ETC,TRX}, first
3000 bars. OUT/LOG overridable via ITER_H33_OUT / ITER_H33_LOG.
Output: results/iter_H33_vwap.json.
"""
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
from strategy_manager.config import (
    FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX,
    LEV, FUND, FEE,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 8760.0
SCALE = 4
VWAP_WIN = 24  # 24 x 1h = 24h rolling VWAP
OUT = pathlib.Path(os.getenv("ITER_H33_OUT", "results/iter_H33_vwap.json"))
LOG = pathlib.Path(os.getenv("ITER_H33_LOG", "logs/iter_H33_vwap.log"))
SMOKE = os.getenv("ITER_H33_SMOKE") == "1"

CONFIG = {
    "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; VWAP(24) gate on legs pre-cooldown (long above / short below); static equal 0.2 Top5; 1h native cd/ts/vw x4",
    "formula": list(FORMULA),
    "basket": {c: dict(SPECS[c]) for c in COINS},
    "weights": dict(W),
    "venue": "aster",
    "lev": LEV,
    "fund": FUND,
    "fee": FEE,
    "bpy": BPY,
    "grid": "1h",
    "scale": SCALE,
    "vwap": {
        "window": VWAP_WIN,
        "window_hours": 24.0,
        "typical": "(h+l+c)/3",
        "long_gate": "close>vwap",
        "short_gate": "close<vwap",
        "tie": "blocks both",
        "zero_volume_fallback": "close[t]",
        "applied": "on legs pre-cooldown (post quantile mask)",
    },
    "data_dir": "data/data_1y/1h",
    "smoke": SMOKE,
    "note": "E10 FORMULA untouched; live default untouched; offline read-only",
}


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]


def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars = {}
    closes = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [r[4] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
        closes[c] = closes[c][:n]
    return bars, closes


def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]),
           "high": torch.tensor([[b[1] for b in bars]]),
           "low": torch.tensor([[b[2] for b in bars]]),
           "close": torch.tensor([[b[3] for b in bars]]),
           "volume": torch.tensor([[b[4] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
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


def rolling_vwap(bars, window=VWAP_WIN):
    """24-bar rolling VWAP on typical price. bars: [(o,h,l,c,v)...] -> [vwap...]."""
    n = len(bars)
    out = [0.0] * n
    for t in range(n):
        a = max(0, t - window + 1)
        num = 0.0
        den = 0.0
        for i in range(a, t + 1):
            o, h, l, c, v = bars[i]
            typ = (h + l + c) / 3.0
            num += typ * v
            den += v
        out[t] = num / den if den > 0 else bars[t][3]
    return out


def vwap_gate_masks(closes, vwaps):
    """Strict direction masks: long only above, short only below, tie blocks both."""
    long_ok = [1.0 if c > v else 0.0 for c, v in zip(closes, vwaps)]
    short_ok = [1.0 if c < v else 0.0 for c, v in zip(closes, vwaps)]
    return long_ok, short_ok


def leg_split(raw, rt, sig, spec, fee, fund, lev, gate=None):
    """Joint lp/sp pipeline split into LONG/SHORT legs after shared cooldown+stops.

    gate=None -> ungated baseline. Otherwise gate=(long_ok, short_ok) masks
    applied post-quantile, pre-cooldown (long above VWAP / short below VWAP).
    """
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY,
                      stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    if gate is not None:
        lok, sok = gate
        lp = lp * torch.tensor([lok])
        sp = sp * torch.tensor([sok])
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    lt = (lp - lp.roll(1, dims=1)).abs()
    st = (sp - sp.roll(1, dims=1)).abs()
    lv = bt.leverage
    lnet = (lp * rt * lv - lt * bt.base_fee * lv - lp * bt.default_funding_rate * lv)[0].tolist()
    snet = (-sp * rt * lv - st * bt.base_fee * lv + sp * bt.default_funding_rate * lv)[0].tolist()
    fnet = [a + b for a, b in zip(lnet, snet)]
    return lnet, snet, fnet, lt[0].tolist(), st[0].tolist(), lp[0].tolist(), sp[0].tolist()


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0
    pk0 = -1e18
    md = 0.0
    for x in s:
        cs += x
        pk0 = max(pk0, cs)
        md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


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


def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H33_vwap start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, closes = common1h(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        closes = {c: closes[c][:n] for c in coins}
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d vwap=%d" % (n, coins, SMOKE, SCALE, VWAP_WIN))
    assert n > 2000, n
    vwap_map = {c: rolling_vwap(bars[c], VWAP_WIN) for c in coins}
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    cfg = dict(CONFIG, grid_bars=n, coins=list(coins), weights=dict(w))
    res = {"config": cfg, "coins": {}, "basket": {}, "baseline": {},
           "verdict": "PENDING", "decision": "NO_ADOPTION",
           "decision_note": "P0-3 permutation FAIL => H33 verdict PENDING; VWAP-gate attribution only, nothing promoted, live untouched.",
           "status": "PARTIAL"}
    dump(res)
    legs_g, legs_b = {}, {}
    for c in coins:
        raw, rt, sg = mats[c]
        vw = vwap_map[c]
        cl = closes[c]
        gate = vwap_gate_masks(cl, vw)
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, gate)
        bl_, bs_, bf_, blt_, bst_, _, _ = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, None)
        legs_g[c] = {"long": lnet, "short": snet, "full": fnet, "lt": lt, "st": st, "lp": lp, "sp": sp}
        legs_b[c] = {"full": bf_, "lt": blt_, "st": bst_}
        f = seg(fnet, [a + b for a, b in zip(lt, st)], 0, n)
        ls = seg(lnet, lt, 0, n)
        ss = seg(snet, st, 0, n)
        cumF = sum(fnet)
        ls["trades"] = entries_of(lp)
        ls["active_bars"] = active_of(lp)
        ss["trades"] = entries_of(sp)
        ss["active_bars"] = active_of(sp)
        f["trades"] = ls["trades"] + ss["trades"]
        ls["pnl_share"] = round(sum(lnet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        ss["pnl_share"] = round(sum(snet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        n_lo = sum(1 for a, b in zip(cl, vw) if a > b)
        n_so = sum(1 for a, b in zip(cl, vw) if a < b)
        vmean = sum(vw) / n if n else 0.0
        res["coins"][c] = {
            "FULL": f, "LONG": ls, "SHORT": ss,
            "BASE_FULL": seg(bf_, [a + b for a, b in zip(blt_, bst_)], 0, n),
            "vwap": {"window": VWAP_WIN,
                     "mean": round(vmean, 6),
                     "min": round(min(vw), 6),
                     "max": round(max(vw), 6),
                     "n_long_open": n_lo, "n_short_open": n_so,
                     "frac_long_open": round(n_lo / n, 4),
                     "frac_short_open": round(n_so / n, 4)},
        }
        log("%s gated cumF=%.4f cumL=%.4f cumS=%.4f baseF=%.4f tradesL=%d tradesS=%d vwapL=%.3f vwapS=%.3f" % (
            c, sum(fnet), sum(lnet), sum(snet), sum(bf_), ls["trades"], ss["trades"], n_lo / n, n_so / n))
        dump(res)  # incremental: partial survives interruption
    bfull = [sum(legs_g[c]["full"][t] * w[c] for c in coins) for t in range(n)]
    blong = [sum(legs_g[c]["long"][t] * w[c] for c in coins) for t in range(n)]
    bshort = [sum(legs_g[c]["short"][t] * w[c] for c in coins) for t in range(n)]
    bturn = [sum((legs_g[c]["lt"][t] + legs_g[c]["st"][t]) * w[c] for c in coins) for t in range(n)]
    blt = [sum(legs_g[c]["lt"][t] * w[c] for c in coins) for t in range(n)]
    bst = [sum(legs_g[c]["st"][t] * w[c] for c in coins) for t in range(n)]
    bf = seg(bfull, bturn, 0, n)
    bl = seg(blong, blt, 0, n)
    bs = seg(bshort, bst, 0, n)
    bf["trades"] = sum(res["coins"][c]["FULL"]["trades"] for c in coins)
    bl["pnl_share"] = round(sum(blong) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bs["pnl_share"] = round(sum(bshort) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    res["basket"] = {"FULL": bf, "LONG": bl, "SHORT": bs,
                     "coin_pnl_share": {c: round(sum(legs_g[c]["full"]) * w[c] / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in coins}}
    bbase = [sum(legs_b[c]["full"][t] * w[c] for c in coins) for t in range(n)]
    bbaseturn = [sum((legs_b[c]["lt"][t] + legs_b[c]["st"][t]) * w[c] for c in coins) for t in range(n)]
    res["baseline"] = {"basket_FULL": seg(bbase, bbaseturn, 0, n),
                       "coin_FULL": {c: round(sum(legs_b[c]["full"]), 4) for c in coins}}
    res["status"] = "COMPLETE"
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION gated sh=%.3f (base %.3f) long_share=%.3f short_share=%.3f" % (
        OUT, bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"], bl["pnl_share"], bs["pnl_share"]))


if __name__ == "__main__":
    main()
