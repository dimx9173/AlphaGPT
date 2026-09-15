"""H93 1h Williams v2 %R filter (Top5): entry-only %R(56)>-20 long / <-80 short gate.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005 fee0.0004; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native:
cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760. Data
data/data_1y/1h/{COIN}.csv (~8760 rows). Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

H93 addition: per-coin Williams %R(56) on 1h bars (56x1h = 56h window).
%R(t) = (HH56 - close)/(HH56 - LL56) * -100 in [-100, 0]. LONG leg gated to
%R>-20 (close pinned near highs), SHORT leg to %R<-80 (close pinned near
lows). Gate applied to lp/sp right after the quantile mask and BEFORE the
joint cooldown+stops (entry-only: exits still flow through cooldown/stops),
so FULL == LONG + SHORT exactly. Warmup (first 56 bars): %R=-50.0 neutral
=> both legs closed.

Baseline (ungated, identical pipeline minus the %R gate) reported for
contrast. Offline read-only; verdict PENDING (P0-3 FAIL), no adoption,
live untouched. Incremental dump: results JSON rewritten after each coin
(partial survives). Smoke: ITER_H93_SMOKE=1 -> coins {ETC,TRX}, first
3000 bars. OUT/LOG overridable via ITER_H93_OUT / ITER_H93_LOG.
Output: results/iter_H93_willr.json.
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
WILLR_WINDOW = 56  # 28 x 1h = 28h
WILLR_LONG = -20.0
WILLR_SHORT = -80.0
WILLR_WARMUP = WILLR_WINDOW  # 28 bars neutral
WILLR_NEUTRAL = -50.0
OUT = pathlib.Path(os.getenv("ITER_H93_OUT", "results/iter_H93_willr.json"))
LOG = pathlib.Path(os.getenv("ITER_H93_LOG", "logs/iter_h93_willr.log"))
SMOKE = os.getenv("ITER_H93_SMOKE") == "1"

CONFIG = {
    "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; WillR(56) entry gate on legs pre-cooldown (long R>-20 / short R<-80); static equal 0.2 Top5; 1h native cd/ts/vw x4",
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
    "willr": {
        "window": WILLR_WINDOW,
        "window_hours": 56.0,
        "def": "%R=(HH56-close)/(HH56-LL56)*-100 in [-100,0]",
        "long_gate": "R>-20",
        "short_gate": "R<-80",
        "long_th": WILLR_LONG,
        "short_th": WILLR_SHORT,
        "warmup_value": WILLR_NEUTRAL,
        "warmup_bars": WILLR_WARMUP,
        "applied": "entry-only on legs pre-cooldown (post quantile mask)",
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


def willr(highs, lows, closes, window=WILLR_WINDOW):
    """Williams %R over bars. First WILLR_WARMUP bars are -50.0 (neutral)."""
    n = len(closes)
    R = [WILLR_NEUTRAL] * n
    if n <= window:
        return R
    for t in range(window - 1, n):
        hh = max(highs[t - window + 1:t + 1])
        ll = min(lows[t - window + 1:t + 1])
        if hh <= ll:
            R[t] = WILLR_NEUTRAL
        else:
            R[t] = (hh - closes[t]) / (hh - ll) * -100.0
    for t in range(min(WILLR_WARMUP, n)):
        R[t] = WILLR_NEUTRAL
    return R


def leg_split(raw, rt, sig, spec, fee, fund, lev, r=None):
    """Joint lp/sp pipeline split into LONG/SHORT legs after shared cooldown+stops.

    r=None -> ungated baseline. Otherwise LONG gated to %R>-20, SHORT to
    %R<-80 (masks applied post-quantile, pre-cooldown: entry-only gate).
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
    if r is not None:
        rr = torch.tensor([r])
        lp = lp * (rr > WILLR_LONG).float()
        sp = sp * (rr < WILLR_SHORT).float()
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
    open(LOG, "w").write("iter_H93_willr start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, closes = common1h(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        closes = {c: closes[c][:n] for c in coins}
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d willr=%d" % (n, coins, SMOKE, SCALE, WILLR_WINDOW))
    assert n > 2000, n
    r_map = {}
    for c in coins:
        h = [b[1] for b in bars[c]]
        l = [b[2] for b in bars[c]]
        r_map[c] = willr(h, l, closes[c], WILLR_WINDOW)
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked) + WillR%d" % WILLR_WINDOW)
    cfg = dict(CONFIG, grid_bars=n, coins=list(coins), weights=dict(w))
    res = {"config": cfg, "coins": {}, "basket": {}, "baseline": {},
           "verdict": "PENDING", "decision": "NO_ADOPTION",
           "decision_note": "P0-3 permutation FAIL => H93 verdict PENDING; WillR-gate attribution only, nothing promoted, live untouched.",
           "conclusion": "PENDING: H93 1h Williams %R(56) entry gate Top5; no adoption, live untouched.",
           "status": "PARTIAL"}
    dump(res)
    legs_g, legs_b = {}, {}
    for c in coins:
        raw, rt, sg = mats[c]
        r = r_map[c]
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, r)
        bl, bs, bf, _, _, _, _ = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, None)
        legs_g[c] = {"long": lnet, "short": snet, "full": fnet, "lt": lt, "st": st, "lp": lp, "sp": sp}
        legs_b[c] = {"full": bf}
        f = seg(fnet, [a + b for a, b in zip(lt, st)], 0, n)
        ls = seg(lnet, lt, 0, n)
        ss = seg(snet, st, 0, n)
        cumF = sum(fnet)
        tL = entries_of(lp)
        tS = entries_of(sp)
        aL = active_of(lp)
        aS = active_of(sp)
        f["trades"] = tL + tS
        f["active_bars"] = aL + aS
        ls["trades"] = tL
        ls["active_bars"] = aL
        ss["trades"] = tS
        ss["active_bars"] = aS
        ls["pnl_share"] = round(sum(lnet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        ss["pnl_share"] = round(sum(snet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        n_lo = sum(1 for v in r if v > WILLR_LONG)
        n_so = sum(1 for v in r if v < WILLR_SHORT)
        res["coins"][c] = {
            "FULL": f, "LONG": ls, "SHORT": ss,
            "BASE_FULL": seg(bf, [0.0] * n, 0, n),
            "willr": {"mean": round(sum(r) / n, 3), "min": round(min(r), 3),
                      "max": round(max(r), 3),
                      "n_long_open": n_lo, "n_short_open": n_so,
                      "frac_long_open": round(n_lo / n, 4),
                      "frac_short_open": round(n_so / n, 4),
                      "warmup_bars": WILLR_WARMUP},
        }
        log("%s gated cumF=%.4f cumL=%.4f cumS=%.4f baseF=%.4f tradesL=%d tradesS=%d rL=%.3f rS=%.3f" % (
            c, sum(fnet), sum(lnet), sum(snet), sum(bf), tL, tS, n_lo / n, n_so / n))
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
    bl["pnl_share"] = round(sum(blong) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bs["pnl_share"] = round(sum(bshort) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bf["trades"] = sum(res["coins"][c]["FULL"]["trades"] for c in coins)
    bf["active_bars"] = sum(res["coins"][c]["FULL"]["active_bars"] for c in coins)
    res["basket"] = {"FULL": bf, "LONG": bl, "SHORT": bs,
                     "coin_pnl_share": {c: round(sum(legs_g[c]["full"]) * w[c] / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in coins}}
    bbase = [sum(legs_b[c]["full"][t] * w[c] for c in coins) for t in range(n)]
    res["baseline"] = {"basket_FULL": seg(bbase, [0.0] * n, 0, n),
                       "coin_FULL": {c: round(sum(legs_b[c]["full"]), 4) for c in coins}}
    res["status"] = "COMPLETE"
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION gated sh=%.3f (base %.3f) long_share=%.3f short_share=%.3f trades=%d" % (
        OUT, bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"], bl["pnl_share"], bs["pnl_share"], bf["trades"]))


if __name__ == "__main__":
    main()
