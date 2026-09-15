"""H64 1h Bollinger v4 (Top5): entry only outside BB(192,2) band.

Mirror research/run_weight_modes.py leg_net + research/run_iter_h27_rsi.py
leg_split: E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer;
MemeBacktest venue=aster lev2 short_enabled fund0.0005 fee0.0004; quantile q0.3
long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
Top5 weights. 1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows, cols
timestamp,open,high,low,close,volume,quote_volume,trades). Equal 0.2 weights
unless round varies them.

H64 addition: per-coin Bollinger BB(192,2) on 1h closes (192x1h = 192h lookback).
SMA(192) + k*SD(192) with population std (1/N), causal current-bar inclusive.
ENTRY ONLY OUTSIDE band: both LONG and SHORT legs gated to bars where
close > upper OR close < lower (breakout regime). Gate masks applied to lp/sp
right after the quantile mask and BEFORE the joint cooldown+stops, so
FULL == LONG + SHORT exactly. Warmup (first 192 bars): neutral, both legs
closed (SMA undefined on a partial window).

Baseline (ungated, identical pipeline minus the BB gate) reported for
contrast. Metric per coin: FULL sharpe/trades (+ann/mdd/cum/final_x/n/
turnover/active_bars) + LONG/SHORT splits + coverage (frac bars outside).
Read-only, verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Output: results/iter_H64_bb.json (incrementally dumped after each coin, so a
partial file survives interruption).

Smoke mode (for tests): ITER_H64_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
ITER_H64_OUT / ITER_H64_LOG override output paths.
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
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE

assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units
BB_WINDOW = 192   # 192 x 1h = 192h
BB_K = 2.0
WARMUP = BB_WINDOW  # first 192 bars neutral, both legs closed
OUT = pathlib.Path(os.getenv("ITER_H64_OUT", "results/iter_H64_bb.json"))
LOG = pathlib.Path(os.getenv("ITER_H64_LOG", "logs/iter_H64_bb.log"))
SMOKE = os.getenv("ITER_H64_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_BARS = 3000

CONFIG = {
    "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; BB(192,2) outside-band entry gate on both legs pre-cooldown; static equal 0.2 Top5; 1h native cd/ts/vw x4",
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
    "data_dir": "data/data_1y/1h",
    "bb": {
        "window": BB_WINDOW,
        "window_hours": 192.0,
        "k": BB_K,
        "method": "SMA(192) +/- k * population-SD(192), causal current-bar inclusive",
        "gate": "entry only outside band: close > upper OR close < lower (both legs)",
        "warmup_value": "neutral (both legs closed)",
        "warmup_bars": WARMUP,
        "applied": "on both legs pre-cooldown (post quantile mask)",
    },
    "note": "E10 FORMULA untouched; live default untouched; offline read-only",
}


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]


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


def bollinger(closes, window=BB_WINDOW, k=BB_K):
    """Causal BB(window,k): sma + k*popSD over closes[i-window+1..i].

    First `window` bars (i < window): bands None (warmup, gate closed).
    Returns (mid, upper, lower, width) lists; warmup entries 0.0/None-safe 0.0.
    """
    n = len(closes)
    w = max(int(window), 1)
    mid = [0.0] * n
    upper = [0.0] * n
    lower = [0.0] * n
    width = [0.0] * n
    for i in range(n):
        if i < w:
            continue
        win = closes[i - w + 1:i + 1]
        m = sum(win) / w
        v = sum((x - m) ** 2 for x in win) / w
        sd = math.sqrt(v) if v > 0 else 0.0
        mid[i] = m
        upper[i] = m + k * sd
        lower[i] = m - k * sd
        width[i] = 2.0 * k * sd
    return mid, upper, lower, width


def gate_outside(closes, upper, lower):
    """Entry-only-outside mask; warmup bars closed."""
    n = len(closes)
    g = [0.0] * n
    for i in range(n):
        if i < WARMUP:
            continue
        if closes[i] > upper[i] or closes[i] < lower[i]:
            g[i] = 1.0
    return g


def leg_split(raw, rt, sig, spec, fee, fund, lev, mask=None):
    """Joint lp/sp pipeline split into LONG/SHORT legs after shared cooldown+stops.

    mask=None -> ungated baseline. Otherwise the outside-band mask gates BOTH
    legs (applied post-quantile, pre-cooldown).
    """
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    if mask is not None:
        g = torch.tensor([mask])
        lp = lp * g
        sp = sp * g
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


def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H64_bb start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    w = 1.0 / len(coins)
    bars, closes = common1h(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, SMOKE_BARS)
        for c in coins:
            bars[c] = bars[c][:n]
            closes[c] = closes[c][:n]
    h2a = n // 2
    log("common 1h native n=%d h2a=%d coins=%s smoke=%s scale=x%d bb=(%d,%.1f) warmup=%d" % (n, h2a, coins, SMOKE, SCALE, BB_WINDOW, BB_K, WARMUP))
    assert n > WARMUP + 100, n
    bb_map = {}
    mask_map = {}
    for c in coins:
        mid, upper, lower, width = bollinger(closes[c])
        bb_map[c] = (mid, upper, lower, width)
        mask_map[c] = gate_outside(closes[c], upper, lower)
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    cfg = dict(CONFIG, grid_bars=n, h2_start=h2a, smoke=SMOKE, coins=list(coins))
    if SMOKE:
        cfg["basket"] = {c: dict(SPECS[c]) for c in coins}
        cfg["weights"] = {c: round(1.0 / len(coins), 4) for c in coins}
    res = {"config": cfg, "coins": {}, "basket": {}, "baseline": {}, "verdict": "PENDING", "decision": "NO_ADOPTION",
           "decision_note": "P0-3 permutation FAIL => H64 verdict PENDING; BB-gate attribution only, nothing promoted, live untouched.",
           "status": "PARTIAL"}
    dump(res)
    legs_g, legs_b = {}, {}
    for c in coins:
        raw, rt, sg = mats[c]
        mask = mask_map[c]
        mid, upper, lower, width = bb_map[c]
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, mask)
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
        ls["pnl_share"] = round(sum(lnet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        ss["pnl_share"] = round(sum(snet) / cumF, 4) if abs(cumF) > 1e-12 else 0.0
        f["trades"] = ls["trades"] + ss["trades"]
        f["active_bars"] = ls["active_bars"] + ss["active_bars"]
        n_out = sum(1 for v in mask if v > 0.5)
        res["coins"][c] = {
            "FULL": f, "LONG": ls, "SHORT": ss,
            "BASE_FULL": seg(bf_, [a + b for a, b in zip(blt_, bst_)], 0, n),
            "H2_FULL_sharpe": seg(fnet, [a + b for a, b in zip(lt, st)], h2a, n)["sharpe"],
            "bb": {"width_mean": round(sum(width) / n, 6), "width_max": round(max(width), 6),
                   "n_outside": n_out, "frac_outside": round(n_out / n, 4)},
        }
        log("%s gated cumF=%.4f cumL=%.4f cumS=%.4f baseF=%.4f trades=%d (L%d/S%d) outside=%.3f" % (
            c, sum(fnet), sum(lnet), sum(snet), sum(bf_), f["trades"], ls["trades"], ss["trades"], n_out / n))
        dump(res)  # incremental: partial survives interruption
    bfull = [sum(legs_g[c]["full"][t] * w for c in coins) for t in range(n)]
    blong = [sum(legs_g[c]["long"][t] * w for c in coins) for t in range(n)]
    bshort = [sum(legs_g[c]["short"][t] * w for c in coins) for t in range(n)]
    bturn = [sum((legs_g[c]["lt"][t] + legs_g[c]["st"][t]) * w for c in coins) for t in range(n)]
    blt = [sum(legs_g[c]["lt"][t] * w for c in coins) for t in range(n)]
    bst = [sum(legs_g[c]["st"][t] * w for c in coins) for t in range(n)]
    bf = seg(bfull, bturn, 0, n)
    bh2 = seg(bfull, bturn, h2a, n)["sharpe"]
    bl = seg(blong, blt, 0, n)
    bs = seg(bshort, bst, 0, n)
    bl["pnl_share"] = round(sum(blong) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bs["pnl_share"] = round(sum(bshort) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    res["basket"] = {"FULL": bf, "LONG": bl, "SHORT": bs, "H2_FULL_sharpe": bh2,
                     "coin_pnl_share": {c: round(sum(legs_g[c]["full"]) * w / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in coins}}
    bbase = [sum(legs_b[c]["full"][t] * w for c in coins) for t in range(n)]
    bbaseturn = [sum((legs_b[c]["lt"][t] + legs_b[c]["st"][t]) * w for c in coins) for t in range(n)]
    res["baseline"] = {"basket_FULL": seg(bbase, bbaseturn, 0, n),
                       "coin_FULL": {c: round(sum(legs_b[c]["full"]), 4) for c in coins}}
    res["status"] = "COMPLETE"
    res["conclusion"] = ("PENDING (P0-3 FAIL): H64 1h BB(192,2) outside-band entry gate "
                         "gated sh=%.3f vs base sh=%.3f; "
                         "diagnostic only, no adoption, no live change, live untouched." % (
                             bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"]))
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION gated sh=%.3f (base %.3f)" % (
        OUT, bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"]))


if __name__ == "__main__":
    main()
