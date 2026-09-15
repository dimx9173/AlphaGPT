"""H55 1h Stoch filter v2 (Top5): entry-only Stoch(56,3) %K>%D long / %K<%D short gate.

Mirror research/run_weight_modes.py leg_net + research/run_iter_z6_ls.py leg_split:
E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer;
MemeBacktest venue=aster lev2 short_enabled fund0.0005 fee0.0004;
quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1.
Static equal 0.2 Top5 weights. 1h native: cd/ts/vw x4, BPY=8760.

H55 addition: per-coin Stochastic(56,3) on 1h bars (56x1h = 56h window).
%K = SMA3(Fast%K_56), %D = SMA3(%K). The LONG leg is gated to bars with
%K>%D, the SHORT leg to bars with %K<%D. The gate is applied to lp/sp
right after the quantile mask and BEFORE the joint cooldown+stops
(entry-only: exits still flow through cooldown/stops), so
FULL == LONG + SHORT exactly. Warmup (first 60 bars): K=D=50.0 neutral
=> both legs closed.

Baseline (ungated, identical pipeline minus the Stoch gate) is reported for
contrast. Read-only, verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Output: results/iter_H55_stoch.json (incrementally dumped after each coin, so a
partial file survives interruption).
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
SCALE = 4
STOCH_WINDOW = 56  # 56 x 1h = 56h
STOCH_SMOOTH = 3
STOCH_WARMUP = STOCH_WINDOW + 2 * STOCH_SMOOTH - 2  # 60 bars neutral
OUT = pathlib.Path("results/iter_H55_stoch.json")
LOG = pathlib.Path("logs/iter_H55_stoch.log")

CONFIG = {
    "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; Stoch(56,3) entry gate on legs pre-cooldown (long K>D / short K<D); static equal 0.2 Top5; 1h native cd/ts/vw x4",
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
    "stoch": {
        "window": STOCH_WINDOW,
        "smooth": STOCH_SMOOTH,
        "window_hours": 56.0,
        "k_def": "%K=SMA3(Fast%K_56)",
        "d_def": "%D=SMA3(%K)",
        "long_gate": "K>D",
        "short_gate": "K<D",
        "warmup_value": 50.0,
        "warmup_bars": STOCH_WARMUP,
        "applied": "entry-only on legs pre-cooldown (post quantile mask)",
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


def stoch_kd(highs, lows, closes, window=STOCH_WINDOW, smooth=STOCH_SMOOTH):
    """Stochastic %K/%D over bars. First STOCH_WARMUP bars are 50.0/50.0 (neutral)."""
    n = len(closes)
    K = [50.0] * n
    D = [50.0] * n
    if n <= window:
        return K, D
    raw = [50.0] * n
    for t in range(window - 1, n):
        hh = max(highs[t - window + 1:t + 1])
        ll = min(lows[t - window + 1:t + 1])
        if hh <= ll:
            raw[t] = 50.0
        else:
            raw[t] = 100.0 * (closes[t] - ll) / (hh - ll)
    for t in range(window - 1 + smooth - 1, n):
        K[t] = sum(raw[t - smooth + 1:t + 1]) / smooth
    for t in range(window - 1 + 2 * (smooth - 1), n):
        D[t] = sum(K[t - smooth + 1:t + 1]) / smooth
    for t in range(min(STOCH_WARMUP, n)):
        K[t] = 50.0
        D[t] = 50.0
    return K, D


def leg_split(raw, rt, sig, spec, fee, fund, lev, kd=None):
    """Joint lp/sp pipeline split into LONG/SHORT legs after shared cooldown+stops.

    kd=None -> ungated baseline. Otherwise LONG gated to %K>%D, SHORT to %K<%D
    (masks applied post-quantile, pre-cooldown: entry-only gate).
    """
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    if kd is not None:
        K, D = kd
        rk = torch.tensor([K])
        rd = torch.tensor([D])
        lp = lp * (rk > rd).float()
        sp = sp * (rk < rd).float()
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
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H55_stoch start\n")
    bars, closes = common1h(COINS)
    n = len(bars["ETC"])
    log("common 1h native n=%d scale=x%d stoch=(%d,%d)" % (n, SCALE, STOCH_WINDOW, STOCH_SMOOTH))
    kd_map = {}
    for c in COINS:
        h = [b[1] for b in bars[c]]
        l = [b[2] for b in bars[c]]
        kd_map[c] = stoch_kd(h, l, closes[c], STOCH_WINDOW, STOCH_SMOOTH)
    mats = {c: build_sig(bars[c]) for c in COINS}
    res = {"config": dict(CONFIG, grid_bars=n), "coins": {}, "basket": {}, "baseline": {}, "verdict": "PENDING", "decision": "NO_ADOPTION",
           "decision_note": "P0-3 permutation FAIL => H55 verdict PENDING; Stoch-gate attribution only, nothing promoted, live untouched.",
           "conclusion": "PENDING: H55 1h Stoch(56,3) entry gate Top5; no adoption, live untouched.",
           "status": "PARTIAL"}
    legs_g, legs_b = {}, {}
    for c in COINS:
        raw, rt, sg = mats[c]
        kd = kd_map[c]
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, kd)
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
        K, D = kd
        n_lo = sum(1 for a, b in zip(K, D) if a > b)
        n_so = sum(1 for a, b in zip(K, D) if a < b)
        res["coins"][c] = {
            "FULL": f, "LONG": ls, "SHORT": ss,
            "BASE_FULL": seg(bf, [0.0] * n, 0, n),
            "stoch": {"mean_k": round(sum(K) / n, 3), "mean_d": round(sum(D) / n, 3),
                      "min_k": round(min(K), 3), "max_k": round(max(K), 3),
                      "min_d": round(min(D), 3), "max_d": round(max(D), 3),
                      "n_long_open": n_lo, "n_short_open": n_so,
                      "frac_long_open": round(n_lo / n, 4), "frac_short_open": round(n_so / n, 4),
                      "warmup_bars": STOCH_WARMUP},
        }
        log("%s gated cumF=%.4f cumL=%.4f cumS=%.4f baseF=%.4f tradesL=%d tradesS=%d kdL=%.3f kdS=%.3f" % (
            c, sum(fnet), sum(lnet), sum(snet), sum(bf), tL, tS, n_lo / n, n_so / n))
        dump(res)  # incremental: partial survives interruption
    bfull = [sum(legs_g[c]["full"][t] * W[c] for c in COINS) for t in range(n)]
    blong = [sum(legs_g[c]["long"][t] * W[c] for c in COINS) for t in range(n)]
    bshort = [sum(legs_g[c]["short"][t] * W[c] for c in COINS) for t in range(n)]
    bturn = [sum((legs_g[c]["lt"][t] + legs_g[c]["st"][t]) * W[c] for c in COINS) for t in range(n)]
    blt = [sum(legs_g[c]["lt"][t] * W[c] for c in COINS) for t in range(n)]
    bst = [sum(legs_g[c]["st"][t] * W[c] for c in COINS) for t in range(n)]
    bf = seg(bfull, bturn, 0, n)
    bl = seg(blong, blt, 0, n)
    bs = seg(bshort, bst, 0, n)
    bl["pnl_share"] = round(sum(blong) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bs["pnl_share"] = round(sum(bshort) / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0
    bf["trades"] = sum(res["coins"][c]["FULL"]["trades"] for c in COINS)
    bf["active_bars"] = sum(res["coins"][c]["FULL"]["active_bars"] for c in COINS)
    res["basket"] = {"FULL": bf, "LONG": bl, "SHORT": bs,
                     "coin_pnl_share": {c: round(sum(legs_g[c]["full"]) * W[c] / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in COINS}}
    bbase = [sum(legs_b[c]["full"][t] * W[c] for c in COINS) for t in range(n)]
    res["baseline"] = {"basket_FULL": seg(bbase, [0.0] * n, 0, n),
                       "coin_FULL": {c: round(sum(legs_b[c]["full"]), 4) for c in COINS}}
    res["status"] = "COMPLETE"
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION gated sh=%.3f (base %.3f) long_share=%.3f short_share=%.3f trades=%d" % (
        OUT, bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"], bl["pnl_share"], bs["pnl_share"], bf["trades"]))


if __name__ == "__main__":
    main()
