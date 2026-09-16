"""V3 15m RSI filter (Top5): long-only RSI(56)>50 / short-only RSI<50 gate on legs.

Mirror research/run_weight_modes.py leg_net + research/run_iter_z6_ls.py leg_split:
E10 FORMULA via StackVM+FeatureEngineer; MemeBacktest venue=aster lev2
short_enabled fund=FUND fee=FEE; quantile q0.3 long-only + cooldown + stops +
vol_scale(vt None->1.0) + roll1. Static equal 0.2 Top5 weights.
15m native: cd/ts/vw x16, BPY=35040.

V3 addition: per-coin Wilder RSI(56) on 15m closes (56x15m = 14h). The LONG leg
is gated to bars with RSI>50, the SHORT leg to bars with RSI<50. The gate is
applied to lp/sp right after the quantile mask and BEFORE the joint
cooldown+stops, so FULL == LONG + SHORT exactly. Warmup (first 56 bars):
RSI=50.0 neutral => both legs closed.

Baseline (ungated, identical pipeline minus the RSI gate) is reported for
contrast. Read-only, verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Output: results/iter_V3_rsi.json (incrementally dumped after each coin, so a
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
BPY = 35040.0
SCALE = 16
RSI_WINDOW = 56  # 56 x 15m = 14h
RSI_TH = 50.0
OUT = pathlib.Path("results/iter_V3_rsi.json")
LOG = pathlib.Path("logs/iter_V3_rsi.log")

CONFIG = {
    "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; RSI(56) gate on legs pre-cooldown (long RSI>50 / short RSI<50); static equal 0.2 Top5; 15m native cd/ts/vw x16",
    "formula": list(FORMULA),
    "basket": {c: dict(SPECS[c]) for c in COINS},
    "weights": dict(W),
    "venue": "aster",
    "lev": LEV,
    "fund": FUND,
    "fee": FEE,
    "bpy": BPY,
    "grid": "15m",
    "scale": SCALE,
    "rsi": {
        "window": RSI_WINDOW,
        "window_hours": 14.0,
        "method": "Wilder",
        "long_gate": "RSI>50",
        "short_gate": "RSI<50",
        "threshold": RSI_TH,
        "warmup_value": 50.0,
        "warmup_bars": RSI_WINDOW,
        "applied": "on legs pre-cooldown (post quantile mask)",
    },
    "note": "E10 FORMULA untouched; live default untouched; offline read-only",
}


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]


def common15m(coins):
    raw = {c: load15m(c) for c in coins}
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


def wilder_rsi(closes, window=RSI_WINDOW):
    """Wilder RSI over closes. First `window` bars are 50.0 (neutral, both gates closed)."""
    n = len(closes)
    rsi = [50.0] * n
    if n <= window:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains[i] = d
        elif d < 0:
            losses[i] = -d

    def _to_rsi(ag, al):
        if al == 0.0 and ag == 0.0:
            return 50.0
        if al == 0.0:
            return 100.0
        if ag == 0.0:
            return 0.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    ag = sum(gains[1:window + 1]) / window
    al = sum(losses[1:window + 1]) / window
    rsi[window] = _to_rsi(ag, al)
    for i in range(window + 1, n):
        ag = (ag * (window - 1) + gains[i]) / window
        al = (al * (window - 1) + losses[i]) / window
        rsi[i] = _to_rsi(ag, al)
    return rsi


def leg_split(raw, rt, sig, spec, fee, fund, lev, rsi=None):
    """Joint lp/sp pipeline split into LONG/SHORT legs after shared cooldown+stops.

    rsi=None -> ungated baseline. Otherwise LONG gated to RSI>50, SHORT to RSI<50
    (masks applied post-quantile, pre-cooldown).
    """
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    if rsi is not None:
        r = torch.tensor([rsi])
        lp = lp * (r > RSI_TH).float()
        sp = sp * (r < RSI_TH).float()
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
    open(LOG, "w").write("iter_V3_rsi start\n")
    bars, closes = common15m(COINS)
    n = len(bars["ETC"])
    log("common 15m native n=%d scale=x%d rsi=%d" % (n, SCALE, RSI_WINDOW))
    rsi_map = {c: wilder_rsi(closes[c], RSI_WINDOW) for c in COINS}
    mats = {c: build_sig(bars[c]) for c in COINS}
    res = {"config": dict(CONFIG, grid_bars=n), "coins": {}, "basket": {}, "baseline": {}, "verdict": "PENDING", "decision": "NO_ADOPTION",
           "decision_note": "P0-3 permutation FAIL => V3 verdict PENDING; RSI-gate attribution only, nothing promoted, live untouched.",
           "status": "PARTIAL"}
    legs_g, legs_b = {}, {}
    for c in COINS:
        raw, rt, sg = mats[c]
        rsi = rsi_map[c]
        lnet, snet, fnet, lt, st, lp, sp = leg_split(raw, rt, sg, SPECS[c], FEE, FUND, LEV, rsi)
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
        n_lo = sum(1 for v in rsi if v > RSI_TH)
        n_so = sum(1 for v in rsi if v < RSI_TH)
        res["coins"][c] = {
            "FULL": f, "LONG": ls, "SHORT": ss,
            "BASE_FULL": seg(bf_, [a + b for a, b in zip(blt_, bst_)], 0, n),
            "rsi": {"mean": round(sum(rsi) / n, 3), "min": round(min(rsi), 3), "max": round(max(rsi), 3),
                    "n_long_open": n_lo, "n_short_open": n_so,
                    "frac_long_open": round(n_lo / n, 4), "frac_short_open": round(n_so / n, 4)},
        }
        log("%s gated cumF=%.4f cumL=%.4f cumS=%.4f baseF=%.4f tradesL=%d tradesS=%d rsiL=%.3f rsiS=%.3f" % (
            c, sum(fnet), sum(lnet), sum(snet), sum(bf_), ls["trades"], ss["trades"], n_lo / n, n_so / n))
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
    res["basket"] = {"FULL": bf, "LONG": bl, "SHORT": bs,
                     "coin_pnl_share": {c: round(sum(legs_g[c]["full"]) * W[c] / sum(bfull), 4) if abs(sum(bfull)) > 1e-12 else 0.0 for c in COINS}}
    bbase = [sum(legs_b[c]["full"][t] * W[c] for c in COINS) for t in range(n)]
    bbaseturn = [sum((legs_b[c]["lt"][t] + legs_b[c]["st"][t]) * W[c] for c in COINS) for t in range(n)]
    res["baseline"] = {"basket_FULL": seg(bbase, bbaseturn, 0, n),
                       "coin_FULL": {c: round(sum(legs_b[c]["full"]), 4) for c in COINS}}
    res["status"] = "COMPLETE"
    dump(res)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION gated sh=%.3f (base %.3f) long_share=%.3f short_share=%.3f" % (
        OUT, bf["sharpe"], res["baseline"]["basket_FULL"]["sharpe"], bl["pnl_share"], bs["pnl_share"]))


if __name__ == "__main__":
    main()
