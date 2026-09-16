"""H15 1h vol-regime split (Top5): low/mid/high trailing-60-vol terciles.

Engine mirrors research/run_iter_h1_base.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native:
cd/ts/vw x4 (4h-bar units -> 1h-bar units); BPY=8760. Equal 0.2 weights.
Top5 locked specs ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Regimes: per-coin trailing-60 realized vol (annualized, BPY=8760) terciles
(low/mid/high by 1/3-2/3 value quantiles over the full grid). Per-coin
leg sharpe/ann/mdd/cum/turnover per regime + FULL(all bars, with trades).
Basket FULL per regime uses the cross-coin mean trailing-vol tercile.
Vol-target payoff recheck: rerun legs with MemeBacktest vol_target in
{0.006, 0.012} (vw=48 = 12x4) vs base vt None, FULL compare.

Output: results/iter_H15_volreg.json, verdict PENDING (P0-3 FAIL),
no adoption, live untouched.
Incremental dump: results JSON rewritten after each unit (partial survives).
Smoke (for tests): ITER_H15_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
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
    LEV, FUND, FEE, FEE2X,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(LOCKED_ETC),
    "TRX": dict(LOCKED_TRX),
    "ATOM": dict(LOCKED_ATOM),
    "APT": dict(LOCKED_APT),
    "KAS": dict(LOCKED_KAS),
}
BPY = 8760.0
SCALE = 4
WEIGHT = 0.2
VOL_W = 60
VT_GRID = [0.006, 0.012]
VT_VW = 12 * SCALE
REGIMES = ("low", "mid", "high")
OUT = pathlib.Path(os.getenv("ITER_H15_OUT", "results/iter_H15_volreg.json"))
LOG = pathlib.Path(os.getenv("ITER_H15_LOG", "logs/iter_H15_volreg.log"))
SMOKE = os.getenv("ITER_H15_SMOKE") == "1"


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def scaled_spec(base):
    s = dict(base)
    for k in ("cd", "ts", "vw"):
        if s.get(k) is not None:
            s[k] = int(s[k]) * SCALE
    return s


SPECS = {c: scaled_spec(b) for c, b in BASE_SPECS.items()}


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [
        (int(r["timestamp"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["volume"]))
        for r in rows
    ]


def common_grid(coins):
    raw = {c: load1h(c) for c in coins}
    ts0 = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts0 &= set(r[0] for r in raw[c])
    ts_sorted = sorted(ts0)
    bars = {}
    for c in coins:
        m = {r[0]: r for r in raw[c]}
        bars[c] = [(m[t][1], m[t][2], m[t][3], m[t][4], m[t][5]) for t in ts_sorted]
    return ts_sorted, bars


def build_sig(bars):
    n = len(bars)
    raw = {
        "open": torch.tensor([[b[0] for b in bars]]),
        "high": torch.tensor([[b[1] for b in bars]]),
        "low": torch.tensor([[b[2] for b in bars]]),
        "close": torch.tensor([[b[3] for b in bars]]),
        "volume": torch.tensor([[b[4] for b in bars]]),
        "liquidity": torch.full((1, n), 1e7),
        "fdv": torch.full((1, n), 1e8),
    }
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


def leg_net(raw, rt, sig, spec, fee, fund, vt=None, vw=None):
    bt = MemeBacktest(
        venue="aster", leverage=LEV, short_enabled=True,
        funding_override=fund, fee_override=fee,
        long_th=spec["lth"], short_th=spec["sth"],
        cooldown_bars=spec["cd"], bars_per_year=BPY,
        stop_loss=spec["sl"], time_stop=spec["ts"],
        vol_target=spec["vt"] if vt is None else vt,
        vol_window=spec["vw"] if vw is None else vw,
    )
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), (lp - sp)[0].tolist()


def trail_vol_series(closes, window=VOL_W, bpy=BPY):
    """Trailing annualized realized vol of simple returns, per bar."""
    vols = []
    for t in range(len(closes)):
        win = closes[max(0, t - window):t + 1]
        r = [(win[i + 1] - win[i]) / win[i] for i in range(len(win) - 1) if win[i]]
        if len(r) < 2:
            vols.append(0.0)
            continue
        m = sum(r) / len(r)
        var = sum((x - m) ** 2 for x in r) / (len(r) - 1)
        vols.append(math.sqrt(max(var, 0.0)) * math.sqrt(bpy))
    return vols


def tercile_bounds(v):
    s = sorted(v)
    n = len(s)
    return s[n // 3], s[2 * n // 3]


def assign_regime(v, t1, t2):
    return [0 if x <= t1 else (1 if x <= t2 else 2) for x in v]


def count_trades(pos, a, b):
    entries = flips = exits = 0
    prev = pos[a - 1] if a > 0 else 0.0
    for t in range(a, b):
        cur = pos[t]
        if cur != 0.0 and prev == 0.0:
            entries += 1
        elif cur == 0.0 and prev != 0.0:
            exits += 1
        elif cur != 0.0 and prev != 0.0 and (cur > 0) != (prev > 0):
            flips += 1
        prev = cur
    return {"trades": entries + flips, "entries": entries,
            "flips": flips, "exits": exits}


def seg(net, turn, pos, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    out = {
        "sharpe": round(sh, 3),
        "ann": round(cum / n * BPY, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(cum, 4),
        "final_x": round(1.0 + cum, 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }
    out.update(count_trades(pos, a, b))
    return out


def seg_idx(net, turn, idx):
    """Regime-subset stats over non-contiguous index list (no trade counts)."""
    s = [net[i] for i in idx]
    t = [turn[i] for i in idx]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {
        "sharpe": round(sh, 3),
        "ann": round(cum / n * BPY, 4) if n else 0.0,
        "mdd": round(md, 4),
        "cum": round(cum, 4),
        "n": n,
        "turnover": round(sum(t) / n, 6) if n else 0.0,
    }


def dump(state):
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def base_config(coins, n):
    return {
        "engine": "mirror run_iter_h1_base leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins},
        "scale_4h_to_1h": SCALE,
        "weights": {c: WEIGHT for c in coins},
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "fee2x": FEE2X,
        "grid": "1h",
        "grid_bars": n,
        "bpy": BPY,
        "data_dir": "data/data_1y/1h",
        "coins": list(coins),
        "smoke": SMOKE,
        "vol_window": VOL_W,
        "vol_bpy": BPY,
        "regimes": list(REGIMES),
        "regime_def": "per-coin trailing-60 realized vol (annualized) terciles low/mid/high by 1/3-2/3 value quantiles; basket regime uses cross-coin mean trailing vol",
        "vt_grid": list(VT_GRID),
        "vt_vw_1h": VT_VW,
        "vt_note": "vol-target payoff recheck reruns legs with MemeBacktest vol_target=vt (clamp 0.2-2.0 on trailing price-vol) vs base vt None",
        "note": "H15 1h vol-regime split; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H15_volreg start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    log("common 1h n=%d coins=%s smoke=%s scale=x%d vol_w=%d" % (n, coins, SMOKE, SCALE, VOL_W))
    assert n > 2000, n
    cfg = base_config(coins, n)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    closes = {c: [b[3] for b in bars[c]] for c in coins}
    vols = {c: trail_vol_series(closes[c]) for c in coins}
    bounds = {}
    regimes = {}
    for c in coins:
        t1, t2 = tercile_bounds(vols[c])
        bounds[c] = {"t1": round(t1, 6), "t2": round(t2, 6)}
        regimes[c] = assign_regime(vols[c], t1, t2)
        cnt = [sum(1 for r in regimes[c] if r == k) for k in range(3)]
        assert all(k > 0 for k in cnt), (c, cnt)
        log("vol %s t1=%.4f t2=%.4f counts=%s" % (c, t1, t2, cnt))
    mean_vol = [sum(vols[c][t] for c in coins) / len(coins) for t in range(n)]
    bt1, bt2 = tercile_bounds(mean_vol)
    basket_reg = assign_regime(mean_vol, bt1, bt2)
    bcnt = [sum(1 for r in basket_reg if r == k) for k in range(3)]
    assert all(k > 0 for k in bcnt), bcnt
    log("basket mean-vol t1=%.4f t2=%.4f counts=%s" % (bt1, bt2, bcnt))
    dump({"status": "partial", "stage": "regimes", "config": cfg,
          "vol_bounds": bounds,
          "basket_vol_bounds": {"t1": round(bt1, 6), "t2": round(bt2, 6)}})
    legs, turns, poss = {}, {}, {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs[c], turns[c], poss[c] = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg,
              "legs_done": sorted(legs)})
        log("leg %s done" % c)
    per_coin = {}
    for c in coins:
        idx = {k: [t for t in range(n) if regimes[c][t] == k] for k in range(3)}
        d = {REGIMES[k]: seg_idx(legs[c], turns[c], idx[k]) for k in range(3)}
        d["FULL"] = seg(legs[c], turns[c], poss[c], 0, n)
        per_coin[c] = d
        log("%s " % c + " ".join(
            "%s sh=%.3f ann=%.4f mdd=%.4f n=%d" % (REGIMES[k], d[REGIMES[k]]["sharpe"],
             d[REGIMES[k]]["ann"], d[REGIMES[k]]["mdd"], d[REGIMES[k]]["n"]) for k in range(3)))
    dump({"status": "partial", "stage": "per_coin_regimes", "config": cfg,
          "vol_bounds": bounds, "per_coin": per_coin})
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    pos = [sum(poss[c][t] * w for c in coins) for t in range(n)]
    bidx = {k: [t for t in range(n) if basket_reg[t] == k] for k in range(3)}
    basket = {REGIMES[k]: seg_idx(net, turn, bidx[k]) for k in range(3)}
    basket["FULL"] = seg(net, turn, pos, 0, n)
    for k in ("FULL",):
        pass
    f = basket["FULL"]
    log("basket FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f trades=%d | " % (
        f["sharpe"], f["mdd"], f["final_x"], f["turnover"], f["trades"]) + " ".join(
        "%s sh=%.3f" % (REGIMES[k], basket[REGIMES[k]]["sharpe"]) for k in range(3)))
    dump({"status": "partial", "stage": "basket_regimes", "config": cfg,
          "vol_bounds": bounds,
          "basket_vol_bounds": {"t1": round(bt1, 6), "t2": round(bt2, 6)},
          "per_coin": per_coin, "basket": basket})
    vt_recheck = {"base_FULL_sharpe": f["sharpe"], "rows": {}}
    for vt in VT_GRID:
        l2, tu2, po2 = {}, {}, {}
        for c in coins:
            raw, rt, sig = build_sig(bars[c])
            l2[c], tu2[c], po2[c] = leg_net(raw, rt, sig, SPECS[c], FEE, FUND, vt=vt, vw=VT_VW)
        n2 = [sum(l2[c][t] * w for c in coins) for t in range(n)]
        u2 = [sum(tu2[c][t] * w for c in coins) for t in range(n)]
        p2 = [sum(po2[c][t] * w for c in coins) for t in range(n)]
        full2 = seg(n2, u2, p2, 0, n)
        per2 = {c: seg(l2[c], tu2[c], po2[c], 0, n)["sharpe"] for c in coins}
        vt_recheck["rows"][str(vt)] = {
            "FULL": full2,
            "gap_FULL_sharpe": round(full2["sharpe"] - f["sharpe"], 3),
            "per_coin_FULL_sharpe": per2,
        }
        log("vt=%.3f FULL sh=%.3f gap=%+.3f mdd=%.4f to=%.6f" % (
            vt, full2["sharpe"], vt_recheck["rows"][str(vt)]["gap_FULL_sharpe"],
            full2["mdd"], full2["turnover"]))
        dump({"status": "partial", "stage": "vt_%s" % vt, "config": cfg,
              "vol_bounds": bounds, "per_coin": per_coin, "basket": basket,
              "vt_recheck": vt_recheck})
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H15 verdict PENDING; "
            "vol-regime contrast only, no adoption, live untouched.")
    reg_sh = {c: {r: per_coin[c][r]["sharpe"] for r in REGIMES} for c in coins}
    b_sh = {r: basket[r]["sharpe"] for r in REGIMES}
    res = {"status": "final", "config": cfg, "vol_bounds": bounds,
           "basket_vol_bounds": {"t1": round(bt1, 6), "t2": round(bt2, 6)},
           "per_coin": per_coin, "basket": basket, "vt_recheck": vt_recheck,
           "verdict": verdict, "decision": decision, "decision_note": note,
           "conclusion": ("H15 1h vol-tercile per-coin sharpe %s | basket regimes %s | "
                          "vt recheck gaps %s. PENDING; no live change."
                          % (reg_sh, b_sh,
                             {k: vt_recheck["rows"][k]["gap_FULL_sharpe"] for k in vt_recheck["rows"]}))}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
