"""H2 1h weights (Top5). Offline read-only.

Engine mirror of research/run_weight_modes.py leg_net (ONLY engine reference):
  E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer;
  MemeBacktest venue=aster lev2 short_enabled fund0.0005;
  quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1.
1h native: data/data_1y/1h/{COIN}.csv direct (no aggregation),
  cd/ts/vw x4, BPY=8760.
Top5 locked specs (4h units, scaled x4 in SPEC): ETC 0.88/0.12/cd18/None/ts24,
  TRX 0.85/0.12/cd6/0.05/ts24, ATOM 0.85/0.15/cd6/0.05/ts24,
  APT 0.88/0.12/cd18/None/ts24, KAS 0.88/0.12/cd6/None/ts24, q0.3.
Arms: equal(0.2) vs invvol(trailing-60 realized-vol reciprocal, clip 0.10-0.35)
  vs invvol_cap(vol_target 0.35, clamp 0.25-2.0) vs meanvar(FULL max-sharpe,
  no shorts) vs risk-parity(FULL, long-only).
Per arm: FULL sharpe/mdd/final_x/turnover + fee2x FULL + 12fold median + H2.
Incremental: results/iter_H2_weights.json dumped after EACH arm.
"""
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import (
    FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX,
    LEV, FUND, FEE, FEE2X,
)

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
LOCKED_4H = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
             "APT": LOCKED_APT, "KAS": LOCKED_KAS}
SCALE = 4  # 4h-bar units -> 1h-bar units
BPY = 8760.0
VOL_W = 60  # trailing window (1h bars) for realized vol
WMIN, WMAX = 0.10, 0.35  # invvol box clip
VOL_TGT = 0.35  # invvol_cap portfolio vol target (annualized)
LEV_MIN, LEV_MAX = 0.25, 2.0  # invvol_cap lev-scale clamp
ARMS = ["equal", "invvol", "invvol_cap", "meanvar", "riskparity"]
OUT = pathlib.Path("results/iter_H2_weights.json")


def log(m):
    print(m, flush=True)


def scaled_spec(spec):
    out = dict(spec)
    for k in ("cd", "ts", "vw"):
        if out.get(k) is not None:
            out[k] = int(out[k]) * SCALE
    return out


SPECS = {c: scaled_spec(LOCKED_4H[c]) for c in COINS}


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"])) for r in rows]


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


def leg_net(raw, rt, sig, spec, fee, fund, lev_scale=1.0):
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BPY,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
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
    gross = (lp - sp) * rt * bt.leverage * lev_scale
    tx = turn * bt.base_fee * bt.leverage * lev_scale
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage * lev_scale
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), (lp - sp)[0].tolist()


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0
    pk = -1e18
    md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(m * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def fold12_median(net, turn):
    n = len(net)
    fn = n // 12
    ss = []
    for i in range(12):
        a = i * fn
        b = a + fn if i < 11 else n
        ss.append(seg(net, turn, a, b)["sharpe"])
    srt = sorted(ss)
    return {"sharpes": [round(x, 3) for x in ss],
            "median": round(srt[len(srt) // 2], 3),
            "mean": round(sum(ss) / len(ss), 3)}


def realized_vol_series(closes, window=VOL_W):
    """Trailing annualized realized vol per bar (simple returns, 1h grid)."""
    c = np.asarray(closes, dtype=float)
    r = np.zeros_like(c)
    r[1:] = np.where(c[:-1] != 0, (c[1:] - c[:-1]) / np.where(c[:-1] == 0, 1.0, c[:-1]), 0.0)
    n = len(c)
    out = np.zeros(n)
    c1 = np.cumsum(np.insert(r, 0, 0.0))
    c2 = np.cumsum(np.insert(r * r, 0, 0.0))
    for t in range(n):
        a = max(0, t - window + 1)
        m = t - a + 1
        if m < 3:
            out[t] = 0.0
            continue
        s = c1[t + 1] - c1[a]
        s2 = c2[t + 1] - c2[a]
        var = (s2 - s * s / m) / (m - 1)
        out[t] = math.sqrt(max(var, 0.0)) * math.sqrt(BPY)
    return out


def project_box_simplex(raw, lo=WMIN, hi=WMAX):
    """Project raw weights onto {sum=1, lo<=w<=hi} via water-filling."""
    coins = list(raw)
    n = len(coins)
    if not (n * lo <= 1.0 <= n * hi):
        return {c: 1.0 / n for c in coins}
    pinned = {}
    free = list(coins)
    remain = 1.0
    for _ in range(n + 1):
        if not free:
            break
        denom = sum(raw[c] for c in free)
        if denom <= 0:
            share = remain / len(free)
            for c in free:
                pinned[c] = share
            free = []
            break
        scale = remain / denom
        trial = {c: raw[c] * scale for c in free}
        over = [c for c in free if trial[c] > hi]
        under = [c for c in free if trial[c] < lo]
        if not over and not under:
            for c in free:
                pinned[c] = trial[c]
            free = []
            break
        if over and (not under or max(trial[c] - hi for c in over) >= max(lo - trial[c] for c in under)):
            c = max(over, key=lambda x: trial[x])
            pinned[c] = hi
            remain -= hi
            free.remove(c)
        else:
            c = min(under, key=lambda x: trial[x])
            pinned[c] = lo
            remain -= lo
            free.remove(c)
    else:
        return {c: 1.0 / n for c in coins}
    if free:
        denom = sum(raw[x] for x in free)
        if denom <= 0:
            share = remain / len(free)
            for c in free:
                pinned[c] = share
        else:
            for c in free:
                pinned[c] = raw[c] * (remain / denom)
    s = sum(pinned.values())
    return {c: pinned[c] / s for c in coins}


def weights_invvol(volmat):
    n = volmat.shape[1]
    Ws = []
    for t in range(n):
        if t < VOL_W:
            Ws.append({c: 0.2 for c in COINS})
            continue
        vols = {c: float(volmat[i, t]) for i, c in enumerate(COINS)}
        if all(v <= 1e-9 for v in vols.values()):
            Ws.append({c: 0.2 for c in COINS})
            continue
        inv = {c: 1.0 / max(vols[c], 1e-9) for c in COINS}
        tot = sum(inv.values())
        raw = {c: inv[c] / tot for c in COINS}
        Ws.append(project_box_simplex(raw))
    return Ws, [1.0] * n


def lev_series_invvol_cap(closes, Ws):
    """Per-bar lev scale targeting VOL_TGT portfolio vol (closes covariance)."""
    n = len(closes[COINS[0]])
    C = np.array([closes[c] for c in COINS], dtype=float)
    R = np.zeros_like(C)
    nz = C[:, :-1] != 0
    R[:, 1:] = np.where(nz, (C[:, 1:] - C[:, :-1]) / np.where(nz, C[:, :-1], 1.0), 0.0)
    Ls = []
    for t in range(n):
        if t < VOL_W:
            Ls.append(1.0)
            continue
        m = R[:, max(1, t - VOL_W + 1):t + 1]
        mm = m.shape[1]
        if mm < 20:
            Ls.append(1.0)
            continue
        w = np.array([Ws[t][c] for c in COINS])
        cov = np.cov(m)
        pv = math.sqrt(max(float(w @ cov @ w), 0.0)) * math.sqrt(BPY)
        ls = VOL_TGT / pv if pv > 1e-9 else 1.0
        Ls.append(round(min(max(ls, LEV_MIN), LEV_MAX), 4))
    return Ls


def max_sharpe_weights(legs):
    """FULL-sample max-sharpe, long-only (no shorts), sum=1."""
    from scipy.optimize import minimize
    X = np.array([legs[c] for c in COINS]).T
    mu = X.mean(axis=0)
    cov = np.cov(X.T) + np.eye(len(COINS)) * 1e-12

    def neg_sharpe(w):
        port = X @ w
        m = port.mean()
        v = port.var(ddof=1)
        if v <= 0:
            return 0.0
        return -(m / math.sqrt(v) * math.sqrt(BPY))

    x0 = np.full(len(COINS), 1.0 / len(COINS))
    try:
        r = minimize(neg_sharpe, x0, method="SLSQP",
                     bounds=[(0.0, 1.0)] * len(COINS),
                     constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
                     options={"maxiter": 1000, "ftol": 1e-12})
        w = np.clip(r.x, 0, 1)
        s = w.sum()
        w = w / s if s > 0 else x0
    except Exception:
        w = x0
    wr = np.round(w, 4)
    wr[np.argmax(wr)] += 1.0 - wr.sum()  # renormalize rounding residual
    return {c: float(wr[i]) for i, c in enumerate(COINS)}


def risk_parity_weights(legs):
    """FULL-sample risk-parity, long-only: equalize w_i*(Sigma w)_i."""
    from scipy.optimize import minimize
    X = np.array([legs[c] for c in COINS]).T
    cov = np.cov(X.T) + np.eye(len(COINS)) * 1e-12

    def obj(w):
        rc = w * (cov @ w)
        m = rc.mean()
        return float(np.sum((rc - m) ** 2))

    x0 = np.full(len(COINS), 1.0 / len(COINS))
    try:
        r = minimize(obj, x0, method="SLSQP",
                     bounds=[(0.0, 1.0)] * len(COINS),
                     constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
                     options={"maxiter": 1000, "ftol": 1e-14})
        w = np.clip(r.x, 0, 1)
        s = w.sum()
        w = w / s if s > 0 else x0
    except Exception:
        w = x0
    wr = np.round(w, 4)
    wr[np.argmax(wr)] += 1.0 - wr.sum()  # renormalize rounding residual
    return {c: float(wr[i]) for i, c in enumerate(COINS)}


def dump(arms_done, legs_fee=None, legs_fee2x=None, closes=None, n=0, h2a=0, final=False):
    res = {"config": {"engine": "mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1",
                      "formula": list(FORMULA),
                      "locked_4h": {c: dict(LOCKED_4H[c]) for c in COINS},
                      "specs_1h": {c: dict(SPECS[c]) for c in COINS},
                      "scale_4h_to_1h": SCALE, "grid": "1h", "grid_bars": n,
                      "bpy": BPY, "vol_window": VOL_W, "clip": [WMIN, WMAX],
                      "vol_target": VOL_TGT, "lev_clamp": [LEV_MIN, LEV_MAX],
                      "venue": "aster", "lev": LEV, "fund": FUND,
                      "fee": FEE, "fee2x": FEE2X,
                      "arms": ARMS, "note": "E10 FORMULA untouched; live default equal; offline read-only"},
           "arms": arms_done,
           "verdict": "PENDING_P03_FAIL", "decision": "KEEP_equal", "decision_note": "PENDING (P0-3 FAIL): diagnostic only, no adoption, live untouched.",
           "compare": {}}
    if final and arms_done and "equal" in arms_done:
        base = arms_done["equal"]["FULL"]
        comp = {}
        for arm in ARMS:
            if arm == "equal" or arm not in arms_done:
                continue
            f = arms_done[arm]["FULL"]
            beats = bool(f["sharpe"] > base["sharpe"] and f["mdd"] < base["mdd"]
                         and f["turnover"] < base["turnover"])
            comp[arm] = {"beats_equal_on_all3": beats,
                         "d_sharpe": round(f["sharpe"] - base["sharpe"], 3),
                         "d_mdd": round(f["mdd"] - base["mdd"], 4),
                         "d_turnover": round(f["turnover"] - base["turnover"], 6)}
        res["compare"] = comp
        winners = [a for a, v in comp.items() if v["beats_equal_on_all3"]]
        if winners:
            res["decision"] = "REVIEW_" + winners[0]
            res["decision_note"] = ("Challenger %s beats equal on sharpe AND mdd AND turnover; "
                                    "verdict stays PENDING pending review." % winners[0])
        else:
            res["decision_note"] = ("No challenger beats equal on sharpe AND mdd AND turnover; "
                                    "default stays equal.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("dumped %s arms=%s final=%s" % (OUT, list(arms_done), final))


def combine(legs, Ws, Ls):
    n = len(Ws)
    net = [sum(legs[c][t] * Ws[t][c] * Ls[t] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * Ws[t][c] * Ls[t] for c in COINS) for t in range(n)]
    return net, turn


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bars = {c: load1h(c) for c in COINS}
    n = min(len(b) for b in bars.values())
    for c in COINS:
        bars[c] = bars[c][:n]
    closes = {c: [b[3] for b in bars[c]] for c in COINS}
    h2a = n // 2
    log("1h native n=%d h2a=%d arms=%s" % (n, h2a, ARMS))
    mats = {c: build_sig(bars[c]) for c in COINS}
    global turns
    legs = {}
    turns = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs[c], turns[c], _ = leg_net(raw, rt, sg, SPECS[c], FEE, FUND, 1.0)
    legs2 = {}
    turns2 = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs2[c], turns2[c], _ = leg_net(raw, rt, sg, SPECS[c], FEE2X, FUND, 1.0)
    volmat = np.array([realized_vol_series(closes[c]) for c in COINS])
    arms_done = {}

    def run_arm(name, Ws, Ls):
        net, turn = combine(legs, Ws, Ls)
        net2, turn2 = combine(legs2, Ws, Ls)
        full = seg(net, turn, 0, n)
        h2 = seg(net, turn, h2a, n)
        fee2x = seg(net2, turn2, 0, n)
        f12 = fold12_median(net, turn)
        arms_done[name] = {"FULL": full, "H2": h2, "fee2x_FULL": fee2x,
                           "fold12": f12,
                           "w_final": {c: Ws[-1][c] for c in COINS},
                           "lev_final": Ls[-1],
                           "w_mean": {c: round(sum(Ws[t][c] for t in range(n)) / n, 4) for c in COINS},
                           "lev_mean": round(sum(Ls) / n, 4)}
        log("%s FULL sh=%.3f mdd=%.4f fx=%.4f to=%.6f | fee2x sh=%.3f | fold12 med=%.3f | H2 sh=%.3f | w=%s lev=%.3f" % (
            name, full["sharpe"], full["mdd"], full["final_x"], full["turnover"],
            fee2x["sharpe"], f12["median"], h2["sharpe"], arms_done[name]["w_final"], Ls[-1]))
        dump(arms_done, n=n, h2a=h2a, final=False)

    run_arm("equal", [{c: 0.2 for c in COINS} for _ in range(n)], [1.0] * n)
    Wi, Li = weights_invvol(volmat)
    run_arm("invvol", Wi, Li)
    Wc = [dict(w) for w in Wi]
    Lc = lev_series_invvol_cap(closes, Wc)
    run_arm("invvol_cap", Wc, Lc)
    Wm = max_sharpe_weights(legs)
    run_arm("meanvar", [dict(Wm) for _ in range(n)], [1.0] * n)
    Wr = risk_parity_weights(legs)
    run_arm("riskparity", [dict(Wr) for _ in range(n)], [1.0] * n)
    dump(arms_done, n=n, h2a=h2a, final=True)
    log("wrote %s verdict=PENDING_P03_FAIL" % OUT)


if __name__ == "__main__":
    main()
