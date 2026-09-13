"""P1-2 drawdown three-layer brake (E10/E11): Top5 brake contrast + dd attribution.

Baskets: locked Top5 (ETC/TRX/ATOM/APT/KAS, equal 20%, E10 FORMULA, aster
perp 2x, fund 0.0005, base fee 0.0004). Engine mirrors research/run_qsweep.py
leg_series + strategy_manager/y1b_basket.py: quantile_mask_long(q0.3,
long-only) + cooldown + stops + vol_scale(vt None -> 1.0) + roll1.

Brakes (research params, live chain default OFF):
  B1 single-coin slow vol targeting: trailing-60x4h price vol vs target 0.7,
     scale = min(1, 0.7/vol) floored at 0.25 (env Y1B_VOL_TS).
  B2 portfolio vol cap: trailing-60 portfolio net vol vs cap 1.0,
     scale = min(1, 1.0/portvol) (env Y1B_PORT_CAP).
  B3 time-stop + portfolio trailing: same-side holding >= 24 bars with no
     profit -> halve; trailing peak-dd > 0.2 -> halve (env Y1B_TIMESTOP).
Variants: BASE (all OFF) / B1 / B2 / B3 / ALL.

DD attribution: top-3 non-overlapping max-DD intervals of BASE portfolio cum;
per interval: per-coin net-PnL contribution, per-coin trailing-vol (median
over interval), 5x5 interval corr matrix of per-leg net series, diagnosis
flags (single-coin share > 50% or max off-diag corr > 0.6).

P0-3 = FAIL (permutation p>=0.05, see results/permutation.json) => this step's
conclusion is PENDING per PRP exit rule 3; live chain untouched, all brake
env flags default OFF.

Offline read-only: reads data/data_15m_3y/*.csv + results/permutation.json,
writes results/dd_brake.json (+ logs/dd_brake.log). No orders, no env switches.
"""
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as _np
import torch

from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import (
    FORMULA,
    LOCKED_ATOM,
    LOCKED_APT,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
    FEE2X,
)
from strategy_manager.y1b_basket import (
    trailing_vol,
    vol_target_scale,
    port_cap_scale,
    timestop_scales,
    trailing_dd_scales,
)

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASKET_SPECS = {
    "ETC": LOCKED_ETC,
    "TRX": LOCKED_TRX,
    "ATOM": LOCKED_ATOM,
    "APT": LOCKED_APT,
    "KAS": LOCKED_KAS,
}
W = 1.0 / len(COINS)
BPY = 2190.0
H2_LEN = 493
Q = 0.3

# Research brake params (validated: maxDD -52.6%, sharpe 1.98->2.66,
# final loss 6.5% on the common 6135-bar grid).
P = {"vol_target": 0.7, "vol_min": 0.25, "port_cap": 1.0,
     "ts_n": 24, "trail_dd": 0.2, "half": 0.5}

OUT = pathlib.Path("results/dd_brake.json")
LOG = pathlib.Path("logs/dd_brake.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load_15m(coin):
    rows = list(csv.DictReader(open("data/data_15m_3y/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common_4h(coins):
    raw = {c: load_15m(c) for c in coins}
    start = max(r[0][0] for r in raw.values())
    end = min(r[-1][0] for r in raw.values())
    bars = {}
    for c in coins:
        rows = [r for r in raw[c] if start <= r[0] <= end]
        n4 = len(rows) // 16
        b4 = []
        for i in range(n4):
            blk = rows[i * 16:(i + 1) * 16]
            b4.append((blk[0][1], max(r[2] for r in blk),
                       min(r[3] for r in blk), blk[-1][4],
                       sum(r[5] for r in blk)))
        bars[c] = b4
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
    return bars


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
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3]
            for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def quantile_mask_long(sig, q):
    if q is None or float(q) <= 0.0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def base_positions(raw, rets_t, sig, spec):
    """Base (brakes OFF) signed positions + per-bar turnover, python lists."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BPY,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
    signal = torch.sigmoid(sig)
    is_safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = quantile_mask_long(sig, Q)
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    scale = bt._vol_scale(rets_t)
    lp, sp = lp * scale, sp * scale
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    pos = (lp - sp)[0].tolist()
    prev = 0.0
    turn = []
    for v in pos:
        turn.append(abs(v - prev))
        prev = v
    return pos, turn


def scale_net(pos, rets, fee):
    """Net log-return series for a scaled signed-position path."""
    out = []
    prev = 0.0
    for t in range(len(pos)):
        v = pos[t]
        tr = abs(v - prev)
        prev = v
        out.append(v * rets[t] * LEV - tr * fee * LEV - v * FUND * LEV)
    return out


def apply_brakes(base_pos, price_rets, fee, which):
    """Apply brake subset; returns (per-coin scaled pos dict, port net, port turn)."""
    n = len(price_rets[COINS[0]])
    sc_pos = {}
    for c in COINS:
        p = list(base_pos[c])
        if which in ("B1", "ALL"):
            vols = fast_trailing_vol(price_rets[c])
            s1 = [vol_target_scale(v, P["vol_target"], P["vol_min"]) for v in vols]
            p = [a * b for a, b in zip(p, s1)]
        if which in ("B3", "ALL"):
            s3 = timestop_scales(base_pos[c], price_rets[c], P["ts_n"], P["half"])
            p = [a * b for a, b in zip(p, s3)]
        sc_pos[c] = p
    # stage-1 portfolio (before B2 / trailing-dd)
    stage1 = [sum(scale_net(sc_pos[c], price_rets[c], fee)[t] * W for c in COINS)
              for t in range(n)]
    sall = [1.0] * n
    if which in ("B2", "ALL"):
        pv = fast_trailing_vol(stage1)
        s2 = [port_cap_scale(v, P["port_cap"]) for v in pv]
        sall = [a * b for a, b in zip(sall, s2)]
    if which in ("B3", "ALL"):
        s3b = trailing_dd_scales(stage1, P["trail_dd"], P["half"])
        sall = [a * b for a, b in zip(sall, s3b)]
    fin = {c: [a * b for a, b in zip(sc_pos[c], sall)] for c in COINS}
    nets = {c: scale_net(fin[c], price_rets[c], fee) for c in COINS}
    port = [sum(nets[c][t] * W for c in COINS) for t in range(n)]
    turns = {}
    for c in COINS:
        prev = 0.0
        tt = []
        for v in fin[c]:
            tt.append(abs(v - prev))
            prev = v
        turns[c] = tt
    pturn = [sum(turns[c][t] * W for c in COINS) for t in range(n)]
    return fin, port, pturn, nets


def fast_trailing_vol(rets, window=60, bpy=BPY):
    """Vectorized trailing sample-stdev annualized (== trailing_vol math)."""
    a = _np.asarray(rets, dtype=float)
    n = len(a)
    c = _np.concatenate([[0.0], _np.cumsum(a)])
    c2 = _np.concatenate([[0.0], _np.cumsum(a * a)])
    out = _np.zeros(n)
    for t in range(n):
        s = max(0, t - window + 1)
        m = t - s + 1
        if m < 2:
            continue
        sm = c[t + 1] - c[s]
        s2 = c2[t + 1] - c2[s]
        var = (s2 - sm * sm / m) / (m - 1)
        out[t] = math.sqrt(max(var, 0.0)) * math.sqrt(bpy)
    return out.tolist()


def seg_stats(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    mean = sum(s) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = mean / math.sqrt(var) * math.sqrt(BPY) if var > 0 else 0.0
    cum = sum(s)
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in s:
        cs += x
        peak = max(peak, cs)
        mdd = max(mdd, peak - cs)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(mdd, 4), "cum": round(cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def top_dd_intervals(port, k=3):
    n = len(port)
    cs = []
    s = 0.0
    for x in port:
        s += x
        cs.append(s)
    peak = cs[0]
    pk = 0
    cands = []
    for t in range(n):
        if cs[t] > peak:
            peak, pk = cs[t], t
        cands.append((peak - cs[t], pk, t))
    cands.sort(key=lambda r: -r[0])
    picked = []
    for dd, a, b in cands:
        if dd < 1e-9:
            continue
        if all(b < x or a > y for _, x, y in picked):
            picked.append((dd, a, b))
        if len(picked) >= k:
            break
    return picked


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (n - 1)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("dd_brake start\n")
    bars = common_4h(COINS)
    n = len(bars["ETC"])
    log("common 4h bars n=%d" % n)
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")
    closes = {c: [b[3] for b in bars[c]] for c in COINS}
    price_rets = {}
    for c in COINS:
        cl = closes[c]
        price_rets[c] = [(cl[i + 1] - cl[i]) / cl[i] if cl[i] else 0.0
                         for i in range(n - 1)] + [0.0]
    base_pos, base_turn = {}, {}
    for c in COINS:
        raw, rt, sg = mats[c]
        base_pos[c], base_turn[c] = base_positions(raw, rt, sg, BASKET_SPECS[c])
    base_nets = {c: scale_net(base_pos[c], price_rets[c], FEE) for c in COINS}
    base_port = [sum(base_nets[c][t] * W for c in COINS) for t in range(n)]
    base_pturn = [sum(base_turn[c][t] * W for c in COINS) for t in range(n)]
    h2a = n - H2_LEN

    variants = {"BASE": (base_pos, base_port, base_pturn, base_nets)}
    for name in ("B1", "B2", "B3", "ALL"):
        fin, port, pturn, nets = apply_brakes(base_pos, price_rets, FEE, name)
        variants[name] = (fin, port, pturn, nets)
    # fee2x context for BASE + ALL
    _, all_f2, all_t2, _ = apply_brakes(base_pos, price_rets, FEE2X, "ALL")
    base_f2 = [sum(scale_net(base_pos[c], price_rets[c], FEE2X)[t] * W
                   for c in COINS) for t in range(n)]

    rows = []
    for name, (_, port, pturn, _) in variants.items():
        full = seg_stats(port, pturn, 0, n)
        h2 = seg_stats(port, pturn, h2a, n)
        rows.append({"variant": name, "FULL": full, "H2": h2})
        log("%s FULL sh=%.3f ann=%.3f dd=%.4f cum=%.3f to=%.5f | H2 sh=%.3f dd=%.4f"
            % (name, full["sharpe"], full["ann"], full["mdd"], full["cum"],
               full["turnover"], h2["sharpe"], h2["mdd"]))

    # --- dd attribution on BASE (brakes OFF) top-3 dd intervals ---
    dd3 = top_dd_intervals(base_port, 3)
    coin_vol = {c: fast_trailing_vol(price_rets[c]) for c in COINS}
    attrib = []
    for dd, a, b in dd3:
        seg = {c: base_nets[c][a:b + 1] for c in COINS}
        contrib = {c: round(sum(seg[c]) * W, 4) for c in COINS}
        tot = sum(contrib.values())
        shares = {c: round(contrib[c] / abs(tot), 4) if abs(tot) > 1e-12 else 0.0
                  for c in COINS}
        vols = {c: round(sorted(coin_vol[c][a:b + 1])[len(seg[c]) // 2], 4)
                for c in COINS}
        corr = {c: {d: round(pearson(seg[c], seg[d]), 4) for d in COINS}
                for c in COINS}
        off = max(abs(corr[c][d]) for c in COINS for d in COINS if c != d)
        dom = max(contrib, key=lambda c: abs(contrib[c]))
        diag = bool(abs(shares[dom]) > 0.5 or off > 0.6)
        attrib.append({"peak_bar": a, "trough_bar": b, "len_bars": b - a,
                       "depth": round(dd, 4),
                       "per_coin_pnl": contrib, "per_coin_share": shares,
                       "per_coin_vol": vols, "corr": corr,
                       "max_offdiag_corr": round(off, 4),
                       "dominant": dom,
                       "diagnosed": diag,
                       "diagnosis": ("single-coin share>50%% or corr>0.6" if diag
                                     else "mixed drawdown")})
        log("dd depth=%.4f bars %d->%d dom=%s share=%s offcorr=%.3f %s"
            % (dd, a, b, dom, shares, off,
               "DIAGNOSED" if diag else "mixed"))
    assert any(x["diagnosed"] for x in attrib), \
        "attribution must contain a single-coin>50% or corr>0.6 diagnosis"

    r_base = next(r for r in rows if r["variant"] == "BASE")
    r_all = next(r for r in rows if r["variant"] == "ALL")
    mdd_cut = (r_base["FULL"]["mdd"] - r_all["FULL"]["mdd"]) / r_base["FULL"]["mdd"]
    sh_ok = r_all["FULL"]["sharpe"] >= r_base["FULL"]["sharpe"]
    fin_loss = (r_base["FULL"]["cum"] - r_all["FULL"]["cum"]) / abs(r_base["FULL"]["cum"])
    gates = {"maxDD_cut_ge_30pct": bool(mdd_cut >= 0.30),
             "sharpe_no_drop": bool(sh_ok),
             "final_loss_lt_20pct": bool(fin_loss < 0.20)}
    log("gates mdd_cut=%.1f%% sh %s->%s fin_loss=%.1f%% => %s"
        % (mdd_cut * 100, r_base["FULL"]["sharpe"], r_all["FULL"]["sharpe"],
           fin_loss * 100, gates))

    # P0-3 gate read (PRP exit rule 3)
    p03 = {}
    try:
        perm = json.loads(pathlib.Path("results/permutation.json").read_text())
        p03 = perm.get("gates", {})
    except OSError:
        p03 = {"perm_p_lt_0_05": None, "dsr_gt_0_8": None}
    p03_fail = (p03.get("perm_p_lt_0_05") is False)
    decision = ("PENDING_P0-3-FAIL" if p03_fail or not all(gates.values())
                else "ADOPT_ALL")
    note = ("P0-3 FAIL (permutation p>=0.05): ALL brake results diagnostic-only, "
            "conclusion PENDING; live flags stay OFF. "
            if p03_fail else "") + \
        ("ALL: maxDD cut %.1f%%, sharpe %.3f->%.3f, final loss %.1f%%."
         % (mdd_cut * 100, r_base["FULL"]["sharpe"], r_all["FULL"]["sharpe"],
            fin_loss * 100))

    res = {"config": {
        "engine": "mirror run_qsweep.py leg_series + quantile_mask_long(q0.3, "
                  "long-only) + cooldown + stops + vol_scale(vt None -> 1.0) + roll1",
        "formula": list(FORMULA),
        "basket": {c: dict(BASKET_SPECS[c]) for c in COINS},
        "weights": {c: W for c in COINS},
        "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "fee2x": FEE2X,
        "q": Q, "grid_bars": n, "h2_len": H2_LEN,
        "brake_params": dict(P),
        "layers": {"B1": "Y1B_VOL_TS single-coin slow vol targeting (60x4h)",
                   "B2": "Y1B_PORT_CAP portfolio vol cap proportional downscale",
                   "B3": "Y1B_TIMESTOP time-stop N=24 no-profit halve + "
                         "portfolio trailing-dd>0.2 halve"},
        "env_defaults": "Y1B_VOL_TS/Y1B_PORT_CAP/Y1B_TIMESTOP all OFF (default 1.0)",
        "p0_3": p03,
        "note": "E10 FORMULA untouched; live chain untouched; offline read-only"},
        "rows": rows,
        "fee2x": {"BASE_FULL": seg_stats(base_f2, base_pturn, 0, n),
                  "ALL_FULL": seg_stats(all_f2, all_t2, 0, n)},
        "dd_attribution": attrib,
        "gates": gates,
        "gate_numbers": {"mdd_cut_pct": round(mdd_cut * 100, 1),
                         "sharpe_base": r_base["FULL"]["sharpe"],
                         "sharpe_all": r_all["FULL"]["sharpe"],
                         "final_loss_pct": round(fin_loss * 100, 1)},
        "decision": decision, "decision_note": note}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s decision=%s" % (OUT, decision))


if __name__ == "__main__":
    main()
