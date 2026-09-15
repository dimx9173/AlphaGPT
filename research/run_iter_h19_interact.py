"""H19 1h parameter interaction (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05), so this step's conclusion is
"PENDING" and MUST NOT be used as adoption evidence. No adoption, no live
change, live chain untouched. Offline read-only: reads data/data_1y/1h/*.csv
only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), q0.3 long-mask, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

1h NATIVE grid (no 4h aggregation): data/data_1y/1h/{COIN}.csv direct
(~8760 rows). 4h-bar params scale x4 inside the engine:
cooldown_bars=cd*4, time_stop=ts*4, vol_window=vw*4. BPY=8760.
Stop-loss is a return fraction, unscaled. q is the quantile long-mask.

Two 2-way interaction grids (uniform overrides on ALL 5 coins, other
params stay locked):
  A sl_x_q: sl {None,0.03} x q {0.2,0.3}  (2x2 = 4 cells)
  B cd_x_ts: cd {6,12} x ts {12,24}       (2x2 = 4 cells)
Total 8 cells. Metric per cell: FULL sharpe heatmap (+mdd/turnover/
cum/final_x/ann/n, H2 sharpe). Interaction effect per grid =
difference-in-differences of FULL sharpe
(cell11-cell10-cell01+cell00). Report only, no adoption.

Outputs: results/iter_H19_interact.json (+ logs/iter_H19_interact.log).
Incremental: results JSON dumped after EACH cell (partial survives).

Smoke mode (for tests): ITER_H19_SMOKE=1 shrinks to sl {None} x
q {0.2,0.3} (2 cells) + cd {6} x ts {12,24} (2 cells).
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
    FORMULA,
    LOCKED_ATOM,
    LOCKED_APT,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
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
W = {c: 0.2 for c in COINS}
SL_GRID = [None, 0.03]
Q_GRID = [0.2, 0.3]
CD_GRID = [6, 12]
TS_GRID = [12, 24]
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units

OUT = pathlib.Path(os.getenv("ITER_H19_OUT", "results/iter_H19_interact.json"))
LOG = pathlib.Path(os.getenv("ITER_H19_LOG", "logs/iter_H19_interact.log"))

SMOKE = os.getenv("ITER_H19_SMOKE") == "1"
SMOKE_SL = [None]
SMOKE_Q = [0.2, 0.3]
SMOKE_CD = [6]
SMOKE_TS = [12, 24]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load1h(coin):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins):
    """Timestamp-intersected native 1h bars (no aggregation)."""
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars, closes = {}, {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [b[3] for b in bars[c]]
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
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3]
            for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    # quantile long-mask: keep top-q |signal| bars on the long leg
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net(raw, rt, sig, lth, sth, cd, sl, ts, vt, vw, q, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 1h-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=lth, short_th=sth,
                      cooldown_bars=cd * SCALE, bars_per_year=BPY,
                      stop_loss=sl, time_stop=ts * SCALE,
                      vol_target=vt, vol_window=vw * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, q)
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
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist()


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs, pk, md = 0.0, -1e18, 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def portfolio(legs, turns, n):
    net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    return net, turn


def dump_partial(payload):
    OUT.write_text(json.dumps(payload, indent=1))


def heatmap(rows, cols, cells):
    """FULL sharpe heatmap: sharpe[i][j] for rows[i] x cols[j]."""
    m = {(c["r"], c["c"]): c for c in cells}
    return {
        "rows": list(rows),
        "cols": list(cols),
        "sharpe": [[m[(r, c)]["FULL"]["sharpe"] for c in cols] for r in rows],
        "mdd": [[m[(r, c)]["FULL"]["mdd"] for c in cols] for r in rows],
        "turnover": [[m[(r, c)]["FULL"]["turnover"] for c in cols] for r in rows],
    }


def interaction(sh):
    """Difference-in-differences of a 2x2 FULL-sharpe heatmap."""
    if len(sh) != 2 or any(len(r) != 2 for r in sh):
        return None
    return round(sh[1][1] - sh[1][0] - sh[0][1] + sh[0][0], 3)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_h19 start\n")
    sl_grid = SMOKE_SL if SMOKE else SL_GRID
    q_grid = SMOKE_Q if SMOKE else Q_GRID
    cd_grid = SMOKE_CD if SMOKE else CD_GRID
    ts_grid = SMOKE_TS if SMOKE else TS_GRID
    bars, _closes = common1h(COINS)
    n = len(bars[COINS[0]])
    h2a = n // 2
    log("common 1h n=%d h2a=%d sl=%s q=%s cd=%s ts=%s smoke=%s"
        % (n, h2a, sl_grid, q_grid, cd_grid, ts_grid, SMOKE))
    assert n > 8000, "1h grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built (E10 FORMULA, locked)")

    def run_leg(c, sl, q, cd, ts):
        spec = BASE_SPECS[c]
        raw, rt, sg = mats[c]
        return leg_net(raw, rt, sg, spec["lth"], spec["sth"], cd,
                       sl, ts, spec["vt"], spec["vw"], q, FEE, FUND)

    def eval_cell(sl_map, q_map, cd_map, ts_map):
        legs, turns = {}, {}
        for c in COINS:
            legs[c], turns[c] = run_leg(c, sl_map[c], q_map[c], cd_map[c], ts_map[c])
        net, turn = portfolio(legs, turns, n)
        return net, turn, seg(net, turn, 0, n), seg(net, turn, h2a, n)["sharpe"]

    base_sl = {c: BASE_SPECS[c]["sl"] for c in COINS}
    base_q = {c: BASE_SPECS[c]["q"] for c in COINS}
    base_cd = {c: BASE_SPECS[c]["cd"] for c in COINS}
    base_ts = {c: BASE_SPECS[c]["ts"] for c in COINS}
    _, _, base_full, base_h2 = eval_cell(base_sl, base_q, base_cd, base_ts)
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f H2=%.3f" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_h2))

    def base_payload(sq_rows, ct_rows):
        return {
            "config": {
                "engine": "mirror run_weight_modes leg_net + quantile long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 1h native, cd/ts/vw x4",
                "formula": list(FORMULA),
                "basket": {c: dict(BASE_SPECS[c]) for c in COINS},
                "weights": dict(W),
                "sl_grid": list(sl_grid),
                "q_grid": list(q_grid),
                "cd_grid": list(cd_grid),
                "ts_grid": list(ts_grid),
                "ts_unit": "4h-bar, engine x4 internally",
                "grid": "sl_x_q + cd_x_ts 2-way interactions",
                "venue": "aster",
                "lev": LEV,
                "fund": FUND,
                "fee": FEE,
                "bpy": BPY,
                "scale": SCALE,
                "grid_bars": n,
                "h2_start": h2a,
                "smoke": SMOKE,
                "partial": True,
                "note": "sl 2 x q 2 + cd 2 x ts 2 = 8 uniform cells. E10 FORMULA untouched; live chain untouched; offline read-only.",
            },
            "base_FULL": base_full,
            "base_H2_sharpe": base_h2,
            "sl_x_q_rows": sq_rows,
            "cd_x_ts_rows": ct_rows,
            "verdict": "PENDING",
        }

    sq_rows, ct_rows = [], []

    def run_and_dump(tag, r, c, sl_map, q_map, cd_map, ts_map):
        _net, _turn, full, h2 = eval_cell(sl_map, q_map, cd_map, ts_map)
        row = {"r": r, "c": c, "FULL": full, "H2_sharpe": h2,
               "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
               "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4)}
        if tag == "sl_x_q":
            row["sl"], row["q"] = r, c
            sq_rows.append(row)
        else:
            row["cd"], row["ts"] = r, c
            ct_rows.append(row)
        log("%s r=%s c=%s sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f dmdd=%+.4f" % (
            tag, r, c, full["sharpe"], full["mdd"], full["final_x"],
            h2, full["turnover"], row["d_sharpe_vs_base"], row["d_mdd_vs_base"]))
        dump_partial(base_payload(list(sq_rows), list(ct_rows)))
        return row

    for sl in sl_grid:
        for q in q_grid:
            run_and_dump("sl_x_q", sl, q,
                         {c: sl for c in COINS}, {c: q for c in COINS},
                         dict(base_cd), dict(base_ts))
    for cd in cd_grid:
        for ts in ts_grid:
            run_and_dump("cd_x_ts", cd, ts,
                         dict(base_sl), dict(base_q),
                         {c: cd for c in COINS}, {c: ts for c in COINS})

    hm_sq = heatmap(sl_grid, q_grid, sq_rows)
    hm_ct = heatmap(cd_grid, ts_grid, ct_rows)
    best_sq = max(sq_rows, key=lambda r: r["FULL"]["sharpe"])
    best_ct = max(ct_rows, key=lambda r: r["FULL"]["sharpe"])
    log("best sl_x_q: sl=%s q=%s sh=%.3f | interact=%s" % (
        best_sq["sl"], best_sq["q"], best_sq["FULL"]["sharpe"], interaction(hm_sq["sharpe"])))
    log("best cd_x_ts: cd=%s ts=%s sh=%.3f | interact=%s" % (
        best_ct["cd"], best_ct["ts"], best_ct["FULL"]["sharpe"], interaction(hm_ct["sharpe"])))

    out = base_payload(sq_rows, ct_rows)
    out["config"]["partial"] = False
    out["heatmap_sl_x_q"] = hm_sq
    out["heatmap_cd_x_ts"] = hm_ct
    out["interaction_sl_x_q"] = interaction(hm_sq["sharpe"])
    out["interaction_cd_x_ts"] = interaction(hm_ct["sharpe"])
    out["best_sl_x_q"] = {k: best_sq[k] for k in ("sl", "q")}
    out["best_cd_x_ts"] = {k: best_ct[k] for k in ("cd", "ts")}
    out["verdict"] = "PENDING"
    out["conclusion"] = ("\u5f85\u5b9a (P0-3 FAIL): H19 1h slxq/cdxts 2-way interaction "
                         "heatmaps diagnostic only; best cells are shelf values, "
                         "no adoption, no live change, live untouched.")
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_full, "base_H2": base_h2,
                      "heatmap_sl_x_q": hm_sq, "heatmap_cd_x_ts": hm_ct,
                      "interaction_sl_x_q": out["interaction_sl_x_q"],
                      "interaction_cd_x_ts": out["interaction_cd_x_ts"],
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
