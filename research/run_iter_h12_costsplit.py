"""H12 1h turnover decomposition (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows:
timestamp,open,high,low,close,volume,quote_volume,trades).

Per coin (executed signed position series, post cooldown/stops/vol/roll1):
  turnover split: entry_turn (flat->nonzero) + exit_turn (nonzero->flat)
    + flip_turn (long<->short, 2.0 per bar) == total turnover exactly.
  cost attribution (symmetric diagnostic accounting, H5-style):
    fee_cum  = sum(turn) * FEE * LEV
    slip_cum = sum(turn) * SLIP * LEV   (assumed 5bp, H13 ASSUMED_SLIP_BP)
    fund_cum = sum(pos) * FUND * LEV    (signed: long pays positive)
    fund_abs = sum(|pos|) * FUND * LEV  (symmetric |funding| cost view)
  per-coin cost share: share of basket total_cost_abs
    (fee + slip + fund_abs), plus turnover share.
  hold stats (H18 mirror): holding-run lengths of nonzero same-sign
    segments (median/mean/max/n_runs, entries).

Basket: equal-0.2 FULL + H2 (last 730 bars) aggregates.
Verdict PENDING (P0-3 FAIL); no adoption; live untouched.

Incremental dump: results/iter_H12_costsplit.json rewritten after EACH
unit (per-coin leg), status partial->final, so partial progress survives
kills. Smoke: ITER_H12_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
ITER_H12_OUT / ITER_H12_LOG override output paths.
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
assert abs(FUND - 0.0005) < 1e-12, "FUND lock broken: %r" % FUND
assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "FORMULA lock broken"

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
H2_LEN = 730
SLIP = 0.0005  # assumed 5bp, mirrors H13 ASSUMED_SLIP_BP
OUT = pathlib.Path(os.getenv("ITER_H12_OUT", "results/iter_H12_costsplit.json"))
LOG = pathlib.Path(os.getenv("ITER_H12_LOG", "logs/iter_H12_costsplit.log"))
SMOKE = os.getenv("ITER_H12_SMOKE") == "1"

_partial = {"status": "partial", "verdict": "PENDING"}


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def dump():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(_partial, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)


def scaled_spec(base):
    s = dict(base)
    for k in ("cd", "ts", "vw"):
        if s.get(k) is not None:
            s[k] = int(s[k]) * SCALE
    return s


SPECS = {c: scaled_spec(b) for c, b in BASE_SPECS.items()}


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common_grid(coins, cap=None):
    raw = {c: load1h(c) for c in coins}
    ts0 = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts0 &= set(r[0] for r in raw[c])
    ts_sorted = sorted(ts0)
    bars = {}
    for c in coins:
        m = {r[0]: r for r in raw[c]}
        bars[c] = [(m[t][1], m[t][2], m[t][3], m[t][4], m[t][5]) for t in ts_sorted]
    if cap is not None and len(ts_sorted) > cap:
        ts_sorted = ts_sorted[:cap]
        bars = {c: bars[c][:cap] for c in coins}
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
def leg_parts(raw, rt, sig, spec, fee=FEE, fund=FUND):
    """Mirror run_weight_modes leg_net, 1h-native. Returns dict of series."""
    bt = MemeBacktest(
        venue="aster", leverage=LEV, short_enabled=True,
        funding_override=fund, fee_override=fee,
        long_th=spec["lth"], short_th=spec["sth"],
        cooldown_bars=spec["cd"], bars_per_year=BPY,
        stop_loss=spec["sl"], time_stop=spec["ts"],
        vol_target=spec["vt"], vol_window=spec["vw"],
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
    pos = (lp - sp)[0].tolist()
    gross = ((lp - sp) * rt * bt.leverage)[0].tolist()
    turn_l = turn[0].tolist()
    fee_l = [t * fee * LEV for t in turn_l]
    slip_l = [t * SLIP * LEV for t in turn_l]
    fund_l = [p * fund * LEV for p in pos]
    fund_abs_l = [abs(p) * fund * LEV for p in pos]
    net_l = [g - f - s - d for g, f, s, d in zip(gross, fee_l, slip_l, fund_l)]
    return {"net": net_l, "turn": turn_l, "pos": pos, "gross": gross,
            "fee": fee_l, "slip": slip_l, "fund": fund_l, "fund_abs": fund_abs_l}


def split_turnover(pos):
    """Split per-bar turnover into entry/exit/flip components.

    Entry: flat->nonzero (turn 1.0). Exit: nonzero->flat (turn 1.0).
    Flip: long<->short direct (turn 2.0). Residual covers partial-size
    changes (none under vt None->1.0, kept for exactness).
    Bar 0 uses prev = pos[-1] to mirror the engine roll(1) wrap exactly.
    Returns (entry, exit, flip, resid) per-bar lists.
    """
    n = len(pos)
    entry = [0.0] * n
    exit_ = [0.0] * n
    flip = [0.0] * n
    resid = [0.0] * n
    # Mirror the engine turn series exactly: turn[t] = |pos[t]-pos[t-1]|
    # with roll(1) wrap, i.e. turn[0] = |pos[0]-pos[-1]|. Since roll1
    # forces pos[0] == 0, a nonzero pos[-1] makes turn[0] an exit here.
    prev = pos[-1] if n > 0 else 0.0
    for t in range(n):
        cur = pos[t]
        d = abs(cur - prev)
        if cur != 0.0 and prev == 0.0:
            entry[t] = d
        elif cur == 0.0 and prev != 0.0:
            exit_[t] = d
        elif cur != 0.0 and prev != 0.0 and (cur > 0) != (prev > 0):
            flip[t] = d
        elif d > 0:
            resid[t] = d
        prev = cur
    return entry, exit_, flip, resid


def holding_runs(pos):
    """Consecutive same-sign nonzero segments -> lengths."""
    runs = []
    cur = 0.0
    ln = 0
    for v in pos:
        s = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if s != 0.0 and s == cur:
            ln += 1
        elif s != 0.0:
            if ln > 0:
                runs.append(ln)
            cur = s
            ln = 1
        else:
            if ln > 0:
                runs.append(ln)
            cur = 0.0
            ln = 0
    if ln > 0:
        runs.append(ln)
    return runs


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    h = n // 2
    return float(s[h] if n % 2 else (s[h - 1] + s[h]) / 2.0)


def seg(net, turn, a, b):
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
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}
def coin_block(parts, a, b):
    """Cost/turnover decomposition for window [a,b)."""
    pos = parts["pos"]
    entry, exit_, flip, resid = split_turnover(pos)
    sel = lambda v: v[a:b]
    to = sum(sel(parts["turn"]))
    en = sum(sel(entry))
    ex = sum(sel(exit_))
    fl = sum(sel(flip))
    rs = sum(sel(resid))
    fee_c = sum(sel(parts["fee"]))
    slip_c = sum(sel(parts["slip"]))
    fund_c = sum(sel(parts["fund"]))
    fund_a = sum(sel(parts["fund_abs"]))
    tot_abs = fee_c + slip_c + fund_a
    gross_c = sum(sel(parts["gross"]))
    net_c = sum(sel(parts["net"]))
    runs = holding_runs(pos[a:b])
    hold = {"n_runs": len(runs),
            "median": round(median(runs), 2) if runs else 0.0,
            "mean": round(sum(runs) / len(runs), 2) if runs else 0.0,
            "max": max(runs) if runs else 0,
            "entries": sum(1 for t in range(a, b)
                           if pos[t] != 0.0 and (pos[t - 1] if t > 0 else 0.0) == 0.0)}
    seg_s = seg(parts["net"], parts["turn"], a, b)
    return {"seg": seg_s,
            "turnover": {"total": round(to, 4), "entry": round(en, 4),
                         "exit": round(ex, 4), "flip": round(fl, 4),
                         "resid": round(rs, 4),
                         "check_split_eq_total": round(en + ex + fl + rs - to, 6),
                         "share_entry": round(en / to, 4) if to > 0 else 0.0,
                         "share_exit": round(ex / to, 4) if to > 0 else 0.0,
                         "share_flip": round(fl / to, 4) if to > 0 else 0.0},
            "cost": {"fee_cum": round(fee_c, 4),
                     "slip_cum": round(slip_c, 4),
                     "fund_cum_signed": round(fund_c, 4),
                     "fund_cum_abs": round(fund_a, 4),
                     "total_abs": round(tot_abs, 4),
                     "gross_cum": round(gross_c, 4),
                     "net_cum": round(net_c, 4),
                     "share_fee": round(fee_c / tot_abs, 4) if tot_abs > 0 else 0.0,
                     "share_slip": round(slip_c / tot_abs, 4) if tot_abs > 0 else 0.0,
                     "share_fund_abs": round(fund_a / tot_abs, 4) if tot_abs > 0 else 0.0,
                     "fund_share_of_gross": (round(fund_c / gross_c, 4)
                                             if abs(gross_c) > 1e-12 else None),
                     "fund_share_of_net": (round(fund_c / net_c, 4)
                                           if abs(net_c) > 1e-12 else None)},
            "hold": hold}


def base_config(coins, n, h2a):
    return {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins},
        "scale_4h_to_1h": SCALE,
        "weights": {c: WEIGHT for c in coins},
        "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE,
        "slip_assumed": SLIP, "slip_note": "assumed 5bp mirrors H13 ASSUMED_SLIP_BP",
        "accounting": "symmetric H5-style: fee=turn*FEE*LEV slip=turn*SLIP*LEV fund_signed=pos*FUND*LEV fund_abs=|pos|*FUND*LEV",
        "grid": "1h", "grid_bars": n, "h2_len": H2_LEN, "h2_start": h2a,
        "bpy": BPY, "data_dir": "data/data_1y/1h", "coins": list(coins),
        "smoke": SMOKE,
        "note": "H12 1h turnover decomposition; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H12_costsplit start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    cap = 3000 if SMOKE else None
    ts, bars = common_grid(coins, cap=cap)
    n = len(ts)
    h2a = n - H2_LEN
    log("common 1h n=%d h2a=%d coins=%s smoke=%s" % (n, h2a, coins, SMOKE))
    assert n > 2000, n
    assert 0 < H2_LEN < n, (H2_LEN, n)
    cfg = base_config(coins, n, h2a)
    _partial.update({"status": "partial", "stage": "loaded", "config": cfg,
                     "per_coin": {}, "basket": {}})
    dump()
    parts = {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        parts[c] = leg_parts(raw, rt, sig, SPECS[c], FEE, FUND)
        full = coin_block(parts[c], 0, n)
        h2 = coin_block(parts[c], h2a, n)
        _partial["per_coin"][c] = {"FULL": full, "H2": h2}
        _partial["stage"] = "leg_%s" % c
        _partial["legs_done"] = sorted(parts)
        dump()
        log("leg %s FULL turn=%.4f entry=%.4f exit=%.4f flip=%.4f fee=%.4f slip=%.4f fund=%.4f fund_abs=%.4f" % (
            c, full["turnover"]["total"], full["turnover"]["entry"],
            full["turnover"]["exit"], full["turnover"]["flip"],
            full["cost"]["fee_cum"], full["cost"]["slip_cum"],
            full["cost"]["fund_cum_signed"], full["cost"]["fund_cum_abs"]))
    w = 1.0 / len(coins)
    net = [sum(parts[c]["net"][t] * w for c in coins) for t in range(n)]
    turn = [sum(parts[c]["turn"][t] * w for c in coins) for t in range(n)]
    basket = {"FULL": seg(net, turn, 0, n), "H2": seg(net, turn, h2a, n)}
    # basket turnover split: weighted sum of per-coin splits
    for label, a, b in (("FULL", 0, n), ("H2", h2a, n)):
        tot = sum(_partial["per_coin"][c][label]["turnover"]["total"] * w for c in coins)
        en = sum(_partial["per_coin"][c][label]["turnover"]["entry"] * w for c in coins)
        ex = sum(_partial["per_coin"][c][label]["turnover"]["exit"] * w for c in coins)
        fl = sum(_partial["per_coin"][c][label]["turnover"]["flip"] * w for c in coins)
        fee_c = sum(_partial["per_coin"][c][label]["cost"]["fee_cum"] * w for c in coins)
        slip_c = sum(_partial["per_coin"][c][label]["cost"]["slip_cum"] * w for c in coins)
        fund_c = sum(_partial["per_coin"][c][label]["cost"]["fund_cum_signed"] * w for c in coins)
        fund_a = sum(_partial["per_coin"][c][label]["cost"]["fund_cum_abs"] * w for c in coins)
        tot_abs = fee_c + slip_c + fund_a
        basket[label].update({
            "turnover_split": {"total": round(tot, 4), "entry": round(en, 4),
                               "exit": round(ex, 4), "flip": round(fl, 4),
                               "share_entry": round(en / tot, 4) if tot > 0 else 0.0,
                               "share_exit": round(ex / tot, 4) if tot > 0 else 0.0,
                               "share_flip": round(fl / tot, 4) if tot > 0 else 0.0},
            "cost": {"fee_cum": round(fee_c, 4), "slip_cum": round(slip_c, 4),
                     "fund_cum_signed": round(fund_c, 4),
                     "fund_cum_abs": round(fund_a, 4),
                     "total_abs": round(tot_abs, 4),
                     "share_fee": round(fee_c / tot_abs, 4) if tot_abs > 0 else 0.0,
                     "share_slip": round(slip_c / tot_abs, 4) if tot_abs > 0 else 0.0,
                     "share_fund_abs": round(fund_a / tot_abs, 4) if tot_abs > 0 else 0.0}})
    _partial["basket"] = basket
    _partial["stage"] = "basket"
    dump()
    # per-coin cost share of basket total_abs
    for label in ("FULL", "H2"):
        denom = basket[label]["cost"]["total_abs"]
        for c in coins:
            cc = _partial["per_coin"][c][label]["cost"]["total_abs"] * w
            _partial["per_coin"][c][label]["cost"]["share_of_basket_abs"] = (
                round(cc / denom, 4) if denom > 0 else 0.0)
            tc = basket[label]["turnover_split"]["total"]
            ct = _partial["per_coin"][c][label]["turnover"]["total"] * w
            _partial["per_coin"][c][label]["turnover"]["share_of_basket"] = (
                round(ct / tc, 4) if tc > 0 else 0.0)
    dump()
    f, h = basket["FULL"], basket["H2"]
    log("basket FULL sh=%.3f to=%.6f split=%s cost=%s | H2 sh=%.3f" % (
        f["sharpe"], f["turnover"], f["turnover_split"], f["cost"], h["sharpe"]))
    verdict = "PENDING"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H12 verdict PENDING; "
            "turnover/cost decomposition only, no adoption, live untouched.")
    _partial.update({"status": "final", "stage": "done",
                     "verdict": verdict, "decision": decision,
                     "decision_note": note,
                     "conclusion": ("H12 1h Top5 cost split FULL turn=%.4f "
                                    "(entry %.4f/exit %.4f/flip %.4f) cost_abs=%.4f "
                                    "(fee %.4f/slip %.4f/fund_abs %.4f) | H2 sh=%.3f. "
                                    "PENDING; no live change."
                                    % (f["turnover_split"]["total"],
                                       f["turnover_split"]["entry"],
                                       f["turnover_split"]["exit"],
                                       f["turnover_split"]["flip"],
                                       f["cost"]["total_abs"], f["cost"]["fee_cum"],
                                       f["cost"]["slip_cum"], f["cost"]["fund_cum_abs"],
                                       h["sharpe"]))})
    dump()
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
