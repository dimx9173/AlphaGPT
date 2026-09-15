"""H17 1h max losing streak (Top5) -- diagnostic only.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native: reads
data/data_1y/1h directly (no aggregation); cd/ts/vw x4 (4h-bar units
-> 1h-bar units); BPY=8760. Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Streak def: a losing bar is leg/portfolio net < 0 (flat zero breaks the
run). Longest run = max bars, tie-break = more negative cum.
Recovery = bars from streak end to first cum-equity >= pre-streak peak
(peak over cum before run start, floored at 0.0); unrecovered if never.
Vol contrast: leg-net per-bar std annualized (x sqrt(BPY)) full-sample
vs streak-window; cross-coin Pearson(len vs full vol).

Output: results/iter_H17_streak.json with per-coin + portfolio longest
run (bars/days/cum/start/end) + recovery + streak-vs-vol.
Verdict PENDING (P0-3 FAIL), no adoption, live untouched.
Incremental dump: results JSON rewritten after each unit (partial survives).
Smoke (for tests): ITER_H17_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
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
HOURS_PER_DAY = 24.0
OUT = pathlib.Path(os.getenv("ITER_H17_OUT", "results/iter_H17_streak.json"))
LOG = pathlib.Path(os.getenv("ITER_H17_LOG", "logs/iter_h17_streak.log"))
SMOKE = os.getenv("ITER_H17_SMOKE") == "1"


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def dump(state):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))


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


def leg_net(raw, rt, sig, spec, fee, fund):
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
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist()


def losing_runs(net):
    """All maximal consecutive net<0 runs: [(start, end, cum)]."""
    runs = []
    s = None
    for t, x in enumerate(net):
        if x < 0.0:
            if s is None:
                s = t
        else:
            if s is not None:
                runs.append((s, t - 1, sum(net[s:t])))
                s = None
    if s is not None:
        runs.append((s, len(net) - 1, sum(net[s:])))
    return runs


def recovery_after(net, start, end):
    """Bars from streak end to first cum-equity >= pre-streak peak.

    cum[t] = sum(net[:t+1]); pre_peak = max(0, max cum before start).
    Returns (recovered, bars_or_None).
    """
    n = len(net)
    cum = []
    s = 0.0
    for x in net:
        s += x
        cum.append(s)
    pre = max([0.0] + cum[:start]) if start > 0 else 0.0
    for j in range(end + 1, n):
        if cum[j] >= pre:
            return True, j - end
    return False, None


def longest_losing_run(net, ts=None, bpy=BPY):
    """Longest losing run (bars/days/cum) + recovery + run counts."""
    n = len(net)
    runs = losing_runs(net)
    if not runs:
        return {
            "start": None, "end": None, "start_ts": None, "end_ts": None,
            "length_bars": 0, "length_days": 0.0, "cum": 0.0,
            "recovered": True, "recovery_bars": 0, "recovery_days": 0.0,
            "n_runs": 0, "mean_run_len": 0.0, "runner_up_bars": 0,
            "vol_full_ann": round(ann_vol(net, bpy), 6),
            "vol_streak_ann": 0.0, "vol_ratio": 0.0,
        }
    best = max(runs, key=lambda r: (r[1] - r[0] + 1, -r[2]))
    start, end, cum = best
    length = end - start + 1
    rec, rb = recovery_after(net, start, end)
    rest = [r for r in runs if r != best]
    runner = max((r[1] - r[0] + 1 for r in rest), default=0)
    return {
        "start": start, "end": end,
        "start_ts": ts[start] if ts is not None else None,
        "end_ts": ts[end] if ts is not None else None,
        "length_bars": length,
        "length_days": round(length / HOURS_PER_DAY, 4),
        "cum": round(cum, 6),
        "recovered": rec,
        "recovery_bars": rb if rec else None,
        "recovery_days": round(rb / HOURS_PER_DAY, 4) if rec else None,
        "n_runs": len(runs),
        "mean_run_len": round(sum(r[1] - r[0] + 1 for r in runs) / len(runs), 4),
        "runner_up_bars": runner,
        "vol_full_ann": round(ann_vol(net, bpy), 6),
        "vol_streak_ann": round(ann_vol(net[start:end + 1], bpy), 6),
        "vol_ratio": round(ann_vol(net[start:end + 1], bpy) / ann_vol(net, bpy), 4)
        if ann_vol(net, bpy) > 0 else 0.0,
    }


def ann_vol(xs, bpy=BPY):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(v) * math.sqrt(bpy) if v > 0 else 0.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(vx * vy)
    return cov / den if den > 1e-12 else 0.0


def leg_stats(net):
    n = len(net)
    m = sum(net) / n if n else 0.0
    v = sum((x - m) ** 2 for x in net) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = pk = md = 0.0
    for x in net:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    return {"n": n, "cum": round(sum(net), 4), "sharpe": round(sh, 3),
            "mdd": round(md, 4)}


def base_config(coins, n):
    return {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins},
        "scale_4h_to_1h": SCALE,
        "weights": {c: round(1.0 / len(coins), 6) for c in coins},
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "grid": "1h",
        "grid_bars": n,
        "bpy": BPY,
        "data_dir": "data/data_1y/1h",
        "coins": list(coins),
        "smoke": SMOKE,
        "streak_def": "losing bar = leg/portfolio net < 0 (flat zero breaks run); longest = max bars, tie-break more negative cum",
        "recovery_def": "bars from streak end to first cum-equity >= pre-streak peak (max cum before run start, floored at 0); else unrecovered",
        "vol_def": "per-bar net std annualized x sqrt(BPY); vol_ratio = streak-window / full-sample; cross-coin Pearson(max_len vs full vol)",
        "note": "H17 1h max losing streak autopsy; E10 FORMULA untouched; offline read-only",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H17_streak start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    log("common 1h n=%d coins=%s smoke=%s scale=x%d" % (n, coins, SMOKE, SCALE))
    assert n > 2000, n
    cfg = base_config(coins, n)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    legs = {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs[c] = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg,
              "legs_done": sorted(legs)})
        log("leg %s cum=%.4f" % (c, sum(legs[c])))
    streaks = {}
    for c in coins:
        streaks[c] = longest_losing_run(legs[c], ts)
        dump({"status": "partial", "stage": "streak_%s" % c, "config": cfg,
              "streaks_done": sorted(streaks)})
        s = streaks[c]
        log("%s longest=%d bars (%.2fd) cum=%.4f rec=%s" % (
            c, s["length_bars"], s["length_days"], s["cum"],
            s["recovery_bars"] if s["recovered"] else "never"))
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    pf = longest_losing_run(net, ts)
    dump({"status": "partial", "stage": "portfolio", "config": cfg,
          "streaks": streaks, "portfolio": pf})
    log("portfolio longest=%d bars (%.2fd) cum=%.4f rec=%s" % (
        pf["length_bars"], pf["length_days"], pf["cum"],
        pf["recovery_bars"] if pf["recovered"] else "never"))
    per_coin = {c: {"FULL": leg_stats(legs[c]), "streak": streaks[c]} for c in coins}
    lens = [streaks[c]["length_bars"] for c in coins]
    vols = [streaks[c]["vol_full_ann"] for c in coins]
    vol_contrast = {
        "per_coin": {c: {"vol_full_ann": streaks[c]["vol_full_ann"],
                         "vol_streak_ann": streaks[c]["vol_streak_ann"],
                         "vol_ratio": streaks[c]["vol_ratio"],
                         "max_len_bars": streaks[c]["length_bars"]} for c in coins},
        "portfolio": {"vol_full_ann": pf["vol_full_ann"],
                      "vol_streak_ann": pf["vol_streak_ann"],
                      "vol_ratio": pf["vol_ratio"],
                      "max_len_bars": pf["length_bars"]},
        "cross_coin_corr_len_vs_vol": round(pearson(lens, vols), 4),
        "rule": "ratio>1 means the worst streak ran hotter than average leg-net vol",
    }
    worst = max(coins, key=lambda c: streaks[c]["length_bars"])
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = ("P0-3 permutation FAIL (per-coin p>=0.05) => H17 verdict PENDING; "
            "streak autopsy only, no adoption, live untouched.")
    res = {"status": "final", "config": cfg, "per_coin": per_coin,
           "portfolio": {"FULL": leg_stats(net), "weights": dict(cfg["weights"]),
                         "streak": pf},
           "vol_contrast": vol_contrast,
           "worst_leg": worst,
           "verdict": verdict, "decision": decision, "decision_note": note,
           "conclusion": ("H17 1h Top5 max losing streak: portfolio %d bars (%.2fd) cum %.4f "
                          "recovery %s; worst leg %s %d bars cum %.4f; len-vs-vol corr %+.3f. "
                          "PENDING; diagnostic only, no adoption, live untouched."
                          % (pf["length_bars"], pf["length_days"], pf["cum"],
                             pf["recovery_bars"] if pf["recovered"] else "never",
                             worst, streaks[worst]["length_bars"], streaks[worst]["cum"],
                             vol_contrast["cross_coin_corr_len_vs_vol"]))}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))


if __name__ == "__main__":
    main()
