# H16 1h skew/kurtosis + tail (Top5). Per-leg skew/kurt + FULL tail + worst-day attribution.
# Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA via StackVM+FeatureEngineer;
# MemeBacktest venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only + cooldown
# + stops + vol_scale(vt None->1.0) + roll1). 1h native: data/data_1y/1h; cd/ts/vw x4; BPY=8760.
# Equal 0.2 weights. Top5 locked specs ETC/TRX/ATOM/APT/KAS q0.3. Offline read-only;
# verdict PENDING (P0-3 FAIL), no adoption, live untouched. Incremental dump after each unit.
# Smoke: ITER_H16_SMOKE=1 -> coins {ETC,TRX}, first 480 bars.
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
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {"ETC": dict(LOCKED_ETC), "TRX": dict(LOCKED_TRX), "ATOM": dict(LOCKED_ATOM), "APT": dict(LOCKED_APT), "KAS": dict(LOCKED_KAS)}
BPY = 8760.0
SCALE = 4
WEIGHT = 0.2
DAY_BARS = 24
OUT = pathlib.Path(os.getenv("ITER_H16_OUT", "results/iter_H16_tail.json"))
LOG = pathlib.Path(os.getenv("ITER_H16_LOG", "logs/iter_H16_tail.log"))
SMOKE = os.getenv("ITER_H16_SMOKE") == "1"

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
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

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

def leg_net(raw, rt, sig, spec, fee, fund):
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=spec["cd"], bars_per_year=BPY, stop_loss=spec["sl"], time_stop=spec["ts"], vol_target=spec["vt"], vol_window=spec["vw"])
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

def skew_kurt(xs):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    m = sum(xs) / n
    sd2 = sum((x - m) ** 2 for x in xs) / n
    if sd2 <= 0:
        return 0.0, 0.0
    sd = math.sqrt(sd2)
    m3 = sum((x - m) ** 3 for x in xs) / n
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m3 / sd ** 3, m4 / sd ** 4 - 3.0

def var_es(xs, q):
    n = len(xs)
    assert n > 0 and 0.0 < q < 1.0, (n, q)
    s = sorted(xs)
    k = max(1, int(math.ceil(q * n)))
    thr = s[k - 1]
    tail = [x for x in xs if x <= thr]
    es = sum(tail) / len(tail)
    return {"q": q, "var_ret": thr, "var_loss": -thr, "es_ret": es, "es_loss": -es, "tail_n": len(tail), "cut_rank": k}

def worst_day(xs, day_bars=DAY_BARS):
    n = len(xs)
    nd = n // day_bars
    assert nd >= 1, (n, day_bars)
    sums = [sum(xs[d * day_bars:(d + 1) * day_bars]) for d in range(nd)]
    w = min(range(nd), key=lambda d: sums[d])
    return {"day_idx": w, "start_bar": w * day_bars, "end_bar": (w + 1) * day_bars, "day_net": sums[w], "n_days": nd, "leftover_bars": n - nd * day_bars}

def leg_tail(net):
    sk, ku = skew_kurt(net)
    v5 = var_es(net, 0.05)
    v1 = var_es(net, 0.01)
    wd = worst_day(net)
    n = len(net)
    m = sum(net) / n if n else 0.0
    v = sum((x - m) ** 2 for x in net) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    return {"n": n, "mean": round(m, 8), "sd": round(math.sqrt(v), 8) if v > 0 else 0.0, "sharpe": round(sh, 3), "cum": round(sum(net), 4), "skew": round(sk, 4), "kurt_excess": round(ku, 4), "var5_ret": round(v5["var_ret"], 6), "var5_loss": round(v5["var_loss"], 6), "es5_ret": round(v5["es_ret"], 6), "es5_loss": round(v5["es_loss"], 6), "es5_tail_n": v5["tail_n"], "var1_ret": round(v1["var_ret"], 6), "var1_loss": round(v1["var_loss"], 6), "es1_ret": round(v1["es_ret"], 6), "es1_loss": round(v1["es_loss"], 6), "es1_tail_n": v1["tail_n"], "worst_day": {"day_idx": wd["day_idx"], "start_bar": wd["start_bar"], "end_bar": wd["end_bar"], "day_net": round(wd["day_net"], 6), "n_days": wd["n_days"], "leftover_bars": wd["leftover_bars"]}}

def dump(state):
    OUT.write_text(json.dumps(state, indent=1, ensure_ascii=False))

def base_config(coins, n):
    return {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5", "formula": list(FORMULA), "basket": {c: dict(BASE_SPECS[c]) for c in coins}, "scaled_specs_1h": {c: dict(SPECS[c]) for c in coins}, "scale_4h_to_1h": SCALE, "weights": {c: WEIGHT for c in coins}, "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "grid": "1h", "grid_bars": n, "day_bars": DAY_BARS, "bpy": BPY, "data_dir": "data/data_1y/1h", "coins": list(coins), "smoke": SMOKE, "tail_def": "per-bar net returns; VaR_ret=q-quantile (loss threshold); ES_ret=mean(net<=VaR_ret); loss forms negated; skew/kurt population (kurt excess); worst-day=min-sum 24-bar window anchored at bar 0", "note": "H16 1h Top5 tail-risk read-only; E10 FORMULA untouched; offline read-only"}

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H16_tail start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 480)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    log("common 1h n=%d coins=%s smoke=%s scale=x%d" % (n, coins, SMOKE, SCALE))
    assert n > 200, n
    assert n >= DAY_BARS, (n, DAY_BARS)
    cfg = base_config(coins, n)
    dump({"status": "partial", "stage": "loaded", "config": cfg})
    legs = {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        legs[c], _, _ = leg_net(raw, rt, sig, SPECS[c], FEE, FUND)
        dump({"status": "partial", "stage": "leg_%s" % c, "config": cfg, "legs_done": sorted(legs)})
        log("leg %s done" % c)
    per_coin = {}
    for c in coins:
        per_coin[c] = leg_tail(legs[c])
        per_coin[c]["worst_day"]["start_ts"] = ts[per_coin[c]["worst_day"]["start_bar"]]
        dump({"status": "partial", "stage": "tail_%s" % c, "config": cfg, "per_coin": per_coin})
        log("tail %s skew=%.3f kurt=%.3f var5=%.6f es5=%.6f var1=%.6f es1=%.6f worst_day=%d net=%.6f" % (c, per_coin[c]["skew"], per_coin[c]["kurt_excess"], per_coin[c]["var5_ret"], per_coin[c]["es5_ret"], per_coin[c]["var1_ret"], per_coin[c]["es1_ret"], per_coin[c]["worst_day"]["day_idx"], per_coin[c]["worst_day"]["day_net"]))
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    basket = leg_tail(net)
    bwd = worst_day(net)
    contrib = {c: round(sum(legs[c][bwd["start_bar"]:bwd["end_bar"]]) * w, 6) for c in coins}
    day_net = sum(contrib.values())
    share = {c: round(contrib[c] / day_net, 4) if day_net != 0 else 0.0 for c in coins}
    basket["worst_day_attrib"] = {"day_idx": bwd["day_idx"], "start_bar": bwd["start_bar"], "end_bar": bwd["end_bar"], "start_ts": ts[bwd["start_bar"]], "day_net": round(day_net, 6), "contrib": contrib, "share": share, "n_days": bwd["n_days"], "leftover_bars": bwd["leftover_bars"]}
    dump({"status": "partial", "stage": "basket", "config": cfg, "per_coin": per_coin, "basket": basket})
    log("basket skew=%.3f kurt=%.3f var5=%.6f es5=%.6f worst_day=%d net=%.6f contrib=%s" % (basket["skew"], basket["kurt_excess"], basket["var5_ret"], basket["es5_ret"], bwd["day_idx"], day_net, contrib))
    verdict = "PENDING_P03_FAIL"
    decision = "NO_ADOPTION_KEEP_EQUAL"
    note = "P0-3 permutation FAIL (per-coin p>=0.05) => H16 verdict PENDING; tail-risk read-only, no adoption, live untouched."
    cstr = ", ".join("%s skew=%.3f kurt=%.3f var5=%.6f/es5=%.6f var1=%.6f/es1=%.6f" % (c, per_coin[c]["skew"], per_coin[c]["kurt_excess"], per_coin[c]["var5_ret"], per_coin[c]["es5_ret"], per_coin[c]["var1_ret"], per_coin[c]["es1_ret"]) for c in coins)
    res = {"status": "final", "config": cfg, "per_coin": per_coin, "basket": basket, "verdict": verdict, "decision": decision, "decision_note": note, "conclusion": "H16 1h Top5 tail FULL " + cstr + " | basket worst-day %d net=%.6f. PENDING; no live change." % (bwd["day_idx"], day_net)}
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, verdict, decision))

if __name__ == "__main__":
    main()
