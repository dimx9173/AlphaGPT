"""V9 15m VWAP direction filter (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the global exit rule this step's conclusion is "PENDING" and MUST NOT be used
as listing evidence. No adoption: decision stays KEEP (no VWAP), live chain
untouched.

Premise: the locked E10 engine (quantile q0.3 long-only + cooldown + stops +
vol_scale vt None->1.0 + roll1) takes shorts wherever sg < sth, including
counter-trend shorts into strength. V9 asks: does a 15m-native direction
filter -- long only above rolling session-VWAP, short only below -- help the
Top5 basket on the FULL 1y grid?

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA via
StackVM+FeatureEngineer; MemeBacktest venue=aster lev2 short_enabled
fund0.0005): the ONLY change vs the locked leg is the VWAP direction gate
applied to lp/sp right after threshold+liquidity gating (before the q0.3
long mask, cooldown, stops, vol_scale, roll1). 15m native: cd/ts/vw x16,
BPY=35040, equal 0.2 weights.

VWAP definition: 96-bar rolling VWAP on typical price (h+l+c)/3 weighted by
bar volume, computed on the common 15m grid per coin:
  vwap[t] = sum(typ*vol)/sum(vol) over bars[max(0,t-95):t+1].
Gate (strict): long leg kept only where close[t] > vwap[t]; short leg kept
only where close[t] < vwap[t]; equality blocks both (measure-zero).

Units: base (no gate) vs vwap96 (gate). FULL sharpe/trades/turnover at base
fee (FEE) plus fee2x (FEE2X) repricing (positions are fee-independent, so the
same legs are repriced). Results dumped incrementally after each unit, so a
partial JSON survives interruption.

Offline read-only: reads data/data_1y/15m/*.csv only. No orders.
"""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE, FEE2X

assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 35040.0
SCALE = 16
VWAP_WIN = 96
OUT = pathlib.Path("results/iter_V9_vwap.json")
LOG = pathlib.Path("logs/iter_V9_vwap.log")

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def rolling_vwap(bars, window=VWAP_WIN):
    """Pure-python 96-bar rolling VWAP. bars: [(o,h,l,c,v)...] -> [vwap...]."""
    n = len(bars)
    out = []
    for t in range(n):
        a = max(0, t - window + 1)
        num = 0.0; den = 0.0
        for i in range(a, t + 1):
            o, h, l, c, v = bars[i]
            typ = (h + l + c) / 3.0
            num += typ * v; den += v
        out.append(num / den if den > 0 else bars[t][3])
    return out

def vwap_gate_mask(closes, vwaps):
    """Strict direction masks: long only above, short only below, tie blocks both."""
    long_ok = [1.0 if c > v else 0.0 for c, v in zip(closes, vwaps)]
    short_ok = [1.0 if c < v else 0.0 for c, v in zip(closes, vwaps)]
    return long_ok, short_ok

def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common15m(coins):
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; closes = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [r[4] for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]; closes[c] = closes[c][:n]
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

def leg_legs(raw, rt, sig, spec, gate=None):
    """Fee-independent legs: positions + turnover. gate=(long_ok, short_ok) or None."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    if gate is not None:
        lok, sok = gate
        lp = lp * torch.tensor([lok]); sp = sp * torch.tensor([sok])
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    return (lp - sp)[0].tolist(), turn[0].tolist()

def price_net(legs, turns, rts, fee, fund):
    n = len(legs[COINS[0]])
    net = [sum(W[c] * (legs[c][t] * rts[c][t] * LEV - turns[c][t] * fee * LEV - legs[c][t] * fund * LEV) for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    return net, turn

def count_entries(pos):
    el = es = 0; prev = 0.0
    for v in pos:
        cur = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if cur != 0.0 and prev == 0.0:
            if cur > 0: el += 1
            else: es += 1
        prev = cur
    return el, es

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def eval_unit(legs, turns, rts, n):
    net, turn = price_net(legs, turns, rts, FEE, FUND)
    net2, turn2 = price_net(legs, turns, rts, FEE2X, FUND)
    full = seg(net, turn, 0, n); full2 = seg(net2, turn2, 0, n)
    per = {}
    for c in COINS:
        cn = [legs[c][t] * rts[c][t] * LEV - turns[c][t] * FEE * LEV - legs[c][t] * FUND * LEV for t in range(n)]
        ct = list(turns[c])
        el, es = count_entries(legs[c])
        st = seg(cn, ct, 0, n)
        st.update({"long_entries": el, "short_entries": es, "trades": el + es})
        per[c] = st
    el = sum(count_entries(legs[c])[0] for c in COINS); es = sum(count_entries(legs[c])[1] for c in COINS)
    full.update({"long_entries": el, "short_entries": es, "trades": el + es})
    full2.update({"long_entries": el, "short_entries": es, "trades": el + es})
    return {"FULL": full, "FULL_fee2x": full2, "per_coin": per}

def dump(rows_done, n, partial):
    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; VWAP direction gate (long>vw / short<vw) before q-mask; static equal 0.2 Top5; 15m native cd/ts/vw x16", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "fee2x": FEE2X, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "vwap": {"window": VWAP_WIN, "typical": "(h+l+c)/3", "rule": "long only above rolling VWAP / short only below; tie blocks both"}, "units": ["base", "vwap96"], "note": "diagnostic only; P0-3 FAIL => PENDING, no adoption; live untouched"},
        "rows": rows_done, "partial": bool(partial)}
    if not partial:
        b = rows_done["base"]["FULL"]; v = rows_done["vwap96"]["FULL"]
        b2 = rows_done["base"]["FULL_fee2x"]; v2 = rows_done["vwap96"]["FULL_fee2x"]
        res["compare"] = {"d_sharpe": round(v["sharpe"] - b["sharpe"], 3), "d_sharpe_fee2x": round(v2["sharpe"] - b2["sharpe"], 3), "turnover_ratio": round(v["turnover"] / b["turnover"], 4) if b["turnover"] else None, "trades_base": b["trades"], "trades_vwap": v["trades"], "trades_cut": round(1.0 - v["trades"] / b["trades"], 4) if b["trades"] else 0.0}
        res["verdict"] = "PENDING"
        res["decision"] = "KEEP_NO_VWAP"
        res["decision_note"] = "P0-3 permutation FAIL => verdict PENDING regardless of V9 outcome; no adoption, live chain untouched."
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_V9_vwap start\n")
    bars, closes = common15m(COINS)
    n = len(bars["ETC"])
    log("common 15m native n=%d scale=x%d vwap_win=%d" % (n, SCALE, VWAP_WIN))
    assert n > 30000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    rts = {c: mats[c][1][0].tolist() for c in COINS}
    log("signals built")
    gates = {}
    for c in COINS:
        vw = rolling_vwap(bars[c], VWAP_WIN)
        lok, sok = vwap_gate_mask(closes[c], vw)
        gates[c] = (lok, sok)
        log("%s vwap gate: long_ok=%d/%d short_ok=%d/%d" % (c, sum(lok), n, sum(sok), n))
    rows_done = {}
    for unit in ["base", "vwap96"]:
        legs = {}; turns = {}
        for c in COINS:
            raw, rt, sg = mats[c]
            legs[c], turns[c] = leg_legs(raw, rt, sg, SPECS[c], None if unit == "base" else gates[c])
        rows_done[unit] = eval_unit(legs, turns, rts, n)
        f = rows_done[unit]["FULL"]; f2 = rows_done[unit]["FULL_fee2x"]
        log("%s FULL sh=%.3f sh2x=%.3f cum=%.4f mdd=%.4f to=%.6f trades=%d (L%d/S%d)" % (unit, f["sharpe"], f2["sharpe"], f["cum"], f["mdd"], f["turnover"], f["trades"], f["long_entries"], f["short_entries"]))
        last = unit == "vwap96"
        dump(rows_done, n, partial=not last)
        log("dumped %s partial=%s" % (OUT, not last))
    log("wrote %s verdict=PENDING decision=KEEP_NO_VWAP" % OUT)

if __name__ == "__main__":
    main()
