"""Z5 15m quantile sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/15m/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Equal 0.2 weights
unless round varies them.

Top5 locked specs: ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24). Sweep q {0.1,0.2,0.3,0.4,0.5}.

15m native: cd/ts/vw x16, BPY=35040. Data data/data_1y/15m/{COIN}.csv
(~35040 rows: timestamp,open,high,low,close,volume,quote_volume,trades).

Per q: FULL sharpe/turnover/long-short attribution + fee2x curve knee
(max perpendicular distance on (q, FULL fee2x sharpe)).

Output: results/iter_Z5_q.json (+ logs/iter_Z5_q.log).
Verdict PENDING (P0-3 FAIL); no adoption; live untouched.

Smoke mode (for tests): ITER_Z5_SMOKE=1 shrinks to coins {ETC,TRX},
q {0.1,0.3}, first 3000 bars. ITER_Z5_OUT / ITER_Z5_LOG override paths
(tests use temp files so the committed FULL artifact is not clobbered).
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

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
QS = [0.1, 0.2, 0.3, 0.4, 0.5]
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path(os.getenv("ITER_Z5_OUT", "results/iter_Z5_q.json"))
LOG = pathlib.Path(os.getenv("ITER_Z5_LOG", "logs/iter_Z5_q.log"))
SMOKE = os.getenv("ITER_Z5_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_QS = [0.1, 0.3]
SMOKE_N = 3000

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

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

def leg_pnl(raw, rt, sig, spec, fee, fund, q):
    spec = dict(spec); spec["q"] = q
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, q)
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    lturn = (lp - lp.roll(1, dims=1)).abs()
    sturn = (sp - sp.roll(1, dims=1)).abs()
    rate = bt.base_fee
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * rate * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx - fnd)[0].tolist()
    lp_l = lp[0].tolist(); sp_l = sp[0].tolist()
    lt_l = lturn[0].tolist(); st_l = sturn[0].tolist()
    rt_l = rt[0].tolist()
    long_net, short_net = [], []
    for t in range(len(net)):
        ln = lp_l[t] * rt_l[t] * LEV - lt_l[t] * rate * LEV - lp_l[t] * fund * LEV
        sn = -sp_l[t] * rt_l[t] * LEV - st_l[t] * rate * LEV + sp_l[t] * fund * LEV
        long_net.append(ln); short_net.append(sn)
    pos = (lp - sp)[0].tolist()
    turn_l = turn[0].tolist()
    el, es = 0, 0
    prev = 0.0
    for v in pos:
        cur = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if cur != 0.0 and prev == 0.0:
            if cur > 0: el += 1
            else: es += 1
        prev = cur
    return {"net": net, "turn": turn_l, "long_net": long_net, "short_net": short_net, "entries_l": el, "entries_s": es}

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def side_split(lnet, snet, el, es, a, b):
    ln = sum(lnet[a:b]); sn = sum(snet[a:b]); tot = ln + sn
    if abs(tot) < 1e-12:
        shl, shs = 0.0, 0.0
    else:
        base = abs(tot)
        shl = round(ln / base, 4); shs = round(sn / base, 4)
    return {"long_entries": el, "short_entries": es, "long_pnl": round(ln, 4), "short_pnl": round(sn, 4), "long_share": shl, "short_share": shs}

def find_knee(xs, ys):
    n = len(xs)
    if n < 3:
        return xs[0], 0
    x0, x1 = xs[0], xs[-1]; y0, y1 = ys[0], ys[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-12:
        return xs[0], 0
    best_i, best_d = 0, -1e18
    for i in range(n):
        d = abs(dy * xs[i] - dx * ys[i] + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_d, best_i = d, i
    return xs[best_i], best_i

def eval_q(mats, coins, n, h2a, q, fee, fund):
    legs = {}
    for c in coins:
        raw, rt, sg = mats[c]
        legs[c] = leg_pnl(raw, rt, sg, SPECS[c], fee, fund, q)
    w = 1.0 / len(coins)
    net = [sum(legs[c]["net"][t] * w for c in coins) for t in range(n)]
    turn = [sum(legs[c]["turn"][t] * w for c in coins) for t in range(n)]
    lnet = [sum(legs[c]["long_net"][t] * w for c in coins) for t in range(n)]
    snet = [sum(legs[c]["short_net"][t] * w for c in coins) for t in range(n)]
    el = sum(legs[c]["entries_l"] for c in coins)
    es = sum(legs[c]["entries_s"] for c in coins)
    return net, turn, lnet, snet, el, es

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z5_q start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    qs = list(SMOKE_QS) if SMOKE else list(QS)
    bars, _ = common15m(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, SMOKE_N)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 15m native n=%d h2a=%d coins=%s qs=%s smoke=%s scale=x%d" % (n, h2a, coins, qs, SMOKE, SCALE))
    assert n > 2000, n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    rows = []
    for q in qs:
        net, turn, lnet, snet, el, es = eval_q(mats, coins, n, h2a, q, FEE, FUND)
        full = seg(net, turn, 0, n); h2 = seg(net, turn, h2a, n)
        net2, turn2, _, _, _, _ = eval_q(mats, coins, n, h2a, q, FEE2X, FUND)
        full2 = seg(net2, turn2, 0, n); h22 = seg(net2, turn2, h2a, n)
        row = {"q": q, "FULL": full, "H2": h2, "FULL_fee2x": full2, "H2_fee2x": h22,
               "attribution_full": side_split(lnet, snet, el, es, 0, n),
               "attribution_h2": side_split(lnet, snet, el, es, h2a, n)}
        rows.append(row)
        log("q=%.1f FULL sh=%.3f cum=%.4f mdd=%.4f to=%.6f | H2 sh=%.3f | FULL2x sh=%.3f H22x sh=%.3f | L/S entries=%d/%d pnl=%.4f/%.4f" % (
            q, full["sharpe"], full["cum"], full["mdd"], full["turnover"], h2["sharpe"], full2["sharpe"], h22["sharpe"], el, es,
            row["attribution_full"]["long_pnl"], row["attribution_full"]["short_pnl"]))
    curve = [r["FULL_fee2x"]["sharpe"] for r in rows]
    base_curve = [r["FULL"]["sharpe"] for r in rows]
    knee_q, knee_i = find_knee(qs, curve)
    res = {
        "config": {
            "engine": "mirror run_weight_modes.py leg_net + quantile q long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1",
            "formula": list(FORMULA),
            "basket": {c: {"lth": SPECS[c]["lth"], "sth": SPECS[c]["sth"], "cd": SPECS[c]["cd"], "sl": SPECS[c]["sl"], "ts": SPECS[c]["ts"], "q": "sweep"} for c in coins},
            "coins": list(coins),
            "weights": {c: round(1.0 / len(coins), 4) for c in coins},
            "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "fee2x": FEE2X,
            "qs": list(qs), "grid": "15m", "grid_bars": n, "h2_start": h2a, "h2_len": n - h2a,
            "bpy": BPY, "scale": SCALE,
            "smoke": SMOKE,
            "knee": "max perpendicular distance from chord on (q, FULL fee2x sharpe)",
            "attribution": "per-leg long_net/short_net split (fee+funding allocated by side turnover/position); entries count 0->side flips",
            "note": "E10 FORMULA untouched; live chain untouched; offline read-only; PENDING (P0-3 FAIL), not demo evidence",
        },
        "rows": rows,
        "fee2x_curve": [{"q": q, "FULL_fee2x_sharpe": v} for q, v in zip(qs, curve)],
        "base_curve": [{"q": q, "FULL_sharpe": v} for q, v in zip(qs, base_curve)],
        "knee": {"q": knee_q, "idx": knee_i, "curve": list(curve)},
    }
    if (not SMOKE) and any(abs(r["q"] - 0.3) < 1e-9 for r in rows):
        r03 = next(r for r in rows if abs(r["q"] - 0.3) < 1e-9)
        others = [r["FULL_fee2x"]["sharpe"] for r in rows if abs(r["q"] - 0.3) > 1e-9]
        mean_others = sum(others) / len(others) if others else 0.0
        gap = (r03["FULL_fee2x"]["sharpe"] - mean_others) / max(abs(mean_others), 1e-9)
        isolated = bool(gap > 0.15 and all(r03["FULL_fee2x"]["sharpe"] > r["FULL_fee2x"]["sharpe"] for r in rows if abs(r["q"] - 0.3) > 1e-9))
        res["q03_check"] = {"FULL_fee2x": r03["FULL_fee2x"]["sharpe"], "FULL": r03["FULL"]["sharpe"],
                            "mean_others_fee2x": round(mean_others, 3), "gap_pct": round(gap * 100, 1), "isolated_peak": isolated}
        note_extra = "q0.3 isolated peak (gap %+.1f%%)" % (gap * 100) if isolated else "q0.3 not isolated (gap %+.1f%%)" % (gap * 100)
    else:
        res["q03_check"] = {"FULL_fee2x": None, "FULL": None, "mean_others_fee2x": None, "gap_pct": None, "isolated_peak": False, "note": "smoke grid; q0.3 check on FULL run only"}
        note_extra = "smoke grid"
    res["verdict"] = "PENDING"
    res["verdict_detail"] = "PENDING_P03_FAIL"
    res["decision"] = "KEEP_q0.3"
    res["decision_note"] = "待定(P0-3 FAIL): no adoption, live untouched; knee q=%.1f diagnostic only (%s); fee2x curve %s" % (knee_q, note_extra, curve)
    res["conclusion"] = "PENDING: Z5 15m quantile sweep Top5 %s; knee q=%.1f; no live change" % (qs, knee_q)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s decision=%s knee=%.1f" % (OUT, res["verdict"], res["decision"], knee_q))

if __name__ == "__main__":
    main()
