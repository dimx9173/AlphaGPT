"""H18 1h signal persistence (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Equal 0.2 weights.

Top5 locked specs: ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows:
timestamp,open,high,low,close,volume,quote_volume,trades).

Per coin (executed signed position series, post cooldown/stops/vol/roll1):
  acf_1_96     : autocorrelation of executed pos at lags 1..96
  sig_acf      : autocorrelation of raw logit signal at key lags
  holding      : run-length stats of nonzero same-sign segments
                 (median/mean/max/n_runs, entries)
  flips        : reversal events (nonzero sign vs previous nonzero sign):
                 n_reversals, n_direct (adjacent), gaps median,
                 share gaps<=24h (clustering), max reversals in any 96-bar window
  trades       : entry->exit segments: n, median hold, corr(hold,pnl),
                 short-hold (<=24) vs long-hold (>24) mean pnl
Portfolio: equal-weight FULL leg-net stats + cross-coin
corr(median_hold, sharpe) as persistence-vs-PnL read.

Output: results/iter_H18_persist.json (+ logs/iter_h18_persist.log).
Verdict PENDING (P0-3 FAIL); no adoption; live untouched.

Smoke mode (for tests): ITER_H18_SMOKE=1 shrinks to coins {ETC,TRX},
first 3000 bars. ITER_H18_OUT / ITER_H18_LOG override paths
(tests use temp files so the committed FULL artifact is not clobbered).
Incremental dump: OUT rewritten after EACH unit (per-coin signal,
per-coin metrics, portfolio), status partial->done, so partial
progress survives kills.
"""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 8760.0
SCALE = 4
LAGS = list(range(1, 97))
SIG_LAGS = [1, 2, 4, 6, 12, 24, 48, 72, 96]
SHORT_HOLD = 24
CLUSTER_GAP = 24
CLUSTER_WIN = 96
OUT = pathlib.Path(os.getenv("ITER_H18_OUT", "results/iter_H18_persist.json"))
LOG = pathlib.Path(os.getenv("ITER_H18_LOG", "logs/iter_h18_persist.log"))
SMOKE = os.getenv("ITER_H18_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_N = 3000

_partial = {"verdict": "PENDING"}

def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")

def dump():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(_partial, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common1h(coins):
    raw = {c: load1h(c) for c in coins}
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

def leg_series(raw, rt, sig, spec, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 1h-native params.

    Returns per-bar net/turn/executed-signed-pos (+1/-1/0) series."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    pos = (lp - sp)[0].tolist()
    spos = [1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0) for v in pos]
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), spos

def acf(x, lag):
    """Lag-k autocorrelation (pearson of x[t] vs x[t-lag]); 0.0 if degenerate."""
    n = len(x)
    if lag < 1 or lag >= n:
        return 0.0
    a = x[lag:]; b = x[:n - lag]
    m = len(a)
    ma = sum(a) / m; mb = sum(b) / m
    cov = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    va = sum((u - ma) ** 2 for u in a); vb = sum((v - mb) ** 2 for v in b)
    den = math.sqrt(va * vb)
    return cov / den if den > 1e-12 else 0.0

def median(v):
    s = sorted(v); n = len(s)
    if n == 0:
        return 0.0
    h = n // 2
    return float(s[h] if n % 2 else (s[h - 1] + s[h]) / 2.0)

def holding_runs(pos):
    """Consecutive same-sign nonzero segments -> lengths."""
    runs = []; cur = 0.0; ln = 0
    for v in pos:
        if v != 0.0 and v == cur:
            ln += 1
        elif v != 0.0:
            if ln > 0:
                runs.append(ln)
            cur = v; ln = 1
        else:
            if ln > 0:
                runs.append(ln)
            cur = 0.0; ln = 0
    if ln > 0:
        runs.append(ln)
    return runs

def reversal_events(pos):
    """Times t where pos[t] is nonzero and the previous nonzero bar has opposite sign.
    Returns (events, n_direct) where n_direct counts adjacent (gap==1) reversals."""
    events = []; n_direct = 0
    last_sign = 0.0; last_t = -1
    for t, v in enumerate(pos):
        if v == 0.0:
            continue
        s = 1.0 if v > 0 else -1.0
        if last_sign != 0.0 and s != last_sign:
            events.append(t)
            if t - last_t == 1:
                n_direct += 1
        last_sign = s; last_t = t
    return events, n_direct

def flip_cluster(events, gap=CLUSTER_GAP, win=CLUSTER_WIN):
    gaps = [events[i + 1] - events[i] for i in range(len(events) - 1)]
    if not events:
        return {"n_reversals": 0, "gaps_median": 0.0, "share_gap_le": 0.0, "max_in_window": 0}
    share = sum(1 for g in gaps if g <= gap) / len(gaps) if gaps else 0.0
    mx = 1; j = 0
    for i in range(len(events)):
        while events[i] - events[j] >= win:
            j += 1
        mx = max(mx, i - j + 1)
    return {"n_reversals": len(events), "gaps_median": round(median(gaps), 2) if gaps else 0.0,
            "share_gap_le": round(share, 4), "max_in_window": mx}

def trade_segments(pos, net):
    """Entry(0->nonzero)->exit segments; pnl = sum net over [entry, exit)."""
    trades = []
    t0 = -1
    for t, v in enumerate(pos):
        prev = pos[t - 1] if t > 0 else 0.0
        if prev == 0.0 and v != 0.0:
            t0 = t
        if t0 >= 0 and (v == 0.0 or (v != 0.0 and prev != 0.0 and (v > 0) != (prev > 0))):
            hold = t - t0
            pnl = sum(net[t0:t])
            if hold > 0:
                trades.append((hold, pnl))
            t0 = t if v != 0.0 else -1
    if t0 >= 0:
        trades.append((len(pos) - t0, sum(net[t0:])))
    return trades

def pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n; mb = sum(b) / n
    cov = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    va = sum((u - ma) ** 2 for u in a); vb = sum((v - mb) ** 2 for v in b)
    den = math.sqrt(va * vb)
    return cov / den if den > 1e-12 else 0.0

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def coin_persistence(pos, net, sigv):
    pa = [round(acf(pos, k), 4) for k in LAGS]
    sa = {str(k): round(acf(sigv, k), 4) for k in SIG_LAGS}
    runs = holding_runs(pos)
    entries = sum(1 for t, v in enumerate(pos) if v != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    changes = sum(1 for t in range(1, len(pos)) if pos[t] != pos[t - 1])
    events, n_direct = reversal_events(pos)
    fc = flip_cluster(events)
    trades = trade_segments(pos, net)
    holds = [h for h, _ in trades]; pnls = [p for _, p in trades]
    sh = [p for h, p in trades if h <= SHORT_HOLD]; lh = [p for h, p in trades if h > SHORT_HOLD]
    return {
        "acf_1_96": pa,
        "acf_lag1": pa[0], "acf_lag24": pa[23], "acf_lag96": pa[95],
        "sig_acf": sa,
        "holding": {"n_runs": len(runs), "entries": entries, "changes": changes,
                    "median_bars": round(median(runs), 2) if runs else 0.0,
                    "mean_bars": round(sum(runs) / len(runs), 2) if runs else 0.0,
                    "max_bars": max(runs) if runs else 0},
        "flips": {"n_reversals": fc["n_reversals"], "n_direct": n_direct,
                  "n_via_flat": fc["n_reversals"] - n_direct,
                  "gaps_median_bars": fc["gaps_median"],
                  "cluster_share_le24": fc["share_gap_le"],
                  "max_reversals_in_96b": fc["max_in_window"]},
        "trades": {"n": len(trades),
                   "median_hold_bars": round(median(holds), 2) if holds else 0.0,
                   "corr_hold_pnl": round(pearson([float(h) for h in holds], pnls), 4) if len(trades) >= 3 else 0.0,
                   "mean_pnl_hold_le24": round(sum(sh) / len(sh), 6) if sh else 0.0,
                   "mean_pnl_hold_gt24": round(sum(lh) / len(lh), 6) if lh else 0.0,
                   "n_hold_le24": len(sh), "n_hold_gt24": len(lh)},
    }

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H18_persist start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    _partial["config"] = {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 1h native cd/ts/vw x4", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "coins": list(coins), "venue": "aster", "lev": LEV, "lev_locked": LEV, "fee": FEE, "fund": FUND, "grid": "1h", "bpy": BPY, "scale": SCALE, "lags": "1..96", "sig_lags": list(SIG_LAGS), "short_hold": SHORT_HOLD, "cluster_gap": CLUSTER_GAP, "cluster_win": CLUSTER_WIN, "smoke": SMOKE, "note": "persistence diagnostic only, read-only; P0-3 FAIL so no adoption, live untouched"}
    _partial["status"] = "partial"; _partial["coins_done"] = []; _partial["per_coin"] = {}; _partial["legs"] = {}
    _partial["verdict"] = "PENDING"
    dump()
    bars, _ = common1h(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, SMOKE_N)
        bars = {c: bars[c][:n] for c in coins}
    log("common 1h native n=%d coins=%s smoke=%s scale=x%d" % (n, coins, SMOKE, SCALE))
    assert n > 2000, n
    _partial["config"]["grid_bars"] = n
    dump()
    legs = {}
    for c in coins:
        raw, rt, sg = build_sig(bars[c])
        net, turn, spos = leg_series(raw, rt, sg, SPECS[c], FEE, FUND)
        legs[c] = (net, turn, spos)
        _partial["legs"][c] = seg(net, turn, 0, n)
        _partial["coins_done"].append("signal_%s" % c)
        dump()  # incremental: partial survives
        log("%s leg cum=%.4f sh=%.3f" % (c, sum(net), _partial["legs"][c]["sharpe"]))
    for c in coins:
        raw, rt, sg = build_sig(bars[c])
        sigv = sg.detach().float().reshape(-1).tolist()
        net, turn, spos = legs[c]
        _partial["per_coin"][c] = coin_persistence(spos, net, sigv)
        _partial["per_coin"][c]["FULL"] = _partial["legs"][c]
        _partial["coins_done"].append("persist_%s" % c)
        dump()  # incremental: partial survives
        p = _partial["per_coin"][c]
        log("%s acf1=%.3f acf24=%.3f acf96=%.3f medhold=%.1f rev=%d clus=%.3f tr=%d corr=%.3f" % (
            c, p["acf_lag1"], p["acf_lag24"], p["acf_lag96"], p["holding"]["median_bars"],
            p["flips"]["n_reversals"], p["flips"]["cluster_share_le24"], p["trades"]["n"], p["trades"]["corr_hold_pnl"]))
    wv = 1.0 / len(coins)
    net = [sum(legs[c][0][t] * wv for c in coins) for t in range(n)]
    turn = [sum(legs[c][1][t] * wv for c in coins) for t in range(n)]
    _partial["portfolio"] = {"FULL": seg(net, turn, 0, n)}
    meds = [(_partial["per_coin"][c]["holding"]["median_bars"], _partial["legs"][c]["sharpe"]) for c in coins]
    _partial["persistence_vs_pnl"] = {
        "corr_median_hold_vs_sharpe": round(pearson([m for m, _ in meds], [s for _, s in meds]), 4) if len(meds) >= 3 else 0.0,
        "by_coin": {c: {"median_hold_bars": _partial["per_coin"][c]["holding"]["median_bars"], "sharpe": _partial["legs"][c]["sharpe"], "corr_hold_pnl": _partial["per_coin"][c]["trades"]["corr_hold_pnl"]} for c in coins}}
    _partial["status"] = "done"
    _partial["verdict"] = "PENDING"
    _partial["decision"] = "NO_CHANGE"
    _partial["conclusion"] = "PENDING (P0-3 FAIL): 1h Top5 signal persistence diagnostic; no adoption, live untouched."
    dump()
    log("H18 done portfolio sh=%.3f mdd=%.4f" % (_partial["portfolio"]["FULL"]["sharpe"], _partial["portfolio"]["FULL"]["mdd"]))

if __name__ == "__main__":
    main()
