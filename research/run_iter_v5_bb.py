"""V5 15m Bollinger entry filter (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/15m/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Equal 0.2 weights
(Top5 locked specs ETC/TRX/ATOM/APT/KAS).

Filter: entry only outside BB(96,2) band on close. Per bar t (0-indexed):
mid = mean(close[t-95..t]), sd = population std over the same 96 bars,
upper = mid + 2*sd, lower = mid - 2*sd. Entry intent at bar t is kept iff
close[t] > upper or close[t] < lower. First 95 bars (warmup, no full
window) pass through unfiltered (mask=1). The mask gates the RAW
threshold intent (lp_raw/sp_raw after sigmoid thresholds + liquidity
safety + q0.3 long-only mask) BEFORE cooldown/stops; the downstream
pipeline (cooldown + stops + vol_scale + roll1 + fee/funding accounting)
is identical to base. Compare vs base: FULL sharpe + trades (entries).

15m native: cd/ts/vw x16, BPY=35040. Data data/data_1y/15m/{COIN}.csv
(~35040 rows: timestamp,open,high,low,close,volume,quote_volume,trades).

Output: results/iter_V5_bb.json (+ logs/iter_V5_bb.log). Incremental dump
after EACH coin unit (status partial->done); a killed run keeps partials.
Verdict PENDING (P0-3 FAIL); no adoption; live untouched.

Smoke mode (for tests): ITER_V5_SMOKE=1 shrinks to coins {ETC,TRX},
first 3000 bars (BB window stays 96). ITER_V5_OUT / ITER_V5_LOG override
paths (tests use temp files so the committed FULL artifact is not
clobbered).
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
assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
BPY = 35040.0
SCALE = 16
BB_WIN = 96
BB_K = 2.0
OUT = pathlib.Path(os.getenv("ITER_V5_OUT", "results/iter_V5_bb.json"))
LOG = pathlib.Path(os.getenv("ITER_V5_LOG", "logs/iter_V5_bb.log"))
SMOKE = os.getenv("ITER_V5_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
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

def bb_mask(closes, window=BB_WIN, k=BB_K):
    """1.0 where close is strictly outside the BB(window,k) band, else 0.0.

    Population std over the trailing `window` closes. First window-1 bars
    (warmup, no full window) pass through as 1.0.
    """
    n = len(closes)
    out = [0.0] * n
    ps = [0.0] * (n + 1)
    pq = [0.0] * (n + 1)
    for t, x in enumerate(closes):
        ps[t + 1] = ps[t] + x
        pq[t + 1] = pq[t] + x * x
    for t in range(n):
        if t + 1 < window:
            out[t] = 1.0
            continue
        s = ps[t + 1] - ps[t + 1 - window]
        qq = pq[t + 1] - pq[t + 1 - window]
        m = s / window
        var = max(qq / window - m * m, 0.0)
        sd = math.sqrt(var)
        up = m + k * sd
        lo = m - k * sd
        x = closes[t]
        out[t] = 1.0 if (x > up or x < lo) else 0.0
    return out
def leg_net(raw, rt, sig, spec, fee, fund, bb=None):
    """Mirror run_weight_modes leg_net. bb: optional 0/1 gate on raw intent."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    if bb is not None:
        g = torch.tensor([bb], dtype=lp.dtype)
        lp = lp * g; sp = sp * g
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), (lp - sp)[0].tolist()

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def count_trades(pos):
    n = 0; prev = 0.0
    for v in pos:
        cur = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if cur != 0.0 and prev == 0.0:
            n += 1
        prev = cur
    return n

def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_V5_bb start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, closes = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, SMOKE_N)
        bars = {c: bars[c][:n] for c in coins}
        closes = {c: closes[c][:n] for c in coins}
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d bb=(%d,%.1f)" % (n, coins, SMOKE, SCALE, BB_WIN, BB_K))
    assert n > 2000, n
    if not SMOKE:
        assert n == 35040, "15m grid must be 35040, got %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")
    masks = {c: bb_mask(closes[c], BB_WIN, BB_K) for c in coins}
    for c in coins:
        log("%s bb pass=%d/%d (%.3f)" % (c, int(sum(masks[c])), n, sum(masks[c]) / n))

    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; bb BB(96,2) entry gate on raw intent pre-cooldown", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "coins": list(coins), "weights": dict(w), "venue": "aster", "lev": LEV, "lev_locked": LEV, "fund": FUND, "fee": FEE, "bb": {"window": BB_WIN, "k": BB_K, "on": "close", "warmup": "pass-through", "gate": "raw intent pre-cooldown"}, "arms": ["base", "bb"], "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "smoke": SMOKE, "note": "entry-filter only, read-only; live chain untouched; PENDING (P0-3 FAIL), not demo evidence"},
        "status": "partial", "units": {}, "verdict": "PENDING"}
    dump(res)

    legs = {}
    for c in coins:
        raw, rt, sg = mats[c]
        spec = SPECS[c]
        base_net, base_turn, base_pos = leg_net(raw, rt, sg, spec, FEE, FUND, None)
        bb_net, bb_turn, bb_pos = leg_net(raw, rt, sg, spec, FEE, FUND, masks[c])
        legs[c] = {"base": (base_net, base_turn, base_pos), "bb": (bb_net, bb_turn, bb_pos)}
        ub = {"FULL": seg(base_net, base_turn, 0, n), "trades": count_trades(base_pos)}
        uf = {"FULL": seg(bb_net, bb_turn, 0, n), "trades": count_trades(bb_pos)}
        res["units"][c] = {"base": ub, "bb": uf,
            "d_sharpe": round(uf["FULL"]["sharpe"] - ub["FULL"]["sharpe"], 3),
            "d_trades": uf["trades"] - ub["trades"],
            "bb_pass": int(sum(masks[c])), "bb_pass_rate": round(sum(masks[c]) / n, 5)}
        dump(res)
        log("%s base sh=%.3f tr=%d to=%.6f | bb sh=%.3f tr=%d to=%.6f | dsh=%+.3f dtr=%+d" % (
            c, ub["FULL"]["sharpe"], ub["trades"], ub["FULL"]["turnover"],
            uf["FULL"]["sharpe"], uf["trades"], uf["FULL"]["turnover"],
            res["units"][c]["d_sharpe"], res["units"][c]["d_trades"]))

    for arm in ("base", "bb"):
        net = [sum(legs[c][arm][0][t] * w[c] for c in coins) for t in range(n)]
        turn = [sum(legs[c][arm][1][t] * w[c] for c in coins) for t in range(n)]
        st = seg(net, turn, 0, n)
        st["trades"] = sum(count_trades(legs[c][arm][2]) for c in coins)
        res[arm] = st
        dump(res)
    b, f = res["base"], res["bb"]
    res["compare"] = {"d_sharpe": round(f["sharpe"] - b["sharpe"], 3), "d_trades": f["trades"] - b["trades"],
        "d_turnover": round(f["turnover"] - b["turnover"], 6), "d_mdd": round(f["mdd"] - b["mdd"], 4),
        "d_cum": round(f["cum"] - b["cum"], 4)}
    dump(res)

    res["verdict"] = "PENDING"
    res["verdict_detail"] = "PENDING_P03_FAIL"
    res["decision"] = "NO_CHANGE"
    res["decision_note"] = "待定(P0-3 FAIL): no adoption, live untouched; bb entry filter diagnostic only (dsh=%+.3f dtr=%+d)." % (res["compare"]["d_sharpe"], res["compare"]["d_trades"])
    res["conclusion"] = "PENDING: V5 15m BB(96,2) entry filter Top5 %s; base vs bb FULL sharpe/trades; no live change" % coins
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=%s decision=%s dsh=%+.3f" % (OUT, res["verdict"], res["decision"], res["compare"]["d_sharpe"]))

if __name__ == "__main__":
    main()
