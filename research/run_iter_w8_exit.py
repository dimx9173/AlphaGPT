"""W8 exit-timing (15m native): early-exit {0,1,2} bars before opposite-signal flip vs hold-to-flip.

Mirror research/run_iter_y7_lev.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0), static equal 0.2 Top5 weights,
aster perp lev2, fee=FEE fund=FUND. 15m native: cd/ts/vw x16, BPY=35040.

Method: per-coin signal-space position pos = lp-sp (post cooldown/stops,
pre roll1). Flip events = bars t where pos[t] != pos[t-1] with pos[t-1] != 0
(an open position exits or flips on the opposite signal). Early-exit K
zeroes pos[t-K..t-1] around each flip, then roll1 execution + accounting.
K=0 is the hold-to-flip baseline.

NOTE: K>0 is an ORACLE/lookahead diagnostic (it uses the future flip time).
Not tradable, not adopted. Read-only; LEV/live untouched.
Output: results/iter_W8_exit.json (verdict PENDING P0-3 FAIL, decision
KEEP_HOLD_TO_FLIP). Incremental dump after each K unit (partial survives).
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
KS = [0, 1, 2]
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path("results/iter_W8_exit.json")
LOG = pathlib.Path("logs/iter_W8_exit.log")

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

def base_pos(raw, rt, sig, spec):
    """Signal-space position (post cooldown/stops/vol_scale, pre roll1)."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    return (lp - sp)[0].tolist()

def apply_early_exit(pos, K):
    """Zero the K bars preceding each flip/exit event. K<=0 -> identity."""
    src = [float(x) for x in pos]
    if K is None or int(K) <= 0:
        return src
    K = int(K)
    n = len(src)
    flips = [t for t in range(1, n) if src[t] != src[t - 1] and src[t - 1] != 0.0]
    p = list(src)
    for t in flips:
        for j in range(max(0, t - K), t):
            p[j] = 0.0
    return p

def exec_net(pos, rt, fee, fund, lev):
    """roll1 execution + aster perp accounting. Returns (net, turn, trades)."""
    n = len(pos)
    lp = torch.tensor([pos]); r = torch.tensor([rt])
    exe = lp.roll(1, dims=1); exe[:, 0] = 0
    turn = (exe - exe.roll(1, dims=1)).abs()
    gross = exe * r * lev
    tx = turn * fee * lev
    fnd = exe * fund * lev
    net = (gross - tx - fnd)[0].tolist(); tu = turn[0].tolist()
    e = exe[0].tolist()
    entries = sum(1 for t in range(n) if e[t] != 0.0 and (t == 0 or e[t - 1] == 0.0))
    exits = sum(1 for t in range(n) if (e[t] == 0.0 and t > 0 and e[t - 1] != 0.0) or (t == n - 1 and e[t] != 0.0))
    flips = sum(1 for t in range(1, n) if e[t] != 0.0 and e[t - 1] != 0.0 and e[t] != e[t - 1])
    changes = sum(1 for t in range(1, n) if e[t] != e[t - 1])
    return net, tu, {"trades": int(entries), "entries": int(entries), "exits": int(exits), "flips": int(flips), "changes": int(changes)}

def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}

def dump(path, res):
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_W8_exit start\n")
    bars, closes = common15m(COINS)
    n = len(bars["ETC"])
    log("common 15m native n=%d ks=%s scale=x%d" % (n, KS, SCALE))
    mats = {c: build_sig(bars[c]) for c in COINS}
    base = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        base[c] = {"pos": base_pos(raw, rt, sg, SPECS[c]), "rt": rt[0].tolist()}
        log("base %s nz=%d" % (c, sum(1 for x in base[c]["pos"] if x != 0.0)))
    res = {"config": {"engine": "mirror run_iter_y7_lev leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5; 15m native cd/ts/vw x16; early-exit K bars before opposite-signal flip (oracle lookahead, diagnostic only)", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "ks": list(KS), "venue": "aster", "lev_locked": LEV, "fund": FUND, "fee": FEE, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "oracle": True, "note": "K>0 uses future flip time; not tradable; live untouched"},
        "rows": {},
        "compare": {},
        "verdict": "PENDING",
        "decision": "KEEP_HOLD_TO_FLIP",
        "decision_note": "P0-3 FAIL => verdict PENDING; oracle early-exit diagnostic only, hold-to-flip stays, no adoption, live untouched."}
    dump(OUT, res)  # initial skeleton survives even if a unit crashes
    for K in KS:
        legs = {}; turns = {}; tr = {}
        for c in COINS:
            p = apply_early_exit(base[c]["pos"], K)
            legs[c], turns[c], tr[c] = exec_net(p, base[c]["rt"], FEE, FUND, LEV)
        net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
        turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
        full = seg(net, turn, 0, n)
        tot = {k: sum(v[k] for v in tr.values()) for k in ("trades", "entries", "exits", "flips", "changes")}
        full.update(tot)
        full["by_coin"] = {c: dict(tr[c]) for c in COINS}
        res["rows"][str(K)] = {"FULL": full}
        dump(OUT, res)  # incremental: partial survives per unit
        log("K=%d FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f to=%.6f trades=%d" % (K, full["sharpe"], full["mdd"], full["cum"], full["final_x"], full["turnover"], full["trades"]))
    sh = {k: res["rows"][k]["FULL"]["sharpe"] for k in res["rows"]}
    td = {k: res["rows"][k]["FULL"]["trades"] for k in res["rows"]}
    to = {k: res["rows"][k]["FULL"]["turnover"] for k in res["rows"]}
    res["compare"] = {"sharpe_by_k": sh, "trades_by_k": td, "turnover_by_k": to,
        "d_sharpe_vs_hold": {k: round(sh[k] - sh["0"], 4) for k in sh if k != "0"},
        "d_trades_vs_hold": {k: int(td[k] - td["0"]) for k in td if k != "0"}}
    dump(OUT, res)
    log("wrote %s verdict=%s decision=%s" % (OUT, res["verdict"], res["decision"]))

if __name__ == "__main__":
    main()
