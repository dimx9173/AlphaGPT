"""Z4 15m vol-window sweep (Top5): vw {6,12,24} (x16) x vt {None,0.006,0.012} FULL sharpe/mdd.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp lev2, fee=FEE fund=FUND. 15m native: cd/ts/vw x16, BPY=35040.
E10 FORMULA untouched. Read-only: LEV/FUND/FEE/config untouched.

Grid: 3 vw x 3 vt = 9 cells -> FULL sharpe/mdd (+ann/cum/final_x/n/turnover).
vt=None cells are vw-invariant (vol_scale->1.0); computed once, recorded x3.
Post-cooldown/stop binary positions are vt/vw-independent, so per-coin they
are precomputed once; each cell only applies vol_scale + roll1 + accounting.
Incremental dump: OUT rewritten after EACH cell (status partial->done).

Smoke: ITER_Z4_SMOKE=1 -> coins {ETC,TRX}, first 3000 bars.
OUT/LOG overridable via ITER_Z4_OUT / ITER_Z4_LOG.
Output: results/iter_Z4_vw.json (verdict PENDING P0-3 FAIL, no adoption).
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
VW_GRID = [6, 12, 24]
VT_GRID = [None, 0.006, 0.012]
LEV_USE = 2.0
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path(os.getenv("ITER_Z4_OUT", "results/iter_Z4_vw.json"))
LOG = pathlib.Path(os.getenv("ITER_Z4_LOG", "logs/iter_Z4_vw.log"))
SMOKE = os.getenv("ITER_Z4_SMOKE") == "1"


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


def base_positions(raw, rt, sig, spec):
    """Post-cooldown/stop binary positions; vt/vw-independent (vol_scale later)."""
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=None, vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    return lp, sp


def cell_net(lp0, sp0, rt, vt, vw):
    bt = MemeBacktest(venue="aster", leverage=LEV_USE, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=0.5, short_th=0.5, cooldown_bars=0, bars_per_year=BPY, stop_loss=None, time_stop=0, vol_target=vt, vol_window=int(vw) * SCALE)
    sc = bt._vol_scale(rt)
    lp, sp = lp0 * sc, sp0 * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist()


def cell_key(vt, vw):
    return "vt%s_vw%d" % ("None" if vt is None else vt, vw)


def seg(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s); m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(t) / n, 6) if n else 0.0}


def leg_sharpe(net):
    n = len(net); m = sum(net) / n if n else 0
    v = sum((x - m) ** 2 for x in net) / max(n - 1, 1) if n > 1 else 0
    return round(m / math.sqrt(v) * math.sqrt(BPY), 3) if v > 0 else 0.0


def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_Z4_vw start smoke=%s\n" % SMOKE)
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, _ = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 15m native n=%d coins=%s smoke=%s scale=x%d" % (n, coins, SMOKE, SCALE))
    assert n > 2000, n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built")
    base = {}
    for c in coins:
        raw, rt, sg = mats[c]
        base[c] = (base_positions(raw, rt, sg, SPECS[c]), rt)
    log("base positions precomputed (post-cooldown/stops)")
    res = {"config": {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 15m native cd/ts/vw x16", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in coins}, "weights": dict(w), "vw_grid": list(VW_GRID), "vt_grid": list(VT_GRID), "venue": "aster", "lev": LEV_USE, "lev_locked": LEV, "fund": FUND, "fee": FEE, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "note": "vol-window sweep only, read-only; live vt/vw untouched"}, "rows": {}, "status": "partial"}
    dump(res)
    for vt in VT_GRID:
        for vw in VW_GRID:
            key = cell_key(vt, vw)
            legs = {}; turns = {}
            for c in coins:
                (lp0, sp0), rt = base[c]
                legs[c], turns[c] = cell_net(lp0, sp0, rt, vt, vw)
            net = [sum(legs[c][t] * w[c] for c in coins) for t in range(n)]
            turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
            full = seg(net, turn, 0, n)
            full["legs_sharpe"] = {c: leg_sharpe(legs[c]) for c in coins}
            res["rows"][key] = {"vt": vt, "vw": vw, "FULL": full}
            log("%s FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f to=%.6f" % (key, full["sharpe"], full["mdd"], full["cum"], full["final_x"], full["turnover"]))
            dump(res)
    sh = {k: res["rows"][k]["FULL"]["sharpe"] for k in res["rows"]}
    dd = {k: res["rows"][k]["FULL"]["mdd"] for k in res["rows"]}
    best = max(sh, key=lambda k: sh[k])
    none_keys = [cell_key(None, vw) for vw in VW_GRID]
    none_flat = bool(max(sh[k] for k in none_keys) - min(sh[k] for k in none_keys) < 1e-9)
    vt12 = {("None" if vt is None else str(vt)): sh[cell_key(vt, 12)] for vt in VT_GRID}
    vw_eff = {}
    for vt in VT_GRID:
        vals = [sh[cell_key(vt, vw)] for vw in VW_GRID]
        vw_eff[str(vt)] = {"min": round(min(vals), 3), "max": round(max(vals), 3), "spread": round(max(vals) - min(vals), 4)}
    res["compare"] = {"sharpe_by_cell": sh, "mdd_by_cell": dd, "best_cell": best, "best_sharpe": sh[best], "vt_none_flat_across_vw": none_flat, "vt_effect_at_vw12": vt12, "vw_effect_by_vt": vw_eff}
    res["verdict"] = "PENDING"
    res["p03"] = "FAIL"
    res["decision"] = "NO_CHANGE"
    res["decision_note"] = "待定，不採用。vol-window掃描僅研究對照；P0-3 FAIL，live vt/vw 不動 (vt None, vw12)。"
    res["status"] = "done"
    dump(res)
    log("done best=%s sh=%.3f verdict=PENDING p03=FAIL decision=NO_CHANGE" % (best, sh[best]))


if __name__ == "__main__":
    main()
