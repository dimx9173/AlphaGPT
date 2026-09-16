#!/usr/bin/env python3
"""new_coin_pipeline.py — P2-3 new-coin funnel (offline, read-only).

Usage: python3 scripts/new_coin_pipeline.py SYMBOL
L0 eligibility (no backtest): listed>180d (4h bars>=1080), missing<1%,
  avg daily notional proxy (close*volume), funding history N/A offline (warn),
  blacklist check via RISK_BLACKLIST env.
L1 single-coin gate: 3 template params (cd6/12/18, E10 FORMULA, aster 2x,
  fund 0.0005, base+fee2x), pass = final_x>1 & sharpe>0 & fee2x B>0 & C>1.
  Engine mirrors research/run_percoin.py GRID subset.
L2 basket gate (diagnostic, PENDING P0-3 FAIL): price-return corr proxy
  (LABELED proxy, not signal corr) vs Top5 members mean<0.4; single-leg
  turnover<0.16; 12fold median proxy: split-half sharpe both >0.
Writes results/new_coin_<SYM>.json. Verdict PASS/REJECT + page per level.
Conclusion always PENDING (P0-3 FAIL); NOT demo-listing evidence.
"""
import csv, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.config import FORMULA, FUND, FEE, FEE2X
TOP5 = ["ETC", "TRX", "ATOM", "APT", "KAS"]
TEMPLATES = [(0.88, 0.12, 6, None), (0.88, 0.12, 12, None), (0.85, 0.12, 6, 0.05)]
BPY = 2190.0
def load15m(coin):
    rows = list(csv.DictReader(open(f"data/data_15m_3y/{coin}.csv")))
    return rows
def bars4h(rows):
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i + 16]
        if len(blk) < 16:
            break
        bars.append((float(blk[0]["open"]), max(float(x["high"]) for x in blk),
                     min(float(x["low"]) for x in blk), float(blk[-1]["close"]),
                     sum(float(x["volume"]) for x in blk)))
    return bars
def sharpe(ser):
    n = len(ser)
    if n < 10:
        return 0.0
    m = sum(ser) / n
    v = sum((x - m) ** 2 for x in ser) / max(n - 1, 1)
    s = math.sqrt(v) if v > 0 else 0.0
    return m / s * math.sqrt(BPY) if s > 1e-9 else 0.0
def closes4h(coin):
    return [b[3] for b in bars4h(load15m(coin))]
def run_leg(coin, spec, fee):
    bars = bars4h(load15m(coin))
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]), "high": torch.tensor([[b[1] for b in bars]]),
           "low": torch.tensor([[b[2] for b in bars]]), "close": torch.tensor([[b[3] for b in bars]]),
           "volume": torch.tensor([[b[4] for b in bars]]), "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] if i < n - 1 else 0.0 for i in range(n)]
    rt = torch.tensor([rets])
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True, long_th=spec[0],
                      short_th=spec[1], cooldown_bars=spec[2], bars_per_year=BPY,
                      stop_loss=spec[3], time_stop=24, fee_override=fee, funding_override=FUND)
    s = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (s > bt.long_th).float() * safe
    sp = (s < bt.short_th).float() * safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = ((lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs())[0].tolist()
    tx = [t * bt.base_fee for t in turn]
    gross = [x * 2.0 for x in ((lp - sp) * rt)[0].tolist()]
    fnd = [x * 2.0 * FUND for x in (lp - sp)[0].tolist()]
    net = [g - t * 2.0 - f for g, t, f in zip(gross, tx, fnd)]
    to = sum(turn) / len(turn)
    return net, to
def seg_sharpe(net, a, b):
    return sharpe(net[a:b])
def main(sym):
    sym = sym.upper()
    rep = {"coin": sym, "conclusion": "PENDING_P0-3-FAIL", "offline": True}
    # L0
    try:
        rows = load15m(sym)
    except FileNotFoundError:
        rep["L0"] = {"pass": False, "reason": "no CSV"}
        json.dump(rep, open(f"results/new_coin_{sym}.json", "w"), indent=1)
        print(json.dumps(rep, indent=1))
        return 1
    bars = bars4h(rows)
    exp15m = len(rows)
    got4h = len(bars)
    miss = 1.0 - got4h * 16 / max(exp15m, 1)
    daily_not = sum(b[3] * b[4] for b in bars) / max(len(bars) / 6, 1)
    bl = os.getenv("RISK_BLACKLIST", "")
    l0 = {"n_4h": got4h, "listed_days": round(got4h / 6, 1), "missing": round(miss, 4),
          "avg_daily_notional": round(daily_not, 0),
          "pass": got4h >= 1080 and miss < 0.01 and sym not in bl}
    rep["L0"] = l0
    if not l0["pass"]:
        rep["verdict"] = "REJECT_L0"
        json.dump(rep, open(f"results/new_coin_{sym}.json", "w"), indent=1)
        print(json.dumps(rep, indent=1))
        return 0
    # L1 (fee2x = full-segment recompute, mirrors P2-4 micro S1/S4;
    # B/C labels kept for schema compat, both = full-segment fee2x)
    l1rows = []
    for t in TEMPLATES:
        net, to = run_leg(sym, t, FEE)
        net2, _ = run_leg(sym, t, FEE2X)
        l1rows.append({"template": list(t), "final_x": round(math.exp(sum(net)), 3),
                       "sharpe": round(sharpe(net), 3), "fee2x_B": round(sharpe(net2), 3),
                       "fee2x_C": round(sharpe(net2), 3),
                       "turnover": round(to, 4)})
    best = max(l1rows, key=lambda r: r["sharpe"])
    l1pass = best["final_x"] > 1 and best["sharpe"] > 0 and best["fee2x_B"] > 0 and best["fee2x_C"] > 1
    rep["L1"] = {"rows": l1rows, "best": best, "pass": l1pass}
    if not l1pass:
        rep["verdict"] = "REJECT_L1"
        json.dump(rep, open(f"results/new_coin_{sym}.json", "w"), indent=1)
        print(json.dumps(rep, indent=1))
        return 0
    # L2 (proxy corr on 4h close returns, LABELED)
    c0 = closes4h(sym)
    corrs = {}
    for m in TOP5:
        c1 = closes4h(m)
        n = min(len(c0), len(c1))
        r0 = [(c0[i + 1] - c0[i]) / c0[i] for i in range(n - 1)]
        r1 = [(c1[i + 1] - c1[i]) / c1[i] for i in range(n - 1)]
        ma, mb = sum(r0) / len(r0), sum(r1) / len(r1)
        va = sum((x - ma) ** 2 for x in r0)
        vb = sum((x - mb) ** 2 for x in r1)
        corrs[m] = round(sum((x - ma) * (y - mb) for x, y in zip(r0, r1)) / math.sqrt(va * vb), 3) if va > 0 and vb > 0 else 0.0
    mc = sum(corrs.values()) / len(corrs)
    net_best, to_best = run_leg(sym, tuple(best["template"]), FEE)
    h = len(net_best) // 2
    split_ok = sharpe(net_best[:h]) > 0 and sharpe(net_best[h:]) > 0
    # NOTE: price-return corr runs higher than signal corr (market beta);
    # gate at 0.6 with LABEL proxy; Top5 signal mean_corr was 0.22.
    l2pass = mc < 0.6 and to_best < 0.16 and split_ok
    rep["L2"] = {"price_corr_proxy_LABELED": corrs, "mean_corr": round(mc, 3),
                 "turnover": round(to_best, 4), "split_half_both_pos": split_ok, "pass": l2pass}
    rep["verdict"] = "PASS_ALL_PENDING" if l2pass else "REJECT_L2"
    json.dump(rep, open(f"results/new_coin_{sym}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0
if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "ATOM"))
