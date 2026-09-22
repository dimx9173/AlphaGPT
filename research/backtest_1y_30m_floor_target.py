#!/usr/bin/env python3
"""1-Year 30m Backtest: Target 10% monthly, DD <= 5%"""

import csv, json, math, os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS, LEV, FUND, FEE

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
BPY = 17520.0
SCALE = 1
TARGET_MONTHLY_RETURN = 0.10
TARGET_ANNUAL_RETURN = (1 + TARGET_MONTHLY_RETURN) ** 12 - 1
TARGET_MAX_DD = 0.05
BEST_FORMULA = [4, 3, 5, 5, 3, 14, 19, 20, 20, 4, 19, 26]

def load30m(c):
    path = f"data/data_1y/30m/{c}.csv"
    rows = list(csv.DictReader(open(path)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]

def common30m(coins):
    bars = {}
    n = None
    for c in coins:
        b = load30m(c)
        bars[c] = b
        if n is None:
            n = len(b)
    return bars, n

def build_sig(bars_c):
    n = len(bars_c)
    raw = {
        "open": torch.tensor([[b[1] for b in bars_c]]),
        "high": torch.tensor([[b[2] for b in bars_c]]),
        "low": torch.tensor([[b[3] for b in bars_c]]),
        "close": torch.tensor([[b[4] for b in bars_c]]),
        "volume": torch.tensor([[b[5] for b in bars_c]]),
        "liquidity": torch.full((1, n), 1e7),
        "fdv": torch.full((1, n), 1e8)
    }
    fe = FeatureEngineer.compute_features(raw, use_advanced=True)
    sig = StackVM(use_advanced=True).execute(BEST_FORMULA, fe)
    rets = [(bars_c[i+1][4] - bars_c[i][4]) / bars_c[i][4] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def run_backtest(bars, spec):
    raw, rets_t, sig = build_sig(bars)
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY, stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
    s = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (s > bt.long_th).float() * safe
    sp = (s < bt.short_th).float() * safe
    mask = sig.detach().float().abs() >= torch.topk(sig.detach().float().abs().reshape(-1), max(1, int(len(sig) * float(spec["q"])))).values.min()
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    sc = bt._vol_scale(rets_t)
    lp, sp = lp * sc, sp * sc
    pos = (lp - sp)[0].tolist()
    pos_rolled = [0.0] + pos[:-1]
    dp = [abs(pos_rolled[t] - (pos_rolled[t-1] if t else 0.0)) for t in range(len(pos))]
    net = [pos_rolled[t] * rets_t[0][t] * LEV - dp[t] * FEE * LEV - pos_rolled[t] * FUND * LEV for t in range(len(pos))]
    return pos_rolled, net, rets_t[0].tolist()

def seg(net):
    n = len(net)
    m = sum(net) / n if n else 0
    v = sum((x - m) ** 2 for x in net) / max(n - 1, 1) if n > 1 else 0
    sh = float(m / math.sqrt(v) * math.sqrt(BPY)) if v > 0 else 0.0
    cs = 0.0
    pk0 = -1e18
    md = 0.0
    for x in net:
        cs += x
        pk0 = max(pk0, cs)
        md = max(md, pk0 - cs)
    cum = sum(net)
    return {
        "sharpe": round(float(sh), 3),
        "annual_return": round(float(cum / n * BPY), 4) if n else 0.0,
        "cumulative": round(float(cum), 4),
        "final_x": round(1.0 + float(cum), 4),
        "max_drawdown": round(float(md), 4),
        "n": n
    }

def monthly_returns(net):
    bars_per_month = 48 * 30
    monthly = []
    for i in range(0, len(net), bars_per_month):
        chunk = net[i:i+bars_per_month]
        if len(chunk) >= bars_per_month // 2:
            monthly.append(sum(chunk))
    return monthly

def main():
    print("=== 1-Year 30m Backtest (Floor Formula) ===")
    bars, n = common30m(COINS)
    print(f"Loaded {n} 30m bars per coin ({n/48:.0f} days)")
    print(f"Formula: {BEST_FORMULA}")
    print(f"Targets: Monthly>=10%, DD<=5%, Annual>=120%")
    print()

    all_net = {}
    per_coin = {}
    for c in COINS:
        pos, net, rets = run_backtest(bars[c], SPECS[c])
        all_net[c] = net
        stats = seg(net)
        monthly = monthly_returns(net)
        avg_monthly = float(sum(monthly)) / len(monthly) if monthly else 0.0
        stats["monthly_avg"] = round(float(avg_monthly), 4) if monthly else 0.0
        per_coin[c] = stats
        print(f"  {c}: Sharpe={stats['sharpe']:.3f}, Ann={stats['annual_return']:.1%}, DD={stats['max_drawdown']:.1%}, MonthlyAvg={stats['monthly_avg']:.1%}")

    print("\n=== Portfolio (Equal Weight) ===")
    port_net = [sum(all_net[c][t] * W[c] for c in COINS) for t in range(n)]
    port_stats = seg(port_net)
    port_monthly = monthly_returns(port_net)
    port_monthly_avg = float(sum(port_monthly)) / len(port_monthly) if port_monthly else 0.0
    port_target_hit = sum(1 for m in port_monthly if m >= TARGET_MONTHLY_RETURN)

    print(f"Sharpe: {port_stats['sharpe']:.3f}")
    print(f"Annual Return: {port_stats['annual_return']:.1%}")
    print(f"Cumulative: {port_stats['cumulative']:.1%}")
    print(f"Final X: {port_stats['final_x']:.4f}x")
    print(f"Max Drawdown: {port_stats['max_drawdown']:.1%}")
    print(f"Monthly Avg: {port_monthly_avg:.1%}")
    print(f"Months >= 10%: {port_target_hit}/{len(port_monthly)}")
    print()

    print("=== Target Evaluation ===")
    print(f"Monthly 10%: {'PASS' if port_monthly_avg >= TARGET_MONTHLY_RETURN else 'FAIL'} (actual: {port_monthly_avg:.1%})")
    print(f"DD <= 5%: {'PASS' if port_stats['max_drawdown'] <= TARGET_MAX_DD else 'FAIL'} (actual: {port_stats['max_drawdown']:.1%})")
    print(f"Annual >= 120%: {'PASS' if port_stats['annual_return'] >= TARGET_ANNUAL_RETURN else 'FAIL'} (actual: {port_stats['annual_return']:.1%})")
    print()

    result = {
        "config": {"formula": BEST_FORMULA, "coins": COINS, "weights": W, "lev": LEV, "n_bars": n},
        "targets": {"monthly_return": TARGET_MONTHLY_RETURN, "annual_return": TARGET_ANNUAL_RETURN, "max_dd": TARGET_MAX_DD},
        "portfolio": port_stats,
        "portfolio_monthly_avg": round(float(port_monthly_avg), 4),
        "portfolio_target_hit": port_target_hit,
        "per_coin": per_coin,
        "verdict": {
            "monthly_target": port_monthly_avg >= TARGET_MONTHLY_RETURN,
            "dd_target": port_stats['max_drawdown'] <= TARGET_MAX_DD,
            "annual_target": port_stats['annual_return'] >= TARGET_ANNUAL_RETURN
        }
    }

    with open("results/backtest_1y_30m_floor_target.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/backtest_1y_30m_floor_target.json")

if __name__ == "__main__":
    main()
