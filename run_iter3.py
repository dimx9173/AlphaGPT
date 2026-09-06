"""Iter3 sweep: threshold x cooldown x formula on T1 coins (4h real factors)."""
import csv, json, sys
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

COINS = ["TRX", "BTC", "BNB", "ETH"]
FORMULAS = {"F_MOM": [0, 2, 6, 15], "F_REV": [4, 3, 7]}
GRID = [(0.85, 0.15, 0), (0.90, 0.10, 0), (0.92, 0.08, 0), (0.90, 0.10, 3), (0.92, 0.08, 6)]
KW = dict(venue="aster", leverage=2.0, short_enabled=True, funding_override=0.0005)

def load_4h(coin):
    with open(f"data_15m_3y/{coin}.csv") as f:
        rows = list(csv.DictReader(f))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16:
            break
        bars.append({
            "open": float(blk[0]["open"]),
            "high": max(float(x["high"]) for x in blk),
            "low": min(float(x["low"]) for x in blk),
            "close": float(blk[-1]["close"]),
            "volume": sum(float(x["volume"]) for x in blk),
        })
    return bars

def run_coin(coin):
    bars = load_4h(coin)
    n = len(bars)
    raw = {
        "open": torch.tensor([[b["open"] for b in bars]]),
        "high": torch.tensor([[b["high"] for b in bars]]),
        "low": torch.tensor([[b["low"] for b in bars]]),
        "close": torch.tensor([[b["close"] for b in bars]]),
        "volume": torch.tensor([[b["volume"] for b in bars]]),
        "liquidity": torch.full((1, n), 1e7),
        "fdv": torch.full((1, n), 1e8),
    }
    feats = FeatureEngineer.compute_features(raw)
    vm = StackVM()
    rets = [(bars[i+1]["close"] - bars[i]["close"]) / bars[i]["close"] for i in range(n - 1)] + [0.0]
    target = torch.tensor([rets])
    out = []
    for fname, ftoks in FORMULAS.items():
        sig = vm.execute(ftoks, feats)
        if sig is None:
            print(f"{coin} {fname}: VM invalid")
            continue
        for (lth, sth, cd) in GRID:
            bt = MemeBacktest(long_th=lth, short_th=sth, cooldown_bars=cd,
                              bars_per_year=2190.0, **KW)
            fit, cum = bt.evaluate(sig, raw, target)
            m = bt.last_metrics
            out.append({"coin": coin, "formula": fname, "lth": lth, "sth": sth, "cd": cd,
                        "fit": round(float(fit), 3), "cum": round(float(cum), 3),
                        "sharpe": round(m["sharpe"], 3), "dd": round(m["max_dd"], 3),
                        "turn": round(m["turnover"], 4)})
    return out

if __name__ == "__main__":
    coins = sys.argv[1:] or COINS
    for coin in coins:
        rows = run_coin(coin)
        with open(f"backtest_iter3_{coin}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        best = max(rows, key=lambda r: r["sharpe"])
        print(f"{coin} rows={len(rows)} best={best}", flush=True)
