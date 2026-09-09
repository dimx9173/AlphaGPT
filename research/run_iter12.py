"""Iteration 1+2: real 6-factor + StackVM formulas x 4h resample, 28 coins.

Formulas (postfix over feat ids 0..5 + ops offset 6):
  F_MOM  = [RET, PRESSURE, ADD, DECAY]          -> smoothed momentum+pressure
  F_REV  = [DEV, NEG, REL? no: DEV, FOMO, SUB]  -> deviation minus fomo (reversion)
Pseudocode check via StackVM; falls back to raw feat rows if formula invalid.
Target = next-bar 4h return. Liquidity 1e7. Seeds 7/21/42.
Configs: sol1x / hl2x / aster2x / hl3x (same as baseline).
Output: backtest_iter12_<COIN>.jsonl per coin.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, os, statistics, sys

FEAT = {"RET": 0, "LIQ_SCORE": 1, "PRESSURE": 2, "FOMO": 3, "DEV": 4, "LOG_VOL": 5}
OPS = {"ADD": 6, "SUB": 7, "MUL": 8, "DIV": 9, "NEG": 10, "ABS": 11, "SIGN": 12,
       "GATE": 13, "JUMP": 14, "DECAY": 15, "DELAY1": 16, "MAX3": 17}
FORMULAS = {
    "F_MOM": [FEAT["RET"], FEAT["PRESSURE"], OPS["ADD"], OPS["DECAY"]],
    "F_REV": [FEAT["DEV"], FEAT["FOMO"], OPS["SUB"]],
}
CONFIGS = [
    ("solana_spot_1x", dict(venue="solana", leverage=1.0, short_enabled=False, funding_override=0.0)),
    ("hl_perp_2x", dict(venue="hyperliquid", leverage=2.0, short_enabled=True, funding_override=0.00045)),
    ("aster_perp_2x", dict(venue="aster", leverage=2.0, short_enabled=True, funding_override=0.0005)),
    ("hl_perp_3x", dict(venue="hyperliquid", leverage=3.0, short_enabled=True, funding_override=0.00045)),
]
SEEDS = (7, 21, 42)
D = "data/data_15m_3y"


def load_4h(coin):
    with open(os.path.join(D, f"{coin}.csv")) as f:
        rows = list(csv.DictReader(f))
    out = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16:
            break
        out.append({
            "open": float(blk[0]["open"]), "high": max(float(x["high"]) for x in blk),
            "low": min(float(x["low"]) for x in blk), "close": float(blk[-1]["close"]),
            "volume": sum(float(x["volume"]) for x in blk),
        })
    return out


def run_coin(coin):
    import torch
    from model_core.factors import FeatureEngineer
    from model_core.vm import StackVM
    from model_core.backtest import MemeBacktest
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
    feats = FeatureEngineer.compute_features(raw)  # [1, 6, T]
    vm = StackVM()
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    target = torch.tensor([rets])
    out = []
    for fname, ftoks in FORMULAS.items():
        sig = vm.execute(ftoks, feats)
        if sig is None:
            print(f"{coin} {fname}: VM invalid, skip")
            continue
        for cfg_name, kw in CONFIGS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                bt = MemeBacktest(**kw)
                fit, cum = bt.evaluate(sig, raw, target)
                m = bt.last_metrics
                out.append({"coin": coin, "formula": fname, "config": cfg_name, "seed": seed,
                            "bars_4h": n, "fitness": float(fit), "cum_ret": float(cum),
                            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in m.items()}})
    return out


if __name__ == "__main__":
    coins = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(D) if f.endswith(".csv"))
    for coin in coins:
        rows = run_coin(coin)
        path = f"results/backtest_iter12_{coin}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        means = {}
        for fname in FORMULAS:
            vals = [r["fitness"] for r in rows if r["formula"] == fname and r["config"] == "aster_perp_2x"]
            means[fname] = round(statistics.mean(vals), 3) if vals else None
        print(f"{coin:6s} 4h_bars done rows={len(rows)} aster2x {means}", flush=True)
