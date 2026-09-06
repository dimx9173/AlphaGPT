"""Iter8: 12-factor formulas on TRX 4h (baseline params: 0.88/0.12/cd3/sl3%/2x/aster)."""
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

OFF = 12  # operator offset in 12-factor vocab
OPS = {"ADD": OFF+0, "SUB": OFF+1, "MUL": OFF+2, "DIV": OFF+3, "NEG": OFF+4,
       "ABS": OFF+5, "SIGN": OFF+6, "GATE": OFF+7, "JUMP": OFF+8, "DECAY": OFF+9,
       "DELAY1": OFF+10, "MAX3": OFF+11}
F = {"RET": 0, "PRESSURE": 2, "FOMO": 3, "DEV": 4, "VOL_CLUSTER": 6,
     "MOM_REV": 7, "REL_STRENGTH": 8, "HL_RANGE": 9, "CLOSE_POS": 10, "VOL_TREND": 11}
FORMULAS = {
    "MOM12": [F["RET"], F["PRESSURE"], OPS["ADD"], OPS["DECAY"]],
    "RSI": [F["REL_STRENGTH"], F["CLOSE_POS"], OPS["SUB"]],
    "RSI_D": [F["REL_STRENGTH"], F["CLOSE_POS"], OPS["SUB"], OPS["DECAY"]],
    "VOLGATE": [F["VOL_CLUSTER"], F["RET"], F["PRESSURE"], OPS["ADD"], OPS["GATE"]],
    "REV12": [F["DEV"], F["FOMO"], OPS["SUB"], OPS["DECAY"]],
    "HLR": [F["HL_RANGE"], F["VOL_TREND"], OPS["ADD"]],
}
with open("data_15m_3y/TRX.csv") as f:
    rows = list(csv.DictReader(f))
bars = []
for i in range(0, len(rows), 16):
    blk = rows[i:i+16]
    if len(blk) < 16: break
    bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                 "high": max(float(x["high"]) for x in blk),
                 "low": min(float(x["low"]) for x in blk),
                 "volume": sum(float(x["volume"]) for x in blk)})
n = len(bars)
raw = {"open": torch.tensor([[b["open"] for b in bars]]),
       "high": torch.tensor([[b["high"] for b in bars]]),
       "low": torch.tensor([[b["low"] for b in bars]]),
       "close": torch.tensor([[b["close"] for b in bars]]),
       "volume": torch.tensor([[b["volume"] for b in bars]]),
       "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
feats = FeatureEngineer.compute_features(raw, use_advanced=True)
print("feat shape:", tuple(feats.shape))
vm = StackVM(use_advanced=True)
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
out = []
for fname, ftoks in FORMULAS.items():
    sig = vm.execute(ftoks, feats)
    if sig is None:
        print(f"{fname}: VM invalid"); continue
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=3, bars_per_year=2190.0, stop_loss=0.03)
    fit, cum = bt.evaluate(sig, raw, target)
    m = bt.last_metrics
    ann = float(cum)/n*2190.0
    out.append({"formula": fname, "fit": round(float(fit),2), "cum": round(float(cum),3),
                "ann": round(ann,3), "sharpe": round(m["sharpe"],3),
                "dd": round(m["max_dd"],3), "turn": round(m["turnover"],4)})
    print(out[-1])
with open("backtest_iter8_TRX.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
best = max(out, key=lambda r: r["sharpe"])
print("BEST:", best)
