"""CSV-direct AlphaGPT training on TRX 4h (no DB).

Mirrors AlphaEngine.train but loads feats from data/data_15m_3y CSVs.
Reward = MemeBacktest fitness with baseline winning params
(aster 2x, 0.88/0.12, cd3, sl3%, 4h annualization).
Usage: python3 train_csv.py [steps] [batch]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, sys
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

def load_4h(coins=("TRX",)):
    bars_all = {}
    for coin in coins:
        with open(f"data/data_15m_3y/{coin}.csv") as f:
            rows = list(csv.DictReader(f))
        bars = []
        for i in range(0, len(rows), 16):
            blk = rows[i:i+16]
            if len(blk) < 16: break
            bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                         "high": max(float(x["high"]) for x in blk),
                         "low": min(float(x["low"]) for x in blk),
                         "volume": sum(float(x["volume"]) for x in blk)})
        bars_all[coin] = bars
    return bars_all

def build_tensors(bars_all):
    coins = list(bars_all)
    n = min(len(b) for b in bars_all.values())
    def stack(key):
        return torch.tensor([[bars_all[c][i][key] for i in range(n)] for c in coins])
    raw = {"open": stack("open"), "high": stack("high"), "low": stack("low"),
           "close": stack("close"), "volume": stack("volume"),
           "liquidity": torch.full((len(coins), n), 1e7),
           "fdv": torch.full((len(coins), n), 1e8)}
    feats = FeatureEngineer.compute_features(raw)
    tgt = []
    for c in coins:
        cl = [bars_all[c][i]["close"] for i in range(n)]
        tgt.append([(cl[i+1]-cl[i])/cl[i] if i < n-1 else 0.0 for i in range(n)])
    return feats, torch.tensor(tgt), raw, coins

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    coins = ("TRX", "BTC", "BNB", "ETH")
    feats, target, raw, cnames = build_tensors(load_4h(coins))
    print(f"feats {tuple(feats.shape)} coins={cnames}")
    device = ModelConfig.DEVICE
    feats, target = feats.to(device), target.to(device)
    model = AlphaGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = StackVM()
    best_score, best_formula = -1e9, None
    hist = []
    for step in tqdm(range(steps)):
        inp = torch.zeros((bs, 1), dtype=torch.long, device=device)
        lps, toks, vals, ents = [], [], [], []
        for _ in range(ModelConfig.MAX_FORMULA_LEN):
            logits, value, _ = model(inp)
            dist = Categorical(logits=logits)
            a = dist.sample()
            lps.append(dist.log_prob(a)); ents.append(dist.entropy())
            vals.append(value.squeeze(-1)); toks.append(a)
            inp = torch.cat([inp, a.unsqueeze(1)], dim=1)
        seqs = torch.stack(toks, dim=1)
        rewards = torch.zeros(bs, device=device)
        for i in range(bs):
            res = vm.execute(seqs[i].tolist(), feats)
            if res is None:
                rewards[i] = -5.0; continue
            if res.std() < 1e-4:
                rewards[i] = -2.0; continue
            bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                              funding_override=0.0005, long_th=0.88, short_th=0.12,
                              cooldown_bars=3, bars_per_year=2190.0, stop_loss=0.03)
            score, _ = bt.evaluate(res, {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}, target)
            rewards[i] = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
            if float(rewards[i]) > best_score:
                best_score = float(rewards[i]); best_formula = seqs[i].tolist()
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        pl = torch.stack([-lp * adv for lp in lps], dim=0).sum(dim=0).mean()
        vl = F.mse_loss(torch.stack(vals, dim=0).mean(dim=0), rewards)
        loss = pl + 0.5 * vl - 0.01 * torch.stack(ents, dim=0).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        avg = rewards.mean().item()
        hist.append({"step": step, "avg": round(avg, 3), "best": round(best_score, 3)})
        if step % 20 == 0:
            print(f"step {step} avg={avg:.3f} best={best_score:.3f} formula={best_formula}", flush=True)
    with open("results/train_csv_best.json", "w") as f:
        json.dump({"formula": best_formula, "score": best_score, "hist": hist}, f)
    print(f"DONE best={best_score:.3f} formula={best_formula}")

if __name__ == "__main__":
    main()
