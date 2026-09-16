"""train_12f_30m.py -- 12-factor 30m 1y CSV-direct AlphaGPT retrain (no DB).

Mirrors research/train_csv.py but: 12-factor vocab (USE_ADVANCED=1 default),
StackVM(use_advanced=True), 30m native bars from data/data_1y/30m/*.csv,
Top5 (ETC/TRX/ATOM/APT/KAS), E10-style reward (aster 2x fund0.0005 fee0.0004,
q0.3 long-only + cooldown + stops + vol_scale + roll1 via leg engine is
approximated here by MemeBacktest with matching per-coin thresholds).

Reward per formula: mean per-coin MemeBacktest sharpe on 30m bars
(BPY=17520), invalid/const formulas penalized (-5/-2, mirrors engine).

Output: results/train_12f_30m_best.json {formula, decode, score, hist}.
Offline read-only (CSV reads only); no live/demo change.
Usage: /home/linuxbrew/.linuxbrew/bin/python3 research/train_12f_30m.py [steps] [batch]
Smoke (tests): TRAIN_12F_SMOKE=1 -> steps=3 batch=4 coins={ETC,TRX}.
"""
import csv
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_ADVANCED"] = "1"

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.vocab import get_vocab

COINS5 = ["ETC", "TRX", "ATOM", "APT", "KAS"]
# per-coin E10 thresholds (lth/sth/cd/sl) for reward backtest
TH = {"ETC": (0.88, 0.12, 18, None), "TRX": (0.85, 0.12, 6, 0.05),
      "ATOM": (0.85, 0.15, 6, 0.05), "APT": (0.88, 0.12, 18, None),
      "KAS": (0.88, 0.12, 6, None)}
BPY30M = 17520.0
VOC = get_vocab(True)


def load_30m(coins):
    bars_all = {}
    for coin in coins:
        with open(f"data/data_1y/30m/{coin}.csv") as f:
            rows = list(csv.DictReader(f))
        bars_all[coin] = [{"close": float(r["close"]), "open": float(r["open"]),
                           "high": float(r["high"]), "low": float(r["low"]),
                           "volume": float(r["volume"])} for r in rows]
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
    feats = FeatureEngineer.compute_features(raw, use_advanced=True)
    assert feats.shape[1] == 12, feats.shape
    tgt = []
    for c in coins:
        cl = [bars_all[c][i]["close"] for i in range(n)]
        tgt.append([(cl[i + 1] - cl[i]) / cl[i] if i < n - 1 else 0.0 for i in range(n)])
    return feats, torch.tensor(tgt), raw, coins


def decode(formula):
    names = VOC.token_names
    return [names[t] if 0 <= t < len(names) else f"?{t}" for t in formula]


def main():
    smoke = os.getenv("TRAIN_12F_SMOKE") == "1"
    steps = 3 if smoke else (int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    bs = 4 if smoke else (int(sys.argv[2]) if len(sys.argv) > 2 else 64)
    coins = ("ETC", "TRX") if smoke else tuple(COINS5)
    # smoke never touches real artifacts; floor-constrained run gets its own file
    OUT = "/tmp/train_12f_30m_smoke.json" if smoke else (
        "results/train_12f_30m_floor_best.json" if "TRAIN_12F_FLOOR" in os.environ
        else "results/train_12f_30m_best.json")
    feats, target, raw, cnames = build_tensors(load_30m(coins))
    print(f"feats {tuple(feats.shape)} coins={cnames} vocab={VOC.size}", flush=True)
    device = ModelConfig.DEVICE
    feats, target = feats.to(device), target.to(device)
    raw_d = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    model = AlphaGPT(use_advanced=True).to(device)
    print(f"model vocab_size={model.vocab_size}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = StackVM(use_advanced=True)
    print(f"vm feat_offset={vm.feat_offset}", flush=True)
    best_score, best_formula, best_worst = -1e9, None, None
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
            f = seqs[i].tolist()
            res = vm.execute(f, feats)
            if res is None:
                rewards[i] = -5.0; continue
            if res.std() < 1e-4:
                rewards[i] = -2.0; continue
            scores = []
            worst = 1e9
            for j, c in enumerate(cnames):
                lth, sth, cd, sl = TH[c]
                bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                                  funding_override=0.0005, long_th=lth, short_th=sth,
                                  cooldown_bars=cd, bars_per_year=BPY30M, stop_loss=sl)
                one = {k: (v[j:j + 1] if isinstance(v, torch.Tensor) else v) for k, v in raw_d.items()}
                sc, _ = bt.evaluate(res[j:j + 1], one, target[j:j + 1])
                _s = float(sc.item()) if isinstance(sc, torch.Tensor) else float(sc)
                scores.append(_s)
                worst = min(worst, _s)
            mean_s = sum(scores) / len(scores)
            # per-leg floor: any leg sharpe<=0 vetoes (penalized, not adopted)
            leg_floor = float(os.getenv("TRAIN_12F_FLOOR", "0.0"))
            if worst <= leg_floor:
                rewards[i] = worst  # veto: reward = worst leg (<=floor), keeps gradient signal
            else:
                rewards[i] = mean_s
            if float(rewards[i]) > best_score:
                best_score = float(rewards[i]); best_formula = f; best_worst = worst
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        pl = torch.stack([-lp * adv for lp in lps], dim=0).sum(dim=0).mean()
        vl = F.mse_loss(torch.stack(vals, dim=0).mean(dim=0), rewards)
        loss = pl + 0.5 * vl - 0.01 * torch.stack(ents, dim=0).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        avg = rewards.mean().item()
        hist.append({"step": step, "avg": round(avg, 3), "best": round(best_score, 3)})
        if step % 20 == 0:
            print(f"step {step} avg={avg:.3f} best={best_score:.3f} formula={best_formula} decode={decode(best_formula) if best_formula else None}", flush=True)
    out = {"formula": best_formula, "decode": decode(best_formula) if best_formula else None,
           "score": best_score, "worst_leg": best_worst, "floor": float(os.getenv("TRAIN_12F_FLOOR", "0.0")),
           "hist": hist, "vocab": "12f",
           "coins": list(cnames), "bars": "30m-1y", "bpy": BPY30M}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    print(f"DONE best={best_score:.3f} formula={best_formula} decode={out['decode']}")


if __name__ == "__main__":
    main()
