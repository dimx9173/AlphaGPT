"""train_12f_30m.py -- 12-factor 30m 1y CSV-direct AlphaGPT retrain (no DB).

Mirrors research/train_csv.py but: 12-factor vocab (USE_ADVANCED=1 default),
StackVM(use_advanced=True), 30m native bars (Top5 legacy uses data_1y/30m; TRAIN_UNIVERSE_28C=1 uses
Binance data_3y/30m for the unified 28-coin manifest),
E10-style reward (aster 2x fund0.0005 fee0.0004,
q0.3 long-only + cooldown + stops + vol_scale + roll1 via leg engine is
approximated here by MemeBacktest with matching per-coin thresholds).

Reward per formula: mean per-coin MemeBacktest sharpe on 30m bars
(BPY=17520), invalid/const formulas penalized (-5/-2, mirrors engine).

Output routing (never mix):
  default (no TRAIN_12F_FLOOR): results/train_12f_30m_best.json {formula, decode, score, hist}
  floor run (TRAIN_12F_FLOOR set): results/train_12f_30m_floor_best.json {+ worst_leg, floor}
  smoke (TRAIN_12F_SMOKE=1): /tmp/train_12f_30m_smoke.json (real artifacts untouched)
Per-leg floor veto: TRAIN_12F_FLOOR=0.0 -> any leg sharpe<=0 vetoes (reward=worst leg);
veto branch also tracks best (fix 2026-09-16; was starved at -1e9).
Offline read-only (CSV reads only); no live/demo change.
Usage: /home/linuxbrew/.linuxbrew/bin/python3 research/train_12f_30m.py [steps] [batch]
Smoke (tests): TRAIN_12F_SMOKE=1 -> steps=3 batch=4 coins={ETC,TRX}.
"""
import csv
import json
import math
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
from research.universe_28c import ACCEPTANCE_GATE_28C, COINS_28C, COMMON_THRESHOLDS_30M
from research.data_contract_28c import HISTORY_YEARS_COMMON_28
from research.splits_28c import split_indices
from research.formula_grammar import ALL_TOKENS, update_depth, valid_token_mask
from research.causal_12f import causal_features, evaluate_formula
from research.acceptance_schema_28c import derive_live_adopted, make_acceptance

# ==================== 固定隨機種子 ====================
# 確保每次訓練結果可重現
import random
SEED = int(os.environ.get("TRAIN_SEED", "42"))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
# ==================== 固定隨機種子 ====================

COINS5 = ["ETC", "TRX", "ATOM", "APT", "KAS"]
USE_28C = (os.getenv("TRAIN_UNIVERSE_28C", "0").strip().lower() in {"1", "true", "yes"})
ACTIVE_COINS = list(COINS_28C) if USE_28C else list(COINS5)
# per-coin E10 thresholds (lth/sth/cd/sl) for reward backtest
TH = ({coin: tuple(COMMON_THRESHOLDS_30M[k] for k in ("long", "short", "cooldown_bars", "stop_loss"))
       for coin in ACTIVE_COINS} if USE_28C else
      {"ETC": (0.88, 0.12, 18, None), "TRX": (0.85, 0.12, 6, 0.05),
       "ATOM": (0.85, 0.15, 6, 0.05), "APT": (0.88, 0.12, 18, None),
       "KAS": (0.88, 0.12, 6, None)})
BPY30M = 17520.0
VOC = get_vocab(True)

def _softplus(x: float) -> float:
    x = max(-40.0, min(40.0, x))
    return math.log1p(math.exp(x))


def load_30m(coins):
    data_dir = os.getenv("TRAIN_DATA_DIR", "data/data_3y/30m" if USE_28C else "data/data_1y/30m")
    bars_all = {}
    for coin in coins:
        path = os.path.join(data_dir, f"{coin}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing training data: {path}")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        bars_all[coin] = [{"timestamp": int(float(r["timestamp"])),
                           "close": float(r["close"]), "open": float(r["open"]),
                           "high": float(r["high"]), "low": float(r["low"]),
                           "volume": float(r["volume"])} for r in rows]
    return bars_all


def trim_28c_to_train(bars_all):
    available = list(bars_all)
    # Smoke intentionally loads only the first two coins; do not apply the
    # full-universe contract split to that truncated smoke set.
    if len(available) != len(ACTIVE_COINS):
        if os.getenv("TRAIN_12F_SMOKE") == "1" and set(available) == set(ACTIVE_COINS[:2]):
            return bars_all
        raise ValueError(f"28c trim requires exact {len(ACTIVE_COINS)}-coin universe, got {len(available)}")
    common = set(x["timestamp"] for x in bars_all[available[0]])
    for coin in available[1:]:
        common &= set(x["timestamp"] for x in bars_all[coin])
    ts = sorted(common)
    split = split_indices(len(ts), ts[0], ts[-1])
    train_end_ts = ts[split["train"][1]]
    return {coin: [row for row in rows if row["timestamp"] < train_end_ts] for coin, rows in bars_all.items()}


def build_tensors(bars_all):
    coins = list(bars_all)
    if USE_28C:
        common = set(x["timestamp"] for x in bars_all[coins[0]])
        for c in coins[1:]:
            common &= set(x["timestamp"] for x in bars_all[c])
        timestamps = sorted(common)
        aligned = {}
        for c in coins:
            by_ts = {x["timestamp"]: x for x in bars_all[c]}
            aligned[c] = [by_ts[t] for t in timestamps]
        bars_all = aligned
        n = len(timestamps)
    else:
        n = min(len(b) for b in bars_all.values())
    def stack(key):
        return torch.tensor([[bars_all[c][i][key] for i in range(n)] for c in coins])
    raw = {"open": stack("open"), "high": stack("high"), "low": stack("low"),
           "close": stack("close"), "volume": stack("volume"),
           "liquidity": torch.full((len(coins), n), 1e7),
           "fdv": torch.full((len(coins), n), 1e8)}
    causal_maps = None
    if USE_28C:
        causal_bars = {
            c: {k: np.asarray([row[k] for row in bars_all[c]], dtype=np.float64)
                for k in ("open", "high", "low", "close", "volume")}
            for c in coins
        }
        causal_maps = {c: causal_features(causal_bars[c]) for c in coins}
        feats = torch.tensor(np.stack([causal_maps[c] for c in coins]), dtype=torch.float32)
    else:
        feats = FeatureEngineer.compute_features(raw, use_advanced=True)
    assert feats.shape[1] == 12, feats.shape
    tgt = []
    for c in coins:
        cl = [bars_all[c][i]["close"] for i in range(n)]
        tgt.append([(cl[i + 1] - cl[i]) / cl[i] if i < n - 1 else 0.0 for i in range(n)])
    return feats, torch.tensor(tgt), raw, coins, causal_maps


def decode(formula):
    names = VOC.token_names
    return [names[t] if 0 <= t < len(names) else f"?{t}" for t in formula]


def main():
    # 確保種子設定（防止嵌套模組重置）
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    smoke = os.getenv("TRAIN_12F_SMOKE") == "1"
    steps = 3 if smoke else (int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    bs = 4 if smoke else (int(sys.argv[2]) if len(sys.argv) > 2 else 64)
    coins = tuple(ACTIVE_COINS[:2]) if smoke else tuple(ACTIVE_COINS)
    # smoke never touches real artifacts; floor-constrained run gets its own file.
    # TRAIN_OUT allows exploratory runs to write a candidate without overwriting best.
    default_out = "/tmp/train_12f_30m_smoke.json" if smoke else (
        ("results/train_12f_30m_28c_floor_best.json" if USE_28C else "results/train_12f_30m_floor_best.json")
        if "TRAIN_12F_FLOOR" in os.environ else
        ("results/train_12f_30m_28c_best.json" if USE_28C else "results/train_12f_30m_best.json"))
    OUT = os.getenv("TRAIN_OUT", default_out)
    bars_all = load_30m(coins)
    if USE_28C:
        bars_all = trim_28c_to_train(bars_all)
    feats, target, raw, cnames, causal_maps = build_tensors(bars_all)
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
        stack_depths = [0] * bs
        for pos in range(ModelConfig.MAX_FORMULA_LEN):
            logits, value, _ = model(inp)
            if logits.shape[-1] != ALL_TOKENS:
                raise RuntimeError(f"vocab/model mismatch: {logits.shape[-1]} != {ALL_TOKENS}")
            masks = torch.stack([
                valid_token_mask(depth, pos, ModelConfig.MAX_FORMULA_LEN)
                for depth in stack_depths
            ]).to(logits.device)
            logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=logits)
            a = dist.sample()
            lps.append(dist.log_prob(a)); ents.append(dist.entropy())
            vals.append(value.squeeze(-1)); toks.append(a)
            stack_depths = [update_depth(depth, int(token))
                            for depth, token in zip(stack_depths, a.detach().cpu().tolist())]
            inp = torch.cat([inp, a.unsqueeze(1)], dim=1)
        if any(depth != 1 for depth in stack_depths):
            raise RuntimeError(f"grammar mask produced invalid final depths {stack_depths}")
        seqs = torch.stack(toks, dim=1)
        rewards = torch.zeros(bs, device=device)
        for i in range(bs):
            f = seqs[i].tolist()
            if USE_28C:
                signal_rows = [evaluate_formula(f, causal_maps[c]) for c in cnames]
                res = None if any(x is None for x in signal_rows) else torch.tensor(np.stack(signal_rows), dtype=torch.float32, device=device)
            else:
                res = vm.execute(f, feats)
            if res is None:
                rewards[i] = -100.0 if USE_28C else -5.0
                continue
            if res.std() < 1e-4:
                rewards[i] = -50.0 if USE_28C else -2.0
                continue
            scores = []
            worst = 1e9
            for j, c in enumerate(cnames):
                lth, sth, cd, sl = TH[c]
                bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                                  funding_override=0.0005, long_th=lth, short_th=sth,
                                  cooldown_bars=cd, bars_per_year=BPY30M, stop_loss=sl)
                one = {k: (v[j:j + 1] if isinstance(v, torch.Tensor) else v) for k, v in raw_d.items()}
                sc, _ = bt.evaluate(res[j:j + 1], one, target[j:j + 1])
                if USE_28C:
                    m = bt.last_metrics
                    _s = float(m.get("sharpe", 0.0)) - 0.10 * max(0.0, float(m.get("turnover", 0.0)) - 0.02)
                else:
                    _s = float(sc.item()) if isinstance(sc, torch.Tensor) else float(sc)
                scores.append(_s)
                worst = min(worst, _s)
            mean_s = sum(scores) / len(scores)
            leg_floor = float(os.getenv("TRAIN_12F_FLOOR", "0.0"))
            if USE_28C:
                # Smooth floor penalty; no hard -10 cliff.
                tau = 0.5
                floor_penalty = tau * _softplus((leg_floor - worst) / tau)
                rewards[i] = mean_s - 0.5 * floor_penalty
            elif worst <= leg_floor:
                rewards[i] = worst
            else:
                rewards[i] = mean_s
            if float(rewards[i]) > best_score:
                best_score = float(rewards[i]); best_formula = f; best_worst = worst
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        pl = torch.stack([-lp * adv for lp in lps], dim=0).sum(dim=0).mean()
        vl = F.mse_loss(torch.stack(vals, dim=0).mean(dim=0), rewards)
        # Entropy annealing: start high, decay over time
        ent_coeff_start = float(os.getenv("TRAIN_ENT_START", "0.05"))
        ent_coeff_end = float(os.getenv("TRAIN_ENT_END", "0.01"))
        ent_coeff = ent_coeff_start + (ent_coeff_end - ent_coeff_start) * (step / max(1, steps - 1))
        # LR annealing
        lr_start = float(os.getenv("TRAIN_LR_START", "1e-3"))
        lr_end = float(os.getenv("TRAIN_LR_END", "1e-4"))
        lr = lr_start + (lr_end - lr_start) * (step / max(1, steps - 1))
        for pg in opt.param_groups:
            pg['lr'] = lr
        loss = pl + 0.5 * vl - ent_coeff * torch.stack(ents, dim=0).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        avg = rewards.mean().item()
        hist.append({"step": step, "avg": round(avg, 3), "best": round(best_score, 3)})
        if step % 20 == 0:
            print(f"step {step} avg={avg:.3f} best={best_score:.3f} formula={best_formula} decode={decode(best_formula) if best_formula else None}", flush=True)
    # Fail-closed, and honest about why. This trainer never evaluates the
    # regime gate and its reward is not equity-compound-v2, so the verdict is
    # always `fail` with a reason rather than a bare False.
    acceptance = make_acceptance(
        "fail",
        ("no regime gate evaluated: this trainer's reward uses the legacy "
         "MemeBacktest path (additive equity, bar-frequency Sharpe), not the "
         "shared equity-compound-v2 accounting, so no gate verdict can be "
         "derived here. Run research/ga_28c_30m_3y.py or the standalone gate "
         "in research/regime_gate_28c.py to decide acceptance."),
        min_positive_coins=ACCEPTANCE_GATE_28C["min_positive_coins"] if USE_28C else 0,
        oos_mdd=False, oos_solvent=False, oos_portfolio_sharpe=0.0,
        legacy_bar=ACCEPTANCE_GATE_28C["min_oos_portfolio_sharpe"] if USE_28C else 0.0,
        criteria={})
    out = {"formula": best_formula, "decode": decode(best_formula) if best_formula else None,
           "score": best_score, "worst_leg": best_worst, "floor": float(os.getenv("TRAIN_12F_FLOOR", "0.0")),
           "hist": hist, "vocab": "12f",
           "coins": list(cnames), "bars": f"30m-{HISTORY_YEARS_COMMON_28}y" if USE_28C else "30m-1y", "bpy": BPY30M,
           "universe_mode": "28c_common_transfer" if USE_28C else "legacy_top5",
           "threshold_source": "common_28c_transfer_default" if USE_28C else "legacy_tuned_top5",
           "acceptance_gate": ACCEPTANCE_GATE_28C if USE_28C else None,
           "data_dir": os.getenv("TRAIN_DATA_DIR", "data/data_3y/30m" if USE_28C else "data/data_1y/30m"),
           "history_years": HISTORY_YEARS_COMMON_28 if USE_28C else 1,
           "contract_version": "data-contract-28c-v1" if USE_28C else None,
           "train_bars": int(feats.shape[-1]) if USE_28C else None,
           "lockbox_excluded": bool(USE_28C),
           "status": "training_candidate" if USE_28C else "legacy_training",
           "acceptance_passed": acceptance["verdict"] == "pass",
           "live_adopted": derive_live_adopted(acceptance),
           # This trainer's reward still runs through MemeBacktest, which
           # uses additive equity and bar-frequency Sharpe, NOT the shared
           # equity-compound-v2 accounting. Its `score` is therefore a
           # diagnostic, and no gate verdict can be derived from it here.
           # The record says so explicitly instead of relying on a hardcoded
           # False that reads as if a gate had run and passed.
           "acceptance": acceptance,
           "acceptance_gate_run": False,
           "causal_features": bool(USE_28C),
           "training_caveat": "28c mode uses causal feature/operator path but still requires OOS/acceptance gates before adoption" if USE_28C else "legacy_top5"}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    print(f"DONE best={best_score:.3f} formula={best_formula} decode={out['decode']}")


if __name__ == "__main__":
    main()
