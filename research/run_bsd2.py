"""BSD-2: both-legged AlphaGPT training on BTC+SOL+DOGE front-85% 4h bars.

Reward = MemeBacktest BOTH-LEGGED: aster perp 2x, fund 0.0005, lth 0.85,
sth 0.15, cooldown 6, 4h annualization (2190). No stop-loss (matches BSD
spec: aster perp 2x fund 0.0005, lth 0.85 sth 0.15 cd 6).
Train coins: BTC + SOL + DOGE (ASTER excluded: short history).
OOS verify top3 per coin on LAST-15% segments: BTC/SOL/DOGE 986-bar OOS
H1/H2 (493/493) + ASTER 317-bar OOS H1/H2 (158/159), default thresholds
(same as training: lth 0.85 sth 0.15 cd 6). Compare vs BSD-1 baseline
formula [3,2,7,2,7,11,15,4,4,6,6,10] on same segments.
Usage: /home/linuxbrew/.linuxbrew/bin/python3 run_bsd2.py [steps] [batch]
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
from model_core.vocab import FORMULA_VOCAB

LONG_TH = 0.85
SHORT_TH = 0.15
COOLDOWN = 6
TRAIN_FRAC = 0.85
BARS_PER_YEAR = 2190.0
BSD1_FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
TRAIN_COINS = ("BTC", "SOL", "DOGE")
VERIFY_COINS = ("BTC", "SOL", "DOGE", "ASTER")

def make_bt():
    return MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                        funding_override=0.0005, long_th=LONG_TH, short_th=SHORT_TH,
                        cooldown_bars=COOLDOWN, bars_per_year=BARS_PER_YEAR)

def load_4h(coins):
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

def build_single(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    feats = FeatureEngineer.compute_features(raw)
    cl = [b["close"] for b in bars]
    tgt = torch.tensor([[(cl[i+1]-cl[i])/cl[i] if i < n-1 else 0.0 for i in range(n)]])
    return feats, tgt, raw

def decode(formula):
    names = FORMULA_VOCAB.token_names
    return [names[t] if 0 <= t < len(names) else f"UNK{t}" for t in formula]

def seg_eval(vm, formula, feats, tgt, raw, device):
    out = {}
    T = feats.shape[2] if feats.dim() == 3 else feats.shape[1]
    h = T // 2
    res = vm.execute(formula, feats.to(device))
    if res is None:
        return None
    tgt_d = tgt.to(device)
    rawd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    for name, s, e in [("FULL", 0, T), ("H1", 0, h), ("H2", h, T)]:
        def sl(t):
            return t[:, s:e] if t.dim() == 2 else t[:, :, s:e]
        bt = make_bt()
        fit, cum = bt.evaluate(sl(res), {k: sl(v) for k, v in rawd.items()}, sl(tgt_d))
        m = dict(bt.last_metrics)
        wlen = e - s
        ann = float(cum) * (BARS_PER_YEAR / wlen)
        out[name] = {"fitness": float(fit.item()) if isinstance(fit, torch.Tensor) else float(fit),
                     "sharpe": round(m["sharpe"], 3), "ann": round(ann, 4),
                     "mdd": round(m["max_dd"], 4), "cum": round(float(cum), 4)}
    return out

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    bars_all = load_4h(TRAIN_COINS)
    n = min(len(b) for b in bars_all.values())
    n_train = int(n * TRAIN_FRAC)
    n_oos = n - n_train
    print(f"total 4h bars={n} train={n_train} oos={n_oos} (H1/H2={n_oos//2})", flush=True)
    train_bars = {c: bars_all[c][:n_train] for c in TRAIN_COINS}
    feats, target, raw, cnames = build_tensors(train_bars)
    print(f"train feats {tuple(feats.shape)} coins={cnames}", flush=True)
    device = ModelConfig.DEVICE
    feats, target = feats.to(device), target.to(device)
    raw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    model = AlphaGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = StackVM()
    best_score, best_formula = -1e9, None
    top3 = []
    hist = []
    def consider(score, formula):
        if any(f == formula for _, f in top3):
            return
        top3.append((score, formula))
        top3.sort(key=lambda x: -x[0])
        del top3[3:]
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
            bt = make_bt()
            score, _ = bt.evaluate(res, raw, target)
            rewards[i] = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
            if float(rewards[i]) > best_score:
                best_score = float(rewards[i]); best_formula = seqs[i].tolist()
            consider(float(rewards[i]), seqs[i].tolist())
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
    with open("results/train_bsd_best.json", "w") as f:
        json.dump({"formula": best_formula, "decoded": decode(best_formula),
                   "score": best_score, "hist": hist,
                   "top3": [{"formula": fl, "decoded": decode(fl), "train_score": sc}
                            for sc, fl in top3],
                   "config": {"coins": list(TRAIN_COINS), "n_train": n_train,
                              "long_th": LONG_TH, "short_th": SHORT_TH,
                              "cooldown": COOLDOWN, "short_enabled": True,
                              "venue": "aster", "leverage": 2.0,
                              "funding_override": 0.0005, "steps": steps,
                              "batch": bs, "train_frac": TRAIN_FRAC}}, f)
    print(f"TRAIN DONE best={best_score:.3f} formula={best_formula} decoded={decode(best_formula)}", flush=True)
    # ---- OOS verify top3 per coin + BSD-1 baseline comparison ----
    all_bars = load_4h(VERIFY_COINS)
    report = {"bsd1_formula": BSD1_FORMULA, "bsd1_decoded": decode(BSD1_FORMULA),
              "config": {"long_th": LONG_TH, "short_th": SHORT_TH, "cooldown": COOLDOWN,
                         "short_enabled": True, "venue": "aster", "leverage": 2.0,
                         "funding_override": 0.0005},
              "per_coin": {}}
    for coin in VERIFY_COINS:
        bars = all_bars[coin]
        nn = len(bars)
        no = nn - int(nn * TRAIN_FRAC)
        oos = bars[nn - no:]
        f, t, r = build_single(oos)
        T = f.shape[2] if f.dim() == 3 else f.shape[1]
        print(f"OOS {coin}: bars={nn} oos={T} H1/H2={T//2}/{T-T//2}", flush=True)
        centry = {"oos_bars": T, "results": [], "baseline": None}
        for sc, fl in top3:
            ev = seg_eval(vm, fl, f, t, r, device)
            e = {"formula": fl, "decoded": decode(fl), "train_score": sc}
            if ev is None:
                e["error"] = "vm_failed_oos"
            else:
                e.update(ev)
            centry["results"].append(e)
            print("OOS " + coin + " " + json.dumps(e), flush=True)
        bev = seg_eval(vm, BSD1_FORMULA, f, t, r, device)
        b = {"formula": BSD1_FORMULA, "decoded": decode(BSD1_FORMULA)}
        if bev is None:
            b["error"] = "vm_failed_oos"
        else:
            b.update(bev)
        centry["baseline"] = b
        print("BASE " + coin + " " + json.dumps(b), flush=True)
        report["per_coin"][coin] = centry
    with open("results/backtest_BSD2_train.json", "w") as f:
        json.dump(report, f, indent=1)
    print("OOS DONE -> backtest_BSD2_train.json", flush=True)

if __name__ == "__main__":
    main()
