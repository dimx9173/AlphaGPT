"""Q2: fresh BOTH-legged AlphaGPT training on BTC+SOL+ETC+TRX+DOGE front-85% 4h bars.

Reward = MemeBacktest BOTH-LEGGED, fitness = MEAN over 5 coins front-85%
(5584 bars): aster perp 2x, fund 0.0005, lth 0.85, sth 0.15, cooldown 6,
4h annualization (2190). No stop-loss in training (matches BSD-2 spec).
Keep top3 by train fitness. OOS-verify each on all 5 coins LAST-15%
segments (986 bars: FULL + H1/H2 493/493) with default (0.85/0.15/cd6)
AND with Q1 per-coin H1-best params (backtest_BSD1_baseline.json where
available: BTC/SOL/DOGE; else default). Side split (both/long/short) on H2.
Compare vs BSD-1 baseline formula on same segments.
Usage: /home/linuxbrew/.linuxbrew/bin/python3 run_q2.py [steps] [batch]
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
BASELINE_FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
TRAIN_COINS = ("BTC", "SOL", "ETC", "TRX", "DOGE")


def load_q1_params():
    q1 = {}
    try:
        d = json.load(open("results/backtest_BSD1_baseline.json"))
        for c, info in d.get("coins", {}).items():
            b = info.get("best", {})
            q1[c] = {"lth": b.get("lth", LONG_TH), "sth": b.get("sth", SHORT_TH),
                     "cd": b.get("cd", COOLDOWN), "sl": b.get("sl")}
    except Exception as e:
        print("Q1 params unavailable:", e, flush=True)
    return q1


Q1 = load_q1_params()


def make_bt(lth=LONG_TH, sth=SHORT_TH, cd=COOLDOWN, sl=None, side="both"):
    if side == "long":
        return MemeBacktest(venue="aster", leverage=2.0, short_enabled=False,
                            funding_override=0.0005, long_th=lth, short_th=sth,
                            cooldown_bars=cd, bars_per_year=BARS_PER_YEAR,
                            stop_loss=sl)
    if side == "short":
        # long_th=10.0 -> sigmoid never exceeds -> short leg only
        return MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                            funding_override=0.0005, long_th=10.0, short_th=sth,
                            cooldown_bars=cd, bars_per_year=BARS_PER_YEAR,
                            stop_loss=sl)
    return MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                        funding_override=0.0005, long_th=lth, short_th=sth,
                        cooldown_bars=cd, bars_per_year=BARS_PER_YEAR,
                        stop_loss=sl)


def load_4h(coins):
    bars_all = {}
    for coin in coins:
        with open(f"data/data_15m_3y/{coin}.csv") as f:
            rows = list(csv.DictReader(f))
        bars = []
        for i in range(0, len(rows), 16):
            blk = rows[i:i + 16]
            if len(blk) < 16:
                break
            bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                         "high": max(float(x["high"]) for x in blk),
                         "low": min(float(x["low"]) for x in blk),
                         "volume": sum(float(x["volume"]) for x in blk)})
        bars_all[coin] = bars
    return bars_all


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
    tgt = torch.tensor([[(cl[i + 1] - cl[i]) / cl[i] if i < n - 1 else 0.0 for i in range(n)]])
    return feats, tgt, raw


def decode(formula):
    names = FORMULA_VOCAB.token_names
    return [names[t] if 0 <= t < len(names) else f"UNK{t}" for t in formula]


def fit_of(fit):
    return float(fit.item()) if isinstance(fit, torch.Tensor) else float(fit)


def eval_segs(vm, formula, feats, tgt, raw, device, lth, sth, cd, sl, side="both",
              segs=("FULL", "H1", "H2")):
    res = vm.execute(formula, feats.to(device))
    if res is None:
        return None
    T = feats.shape[2] if feats.dim() == 3 else feats.shape[1]
    h = T // 2
    bounds = {"FULL": (0, T), "H1": (0, h), "H2": (h, T)}
    tgt_d = tgt.to(device)
    rawd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    out = {}
    for name in segs:
        s, e = bounds[name]

        def cut(t, _s=s, _e=e):
            return t[:, _s:_e] if t.dim() == 2 else t[:, :, _s:_e]

        bt = make_bt(lth, sth, cd, sl, side)
        fit, cum = bt.evaluate(cut(res), {k: cut(v) for k, v in rawd.items()}, cut(tgt_d))
        m = dict(bt.last_metrics)
        wlen = e - s
        ann = float(cum) * (BARS_PER_YEAR / wlen)
        out[name] = {"fitness": fit_of(fit), "sharpe": round(m["sharpe"], 3),
                     "ann": round(ann, 4), "mdd": round(m["max_dd"], 4),
                     "cum": round(float(cum), 4)}
    return out


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    bars_all = load_4h(TRAIN_COINS)
    n = min(len(b) for b in bars_all.values())
    n_train = int(n * TRAIN_FRAC)
    n_oos = n - n_train
    print(f"total 4h bars={n} train={n_train} oos={n_oos} (H1/H2={n_oos // 2})", flush=True)
    print(f"Q1 params: {json.dumps(Q1)}", flush=True)
    device = ModelConfig.DEVICE
    # STRICT SPLIT: per-coin train tensors from front-85% only
    train_T = {}
    for c in TRAIN_COINS:
        f, t, r = build_single(bars_all[c][:n_train])
        train_T[c] = (f.to(device), t.to(device),
                      {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in r.items()})
    print(f"train bars/coin={n_train} coins={list(TRAIN_COINS)} device={device}", flush=True)
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

    def mean_fitness(seq):
        tot, cnt = 0.0, 0
        for c in TRAIN_COINS:
            f, t, r = train_T[c]
            res = vm.execute(seq, f)
            if res is None:
                return -5.0
            if res.std() < 1e-4:
                return -2.0
            bt = make_bt()
            score, _ = bt.evaluate(res, r, t)
            tot += fit_of(score)
            cnt += 1
        return tot / max(cnt, 1)

    for step in tqdm(range(steps)):
        inp = torch.zeros((bs, 1), dtype=torch.long, device=device)
        lps, toks, vals, ents = [], [], [], []
        for _ in range(ModelConfig.MAX_FORMULA_LEN):
            logits, value, _ = model(inp)
            dist = Categorical(logits=logits)
            a = dist.sample()
            lps.append(dist.log_prob(a))
            ents.append(dist.entropy())
            vals.append(value.squeeze(-1))
            toks.append(a)
            inp = torch.cat([inp, a.unsqueeze(1)], dim=1)
        seqs = torch.stack(toks, dim=1)
        rewards = torch.zeros(bs, device=device)
        for i in range(bs):
            r = mean_fitness(seqs[i].tolist())
            rewards[i] = r
            if r > best_score:
                best_score = r
                best_formula = seqs[i].tolist()
            consider(r, seqs[i].tolist())
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        pl = torch.stack([-lp * adv for lp in lps], dim=0).sum(dim=0).mean()
        vl = F.mse_loss(torch.stack(vals, dim=0).mean(dim=0), rewards)
        loss = pl + 0.5 * vl - 0.01 * torch.stack(ents, dim=0).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        avg = rewards.mean().item()
        hist.append({"step": step, "avg": round(avg, 3), "best": round(best_score, 3)})
        if step % 20 == 0:
            print(f"step {step} avg={avg:.3f} best={best_score:.3f} formula={best_formula}",
                  flush=True)
    with open("results/train_q5_best.json", "w") as f:
        json.dump({"formula": best_formula, "decoded": decode(best_formula),
                   "score": best_score, "hist": hist,
                   "top3": [{"formula": fl, "decoded": decode(fl), "train_score": sc}
                            for sc, fl in top3],
                   "config": {"coins": list(TRAIN_COINS), "n_train": n_train,
                              "long_th": LONG_TH, "short_th": SHORT_TH,
                              "cooldown": COOLDOWN, "short_enabled": True,
                              "venue": "aster", "leverage": 2.0,
                              "funding_override": 0.0005, "steps": steps,
                              "batch": bs, "train_frac": TRAIN_FRAC,
                              "fitness": "mean_over_5_coins"}}, f)
    print(f"TRAIN DONE best={best_score:.3f} formula={best_formula} "
          f"decoded={decode(best_formula)}", flush=True)
    # ---- OOS verify top3 + baseline per coin ----
    report = {"baseline_formula": BASELINE_FORMULA,
              "baseline_decoded": decode(BASELINE_FORMULA),
              "config": {"long_th": LONG_TH, "short_th": SHORT_TH, "cooldown": COOLDOWN,
                         "short_enabled": True, "venue": "aster", "leverage": 2.0,
                         "funding_override": 0.0005},
              "q1_params": Q1,
              "per_coin": {}}
    all_bars = load_4h(TRAIN_COINS)

    def verify_one(fl):
        entry = {"formula": fl, "decoded": decode(fl)}
        return entry

    for coin in TRAIN_COINS:
        bars = all_bars[coin]
        nn = len(bars)
        no = nn - int(nn * TRAIN_FRAC)
        oos = bars[nn - no:]
        f, t, r = build_single(oos)
        T = f.shape[2] if f.dim() == 3 else f.shape[1]
        q = Q1.get(coin)
        print(f"OOS {coin}: bars={nn} oos={T} H1/H2={T // 2}/{T - T // 2} q1={q}", flush=True)
        centry = {"oos_bars": T, "q1_params": q, "results": [], "baseline": None}
        cands = [(sc, fl) for sc, fl in top3]
        for sc, fl in cands:
            e = verify_one(fl)
            e["train_score"] = sc
            e["default"] = eval_segs(vm, fl, f, t, r, device, LONG_TH, SHORT_TH,
                                     COOLDOWN, None)
            if q is not None:
                e["q1"] = eval_segs(vm, fl, f, t, r, device, q["lth"], q["sth"],
                                    q["cd"], q["sl"])
            else:
                e["q1"] = e["default"]
            e["side_default_H2"] = {
                s: (eval_segs(vm, fl, f, t, r, device, LONG_TH, SHORT_TH, COOLDOWN,
                              None, side=s, segs=("H2",)) or {}).get("H2")
                for s in ("both", "long", "short")}
            if q is not None:
                e["side_q1_H2"] = {
                    s: (eval_segs(vm, fl, f, t, r, device, q["lth"], q["sth"], q["cd"],
                                  q["sl"], side=s, segs=("H2",)) or {}).get("H2")
                    for s in ("both", "long", "short")}
            else:
                e["side_q1_H2"] = e["side_default_H2"]
            centry["results"].append(e)
            print("OOS " + coin + " " + json.dumps(e), flush=True)
        b = verify_one(BASELINE_FORMULA)
        b["default"] = eval_segs(vm, BASELINE_FORMULA, f, t, r, device, LONG_TH,
                                 SHORT_TH, COOLDOWN, None)
        if q is not None:
            b["q1"] = eval_segs(vm, BASELINE_FORMULA, f, t, r, device, q["lth"],
                                q["sth"], q["cd"], q["sl"])
        else:
            b["q1"] = b["default"]
        b["side_default_H2"] = {
            s: (eval_segs(vm, BASELINE_FORMULA, f, t, r, device, LONG_TH, SHORT_TH,
                          COOLDOWN, None, side=s, segs=("H2",)) or {}).get("H2")
            for s in ("both", "long", "short")}
        if q is not None:
            b["side_q1_H2"] = {
                s: (eval_segs(vm, BASELINE_FORMULA, f, t, r, device, q["lth"], q["sth"],
                              q["cd"], q["sl"], side=s, segs=("H2",)) or {}).get("H2")
                for s in ("both", "long", "short")}
        else:
            b["side_q1_H2"] = b["side_default_H2"]
        centry["baseline"] = b
        print("BASE " + coin + " " + json.dumps(b), flush=True)
        report["per_coin"][coin] = centry
    with open("results/backtest_Q2_train.json", "w") as f:
        json.dump(report, f, indent=1)
    print("OOS DONE -> backtest_Q2_train.json", flush=True)


if __name__ == "__main__":
    main()
