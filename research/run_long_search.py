"""Long-engine random search: LONG-ONLY formulas on BTC 4h, unbiased H1 search -> H2+FULL verify."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math, random
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.ops import OPS_CONFIG

BASE_FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
FEATS = 6
ARITY = [c[2] for c in OPS_CONFIG]
OFF = 6
random.seed(7)

def rand_formula(length=12):
    toks, depth = [], 0
    for i in range(length):
        remaining = length - i
        if depth < 2:
            t = random.randrange(FEATS); toks.append(t); depth += 1
        else:
            if random.random() < 0.45:
                oi = random.randrange(len(OPS_CONFIG))
                ar = ARITY[oi]
                if ar <= depth and (depth - ar + 1) + (remaining - 1) >= 1:
                    toks.append(OFF + oi); depth = depth - ar + 1
                else:
                    t = random.randrange(FEATS); toks.append(t); depth += 1
            else:
                t = random.randrange(FEATS); toks.append(t); depth += 1
    while depth > 1:
        toks.append(OFF + 0); depth -= 1
    return toks

def load_bars(coin):
    rows = list(csv.DictReader(open(f"data/data_15m_3y/{coin}.csv")))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                     "high": max(float(x["high"]) for x in blk),
                     "low": min(float(x["low"]) for x in blk),
                     "volume": sum(float(x["volume"]) for x in blk)})
    return bars

def build_mats(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    feats = FeatureEngineer.compute_features(raw)
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), feats, n

def net_series(raw, rets_t, sig, lth, cd, side="long"):
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=False,
                      funding_override=0.0005, long_th=lth, short_th=0.12,
                      cooldown_bars=cd, bars_per_year=2190.0)
    signal = torch.sigmoid(sig)
    is_safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = torch.zeros_like(lp)
    lp, sp = bt._apply_cooldown(lp, sp)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw["liquidity"] + 1e-9), 0.0, 0.05))
    gross = lp * rets_t * bt.leverage
    fnd = lp * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    return net

def stats(ser):
    n = len(ser); mean = sum(ser)/n
    var = sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe = mean/math.sqrt(var)*math.sqrt(2190.0) if var > 0 else 0.0
    cum = sum(ser); ann = cum/n*2190.0
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in ser:
        cs += x; peak = max(peak, cs); mdd = max(mdd, peak-cs)
    turn = sum(abs(ser[i]-ser[i-1]) for i in range(1,n))/n
    return {"sharpe": round(sharpe,3), "ann": round(ann,3), "mdd": round(mdd,3), "cum": round(cum,3), "n": n}

bars = load_bars("BTC")
n = len(bars)
print(f"BTC 4h bars={n}", flush=True)
oos = bars[int(n*0.85):]
h = len(oos)//2
print(f"OOS H1={len(oos[:h])} H2={len(oos[h:])}", flush=True)
segs = {"H1": oos[:h], "H2": oos[h:], "FULL": bars}
mats = {}
for k, b in segs.items():
    mats[k] = build_mats(b)
    print(f"mats {k} n={mats[k][3]}", flush=True)

THS = [0.85, 0.88]
vm = StackVM()
cands, valid, tried = [], 0, 0
N = 1200
while valid < 1000 and tried < 4000:
    tried += 1
    ft = rand_formula(12)
    sig_h1 = vm.execute(ft, mats["H1"][2])
    if sig_h1 is None: continue
    # slice signal to H1 length (feats computed on H1 bars)
    for lth in THS:
        ser = net_series(mats["H1"][0], mats["H1"][1], sig_h1, lth, 6)
        st = stats(ser)
        # activity check: need real trading
        if st["cum"] == 0.0 and st["mdd"] == 0.0: continue
        cands.append({"formula": ft, "lth": lth, "h1": st})
        valid += 1
    if valid % 100 == 0:
        print(f"valid={valid} tried={tried}", flush=True)
print(f"valid={valid} tried={tried}", flush=True)
cands.sort(key=lambda r: r["h1"]["sharpe"], reverse=True)
top = cands[:10]
# verify top10 on H2 + FULL (recompute feats per segment, fresh signal)
out = []
for i, c in enumerate(top):
    ft, lth = c["formula"], c["lth"]
    row = {"rank": i+1, "formula": ft, "lth": lth, "h1": c["h1"]}
    for seg in ["H2", "FULL"]:
        sig = vm.execute(ft, mats[seg][2])
        ser = net_series(mats[seg][0], mats[seg][1], sig, lth, 6)
        row[seg.lower()] = stats(ser)
    # turnover from backtest metrics on H1
    out.append(row)
    print(f"#{i+1} f={ft} lth={lth} H1={c['h1']} H2={row['h2']} FULL={row['full']}", flush=True)

# baseline: BASE_FORMULA long-leg on same segments
base = {"name": "grouptest_baseline_long", "formula": BASE_FORMULA}
for seg in ["H1", "H2", "FULL"]:
    sig = vm.execute(BASE_FORMULA, mats[seg][2])
    for lth in [0.85, 0.88]:
        ser = net_series(mats[seg][0], mats[seg][1], sig, lth, 6)
        base.setdefault(seg.lower(), {})[str(lth)] = stats(ser)
print("BASELINE:", json.dumps(base), flush=True)
json.dump({"top10": out, "baseline": base,
           "config": {"coin": "BTC", "tf": "4h", "long_only": True, "venue": "aster",
                      "lev": 2.0, "fund": 0.0005, "cd": 6, "search_seg": "H1",
                      "n_bars": n, "valid": valid, "tried": tried}},
          open("results/backtest_long_search.json", "w"), indent=1)
print("wrote backtest_long_search.json", flush=True)
