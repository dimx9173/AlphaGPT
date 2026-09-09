"""BSD-3: both-legged (long+short) random formula search on BTC/SOL/DOGE 4h + ASTER transfer test."""
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
THS = [0.85, 0.88]
STH = 0.12
CD = 6

def rand_formula(rng, length=12):
    toks, depth = [], 0
    for i in range(length):
        remaining = length - i
        if depth < 2:
            toks.append(rng.randrange(FEATS)); depth += 1
        else:
            if rng.random() < 0.45:
                oi = rng.randrange(len(OPS_CONFIG)); ar = ARITY[oi]
                if ar <= depth and (depth - ar + 1) + (remaining - 1) >= 1:
                    toks.append(OFF + oi); depth = depth - ar + 1
                else:
                    toks.append(rng.randrange(FEATS)); depth += 1
            else:
                toks.append(rng.randrange(FEATS)); depth += 1
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

def mkbt(lth, sth=STH, cd=CD, side="both"):
    if side == "long":
        return MemeBacktest(venue="aster", leverage=2.0, short_enabled=False,
                            funding_override=0.0005, long_th=lth, short_th=sth,
                            cooldown_bars=cd, bars_per_year=2190.0)
    if side == "short":
        return MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                            funding_override=0.0005, long_th=2.0, short_th=sth,
                            cooldown_bars=cd, bars_per_year=2190.0)
    return MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                        funding_override=0.0005, long_th=lth, short_th=sth,
                        cooldown_bars=cd, bars_per_year=2190.0)

def ev(vm, ft, mat, lth, side="both"):
    sig = vm.execute(ft, mat[2])
    if sig is None: return None
    bt = mkbt(lth, side=side)
    fit, cum = bt.evaluate(sig, mat[0], mat[1])
    m = bt.last_metrics
    return {"sharpe": round(m["sharpe"],3), "ann": round(m["sharpe"]*0+ (cum/mat[3]*2190.0),3),
            "mdd": round(m["max_dd"],3), "cum": round(float(cum),3), "n": mat[3],
            "turn": round(m["turnover"],5), "lturn": round(m["long_turnover"],5),
            "sturn": round(m["short_turnover"],5)}

def search(coin, mats_h1, target_valid, seed, max_tried=6000):
    rng = random.Random(seed); vm = StackVM()
    cands, valid, tried = [], 0, 0
    while valid < target_valid and tried < max_tried:
        tried += 1
        ft = rand_formula(rng, 12)
        sig = vm.execute(ft, mats_h1[2])
        if sig is None: continue
        for lth in THS:
            r = ev(vm, ft, mats_h1, lth, "both")
            if r is None: continue
            if r["turn"] == 0.0 and r["cum"] == 0.0: continue
            cands.append({"formula": ft, "lth": lth, "h1": r})
            valid += 1
            if valid >= target_valid: break
        if valid and valid % 200 == 0:
            print(f"[{coin}] valid={valid} tried={tried}", flush=True)
    print(f"[{coin}] DONE valid={valid} tried={tried}", flush=True)
    cands.sort(key=lambda r: r["h1"]["sharpe"], reverse=True)
    return cands, valid, tried

def verify_top(vm, top, mats, coin):
    out = []
    for i, c in enumerate(top):
        ft, lth = c["formula"], c["lth"]
        row = {"rank": i+1, "formula": ft, "lth": lth, "h1": c["h1"]}
        for seg in ["H2", "FULL"]:
            row[seg.lower()] = ev(vm, ft, mats[seg], lth, "both")
        ss = {}
        for side in ["both", "long", "short"]:
            ss[side] = ev(vm, ft, mats["H2"], lth, side)
        row["side_split_h2"] = ss
        out.append(row)
        print(f"[{coin}] #{i+1} f={ft} lth={lth} H1sh={c['h1']['sharpe']} H2={row['h2']} FULLcum={row['full']['cum']} sides=" +
              json.dumps({k: (v["sharpe"], v["cum"]) for k, v in ss.items()}), flush=True)
    return out

def baseline(vm, mats, coin):
    base = {"name": "grouptest_baseline_both", "formula": BASE_FORMULA}
    for seg in ["H1", "H2", "FULL"]:
        base[seg.lower()] = {}
        for lth in THS:
            base[seg.lower()][str(lth)] = ev(vm, BASE_FORMULA, mats[seg], lth, "both")
    print(f"[{coin}] BASELINE: " + json.dumps(base), flush=True)
    return base

result = {"coins": {}, "aster": {}}
vm = StackVM()

for ci, coin in enumerate(["BTC", "SOL", "DOGE"]):
    bars = load_bars(coin); n = len(bars)
    oos = bars[int(n*0.85):]; h = len(oos)//2
    print(f"{coin} 4h bars={n} OOS H1={len(oos[:h])} H2={len(oos[h:])}", flush=True)
    segs = {"H1": oos[:h], "H2": oos[h:], "FULL": bars}
    mats = {k: build_mats(b) for k, b in segs.items()}
    cands, valid, tried = search(coin, mats["H1"], 800, seed=33+ci)
    top = cands[:10]
    out = verify_top(vm, top, mats, coin)
    result["coins"][coin] = {"top10": out, "baseline": baseline(vm, mats, coin),
        "config": {"coin": coin, "tf": "4h", "both_legs": True, "venue": "aster",
                   "lev": 2.0, "fund": 0.0005, "lth": THS, "sth": STH, "cd": CD,
                   "search_seg": "H1", "n_bars": n, "valid": valid, "tried": tried}}

# ASTER: H1/H2 = 158/158 (OOS 15% of 2107 = 316)
abars = load_bars("ASTER"); an = len(abars)
aoos = abars[int(an*0.85):]; ah = len(aoos)//2
print(f"ASTER 4h bars={an} OOS H1={len(aoos[:ah])} H2={len(aoos[ah:])}", flush=True)
asegs = {"H1": aoos[:ah], "H2": aoos[ah:], "FULL": abars}
amats = {k: build_mats(b) for k, b in asegs.items()}

# transfer: BTC top10 on ASTER segs
transfer = []
for c in result["coins"]["BTC"]["top10"]:
    ft, lth = c["formula"], c["lth"]
    row = {"formula": ft, "lth": lth, "btc_h1_sharpe": c["h1"]["sharpe"],
           "btc_h2_sharpe": c["h2"]["sharpe"]}
    for seg in ["H1", "H2", "FULL"]:
        row[seg.lower()] = ev(vm, ft, amats[seg], lth, "both")
    ss = {s: ev(vm, ft, amats["H2"], lth, s) for s in ["both", "long", "short"]}
    row["side_split_h2"] = ss
    transfer.append(row)
    print(f"[ASTER-transfer] f={ft} lth={lth} H1={row['h1']} H2={row['h2']} FULLcum={row['full']['cum']}", flush=True)
result["aster"]["transfer_from_btc"] = transfer

# ASTER-native search (smaller: 300 valid, H1 only 158 bars so noisy but cheap)
ncands, nvalid, ntried = search("ASTER-native", amats["H1"], 300, seed=99)
ntop = verify_top(vm, ncands[:5], amats, "ASTER-native")
result["aster"]["native_top5"] = ntop
result["aster"]["native_search"] = {"valid": nvalid, "tried": ntried,
    "h1_bars": amats["H1"][3], "h2_bars": amats["H2"][3], "n_bars": an}

json.dump(result, open("results/backtest_BSD3_search.json", "w"), indent=1)
print("wrote backtest_BSD3_search.json", flush=True)
