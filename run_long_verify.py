"""Verify turnover + ETH secondary for long-search top10. Appends fields into backtest_long_search.json."""
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

def load_bars(coin):
    rows = list(csv.DictReader(open(f"data_15m_3y/{coin}.csv")))
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
    return raw, torch.tensor([rets]), feats

d = json.load(open("backtest_long_search.json"))
vm = StackVM()

def turnover_of(raw, rets, sig, lth):
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=False,
                      funding_override=0.0005, long_th=lth, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0)
    bt.evaluate(sig, raw, rets)
    return round(bt.last_metrics["turnover"], 4)

for coin in ["BTC", "ETH"]:
    bars = load_bars(coin)
    n = len(bars)
    oos = bars[int(n*0.85):]; h = len(oos)//2
    segs = {"h1": oos[:h], "h2": oos[h:], "full": bars}
    mats = {k: build_mats(b) for k, b in segs.items()}
    print(f"{coin} n={n} h1={len(segs['h1'])} h2={len(segs['h2'])}", flush=True)
    if coin == "BTC":
        for row in d["top10"]:
            ft, lth = row["formula"], row["lth"]
            for seg in ["h1", "h2", "full"]:
                sig = vm.execute(ft, mats[seg][2])
                row[seg]["turnover"] = turnover_of(mats[seg][0], mats[seg][1], sig, lth)
        for lth in ["0.85", "0.88"]:
            for seg in ["h1", "h2", "full"]:
                sig = vm.execute(d["baseline"]["formula"], mats[seg][2])
                d["baseline"][seg][lth]["turnover"] = turnover_of(mats[seg][0], mats[seg][1], sig, float(lth))
        print("BTC turnover done", flush=True)
    else:
        d["eth_check"] = []
        for row in d["top10"][:3] + [d["top10"][9]]:
            ft, lth = row["formula"], row["lth"]
            e = {"formula": ft, "lth": lth, "rank_btc": row["rank"]}
            for seg in ["h1", "h2", "full"]:
                sig = vm.execute(ft, mats[seg][2])
                # reuse stats from main script via net rebuild
                from run_long_search import net_series, stats
                ser = net_series(mats[seg][0], mats[seg][1], sig, lth, 6)
                st = stats(ser)
                st["turnover"] = turnover_of(mats[seg][0], mats[seg][1], sig, lth)
                e[seg] = st
            d["eth_check"].append(e)
            print(f"ETH rank{row['rank']}: H1={e['h1']} H2={e['h2']} FULL={e['full']}", flush=True)
        eb = {"name": "grouptest_baseline_long"}
        for seg in ["h1", "h2", "full"]:
            sig = vm.execute(d["baseline"]["formula"], mats[seg][2])
            from run_long_search import net_series, stats
            eb[seg] = {}
            for lth in [0.85, 0.88]:
                ser = net_series(mats[seg][0], mats[seg][1], sig, lth, 6)
                st = stats(ser)
                st["turnover"] = turnover_of(mats[seg][0], mats[seg][1], sig, lth)
                eb[seg][str(lth)] = st
        d["eth_check"].append({"rank_btc": "baseline", **eb})
        print(f"ETH baseline: {eb}", flush=True)

json.dump(d, open("backtest_long_search.json", "w"), indent=1)
print("updated backtest_long_search.json", flush=True)
