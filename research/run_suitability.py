"""P2-1 coin whitelist suitability gate (E12/E13).

Independent diagnostic ONLY. P0-3 permutation FAILed
(per-coin p=0.060/0.159 >= 0.05, results/permutation.json), so per the PRP
global exit rule this step's conclusion is "PENDING (待定)" and MUST NOT be
used as demo-listing evidence.

What it does (offline, read-only):
  1. Per coin (4h bars): Hurst (R/S), lag1-4 return autocorrelation,
     trend SNR (efficiency ratio + drift SNR), Amihud illiquidity,
     funding-vol PROXY (no funding series exists offline -> rolling
     vol-of-vol of returns, labelled as proxy, threshold TBD),
     4h gap rate (open vs prev close).
  2. Strategy-regime fitness: 12-param grid mirrored EXACTLY from
     research/run_percoin.py (E10 FORMULA, aster perp 2x, fund 0.0005,
     base fee, no quantile filter -- same engine that produced the
     valid/fail group labels), in-sample bars [0:cut), cut=6580 from
     research/oos_freeze.json (coins shorter than cut use full length).
  3. Gate threshold from valid vs fail separation + PEPE-best-params
     counter-proof transferred back onto ETC/TRX.

Writes results/suitability.json (+ logs/suitability.log).
No orders, no broker imports, no network.
"""
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.config import FORMULA, FUND

VALID = ["ETC", "TRX", "ATOM", "APT", "KAS"]
INVALID = ["PEPE", "ICP"]
COINS = VALID + INVALID

# Exact mirror of research/run_percoin.py GRID (lth, sth, cd, sl).
GRID = [(0.85, 0.15, 6, None), (0.85, 0.15, 6, 0.05),
        (0.88, 0.12, 6, None), (0.88, 0.12, 3, 0.03),
        (0.90, 0.10, 6, None), (0.90, 0.10, 3, 0.03),
        (0.92, 0.08, 6, None), (0.92, 0.08, 12, None),
        (0.85, 0.15, 12, 0.03), (0.90, 0.10, 12, 0.03),
        (0.88, 0.12, 12, None), (0.92, 0.08, 3, 0.03)]

GATE_MIN_SHARPE = 1.0

OUT = pathlib.Path("results/suitability.json")
LOG = pathlib.Path("logs/suitability.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load_4h(coin, cut):
    rows = list(csv.DictReader(open("data/data_15m_3y/%s.csv" % coin)))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i + 16]
        if len(blk) < 16:
            break
        bars.append({
            "open": float(blk[0]["open"]),
            "high": max(float(x["high"]) for x in blk),
            "low": min(float(x["low"]) for x in blk),
            "close": float(blk[-1]["close"]),
            "volume": sum(float(x["volume"]) for x in blk),
            "quote_volume": sum(float(x["quote_volume"]) for x in blk),
        })
    full_n = len(bars)
    return bars[:cut], full_n


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def hurst_rs(closes):
    import math as _m
    lp = [_m.log(c) for c in closes]
    n_all = len(lp)
    ks, k = [], 16
    while k <= n_all // 8:
        ks.append(k)
        k *= 2
    xs, ys = [], []
    for n in ks:
        m = n_all // n
        vals = []
        for i in range(m):
            seg = lp[i * n:(i + 1) * n]
            mu = _mean(seg)
            dev, acc = [], 0.0
            for v in seg:
                acc += v - mu
                dev.append(acc)
            rng = max(dev) - min(dev)
            var = _mean([(v - mu) ** 2 for v in seg])
            sd = _m.sqrt(var)
            if sd > 0:
                vals.append(rng / sd)
        if vals:
            xs.append(_m.log(n))
            ys.append(_m.log(_mean(vals)))
    if len(xs) < 2:
        return 0.5
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.5


def acf(xs, lag):
    n = len(xs)
    m = _mean(xs)
    d = [x - m for x in xs]
    den = sum(v * v for v in d)
    if den <= 0:
        return 0.0
    return sum(d[i] * d[i + lag] for i in range(n - lag)) / den


def describe(bars):
    closes = [b["close"] for b in bars]
    rets = [(closes[i + 1] - closes[i]) / closes[i]
            for i in range(len(closes) - 1)]
    n = len(rets)
    mu = _mean(rets)
    var = _mean([(r - mu) ** 2 for r in rets])
    sd = math.sqrt(var)
    h = hurst_rs(closes)
    ac = {lag: acf(rets, lag) for lag in (1, 2, 3, 4)}
    er = (abs(closes[-1] - closes[0])
          / sum(abs(closes[i] - closes[i - 1])
                for i in range(1, len(closes)))) if len(closes) > 1 else 0.0
    drift_snr = mu / sd * math.sqrt(2190.0) if sd > 0 else 0.0
    qv = [b["quote_volume"] for b in bars][1:]
    amihud = _mean([abs(r) / max(q, 1e-9) for r, q in zip(rets, qv)])
    # Funding-vol PROXY (no funding series offline): vol-of-vol of 4h
    # returns over rolling 30-bar windows. Labelled proxy, threshold TBD.
    w = 30
    rolls = [rets[i:i + w] for i in range(0, n - w + 1, w)]
    wvol = []
    for seg in rolls:
        m2 = _mean(seg)
        wvol.append(math.sqrt(_mean([(x - m2) ** 2 for x in seg])))
    funding_vol_proxy = (math.sqrt(_mean([(v - _mean(wvol)) ** 2
                                          for v in wvol])) / _mean(wvol)
                         if wvol and _mean(wvol) > 0 else 0.0)
    gaps = [abs(bars[i]["open"] - bars[i - 1]["close"]) / bars[i - 1]["close"]
            for i in range(1, len(bars))]
    gap10 = sum(1 for g in gaps if g > 0.001) / len(gaps) if gaps else 0.0
    gap50 = sum(1 for g in gaps if g > 0.005) / len(gaps) if gaps else 0.0
    return {
        "hurst": round(h, 4),
        "acf": {str(k): round(v, 4) for k, v in ac.items()},
        "trend_er": round(er, 6),
        "drift_snr": round(drift_snr, 4),
        "amihud": amihud,
        "funding_vol_proxy": round(funding_vol_proxy, 4),
        "funding_vol_note": "return-vol-of-vol proxy; no funding series offline",
        "gap_rate_10bp": round(gap10, 6),
        "gap_rate_50bp": round(gap50, 6),
    }


def grid_fitness(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = ([(bars[i + 1]["close"] - bars[i]["close"]) / bars[i]["close"]
             for i in range(n - 1)] + [0.0])
    target = torch.tensor([rets])
    rows = []
    for (lth, sth, cd, sl) in GRID:
        bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                          funding_override=FUND, long_th=lth, short_th=sth,
                          cooldown_bars=cd, bars_per_year=2190.0,
                          stop_loss=sl)
        bt.evaluate(sig, raw, target)
        m = bt.last_metrics
        rows.append({"lth": lth, "sth": sth, "cd": cd, "sl": sl,
                     "sharpe": round(m["sharpe"], 3)})
    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    return rows


def eval_params(bars, params):
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = ([(bars[i + 1]["close"] - bars[i]["close"]) / bars[i]["close"]
             for i in range(n - 1)] + [0.0])
    target = torch.tensor([rets])
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=FUND, long_th=params["lth"],
                      short_th=params["sth"], cooldown_bars=params["cd"],
                      bars_per_year=2190.0, stop_loss=params["sl"])
    bt.evaluate(sig, raw, target)
    return round(bt.last_metrics["sharpe"], 3)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("suitability start\n")
    freeze = json.load(open("research/oos_freeze.json"))
    cut = int(freeze["cut"]["cut_index"])
    assert cut == 6580, "freeze cut moved: %s" % cut

    data = {}
    for coin in COINS:
        bars, full_n = load_4h(coin, cut)
        data[coin] = bars
        log("%s full_4h=%d in_sample=%d" % (coin, full_n, len(bars)))

    coins_out = {}
    for coin in COINS:
        bars = data[coin]
        desc = describe(bars)
        rows = grid_fitness(bars)
        best, worst = rows[0], rows[-1]
        npos = sum(1 for r in rows if r["sharpe"] > 0)
        gate_pass = bool(best["sharpe"] >= GATE_MIN_SHARPE)
        coins_out[coin] = {
            "n_full": len(bars) if len(bars) < cut else cut,
            "n_is": len(bars),
            **desc,
            "grid_best": best,
            "grid_worst": {"sharpe": worst["sharpe"]},
            "grid_npos": npos,
            "grid_n": len(rows),
            "gate_pass": gate_pass,
        }
        log("%s H=%.3f acf1=%+.3f ER=%.4f driftSNR=%+.3f amihud=%.2e "
            "fvol=%.3f gap10=%.5f grid_best=%.3f npos=%d/12 %s"
            % (coin, desc["hurst"], desc["acf"]["1"], desc["trend_er"],
               desc["drift_snr"], desc["amihud"],
               desc["funding_vol_proxy"], desc["gap_rate_10bp"],
               best["sharpe"], npos,
               "PASS" if gate_pass else "FAIL"))

    valid_best = min(coins_out[c]["grid_best"]["sharpe"] for c in VALID)
    invalid_best = max(coins_out[c]["grid_best"]["sharpe"] for c in INVALID)
    no_overlap = bool(valid_best >= GATE_MIN_SHARPE > invalid_best)
    log("gate>=%.1f valid_min=%.3f invalid_max=%.3f no_overlap=%s"
        % (GATE_MIN_SHARPE, valid_best, invalid_best, no_overlap))

    # Counter-proof: PEPE in-sample-best params transferred onto ETC/TRX
    # (and full valid group). Regime-specific iff params stay healthy
    # elsewhere while failing on PEPE itself.
    pepe_best = {k: coins_out["PEPE"]["grid_best"][k]
                 for k in ("lth", "sth", "cd", "sl")}
    transfer = {}
    for coin in COINS:
        transfer[coin] = eval_params(data[coin], pepe_best)
        log("pepe-best %s on %s sharpe=%.3f"
            % (pepe_best, coin, transfer[coin]))
    counter_ok = bool(transfer["ETC"] > 0.5 and transfer["TRX"] > 0.5
                      and coins_out["PEPE"]["grid_best"]["sharpe"] < 0)
    log("counter-proof pepe-best-on-ETC=%.3f on-TRX=%.3f "
        "pepe-own-best=%.3f regime_specific=%s"
        % (transfer["ETC"], transfer["TRX"],
           coins_out["PEPE"]["grid_best"]["sharpe"], counter_ok))

    result = {
        "config": {
            "formula": FORMULA,
            "engine": "mirror research/run_percoin.py GRID (12 combos), "
                      "aster perp 2x, fund 0.0005, base fee, no quantile",
            "grid": [{"lth": l, "sth": s, "cd": c, "sl": v}
                     for (l, s, c, v) in GRID],
            "in_sample_rule": "bars[0:cut), cut=6580 (research/oos_freeze.json); "
                              "coins shorter than cut use full length",
            "groups": {"valid": VALID, "invalid": INVALID},
            "gate_rule": "grid_best_sharpe >= %.1f" % GATE_MIN_SHARPE,
            "p0_3_status": "FAIL (per-coin permutation p>=0.05); "
                           "this report is diagnostic-only",
            "role": "independent diagnostic; conclusion PENDING (待定); "
                    "NOT demo-listing evidence",
            "offline": True,
        },
        "coins": coins_out,
        "gate": {
            "threshold": GATE_MIN_SHARPE,
            "valid_min_best": valid_best,
            "invalid_max_best": invalid_best,
            "no_overlap": no_overlap,
            "pass": [c for c in COINS if coins_out[c]["gate_pass"]],
            "fail": [c for c in COINS if not coins_out[c]["gate_pass"]],
        },
        "transfer_pepe_best": {
            "params": pepe_best,
            "on": transfer,
        },
        "counter_proof": {
            "claim": "PEPE failure is regime-specific, not globally-bad params",
            "pepe_best_on_etc": transfer["ETC"],
            "pepe_best_on_trx": transfer["TRX"],
            "pepe_own_best": coins_out["PEPE"]["grid_best"]["sharpe"],
            "regime_specific": counter_ok,
        },
        "conclusion": "待定 (PENDING): P0-3 permutation FAIL, independent "
                      "diagnostic only, not demo-listing evidence",
    }
    OUT.write_text(json.dumps(result, indent=1))
    log("saved results/suitability.json gate_pass=%s counter=%s conclusion=待定"
        % ([c for c in COINS if coins_out[c]["gate_pass"]], counter_ok))


if __name__ == "__main__":
    main()
