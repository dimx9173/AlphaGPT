# X6 decision-frequency contrast (Top5) -- DIAGNOSTIC ONLY.
# P0-3 permutation FAILED => PENDING, no live change.
# 1h replay: hourly vs aligned_4h (risk-only hold off-aligned).
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LEV
from research.run_qsweep import BASKET_SPECS, COINS, FEE, FUND, quantile_mask_long
BPY = 8760.0
H2_LEN = 493 * 4
OUT = pathlib.Path(os.getenv("ITER_X6_OUT", "results/iter_X6_freq.json"))
LOG = pathlib.Path(os.getenv("ITER_X6_LOG", "logs/iter_x6.log"))
SMOKE = os.getenv("ITER_X6_SMOKE") == "1"
ARM = 'hourly'

def log(msg):
    print(msg, flush=True)
    open(LOG, "a").write(msg + chr(10))
def load_15m(coin):
    rows = list(csv.DictReader(open("data/data_15m_3y/%s.csv" % coin)))
    return [(int(r['timestamp']), float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])) for r in rows]

def common_1h(coins):
    raw = {c: load_15m(c) for c in coins}
    start = max(r[0][0] for r in raw.values())
    end = min(r[-1][0] for r in raw.values())
    bars, closes = {}, {}
    for c in coins:
        rows = [r for r in raw[c] if start <= r[0] <= end]
        n1 = len(rows) // 4
        b1 = []
        for i in range(n1):
            blk = rows[i*4:(i+1)*4]
            b1.append((blk[0][1], max(r[2] for r in blk), min(r[3] for r in blk), blk[-1][4], sum(r[5] for r in blk)))
        bars[c] = b1
        closes[c] = [b[3] for b in b1]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
        closes[c] = closes[c][:n]
    return bars, closes

def build_sig_1h(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]), "high": torch.tensor([[b[1] for b in bars]]), "low": torch.tensor([[b[2] for b in bars]]), "close": torch.tensor([[b[3] for b in bars]]), "volume": torch.tensor([[b[4] for b in bars]]), "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def desired_1h(raw, rets_t, sig, spec, q):
    h = dict(spec, cd=int(spec["cd"])*4, ts=int(spec["ts"])*4, vw=int(spec["vw"])*4)
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True, funding_override=FUND, fee_override=FEE, long_th=h["lth"], short_th=h["sth"], cooldown_bars=h["cd"], bars_per_year=BPY, stop_loss=h["sl"], time_stop=h["ts"], vol_target=h["vt"], vol_window=h["vw"])
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = quantile_mask_long(sig, q)
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    scale = bt._vol_scale(rets_t)
    lp, sp = lp * scale, sp * scale
    pos = (lp - sp)[0].tolist()
    return [1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0) for v in pos]

def apply_gate(d, freq):
    n = len(d)
    if freq == "hourly":
        adopted = list(d)
    elif freq == "aligned_4h":
        adopted = []
        cur = 0.0
        for t in range(n):
            if t % 4 == 0:
                cur = d[t]
            adopted.append(cur)
    else:
        raise ValueError(freq)
    return [0.0] + adopted[:-1]

def pnl_from_pos(pos, rets, fee, fund):
    net, turn, flips, entries = [], [], 0, 0
    prev = 0.0
    for t in range(len(pos)):
        pp = pos[t]; r = rets[t]; dp = pp - prev
        turn.append(abs(dp))
        net.append(pp*r*LEV - abs(dp)*fee*LEV - pp*fund*LEV)
        if pp != 0.0 and prev == 0.0:
            entries += 1
        if pp != 0.0 and prev != 0.0 and (pp > 0) != (prev > 0):
            flips += 1
        prev = pp
    return net, turn, flips, entries

def seg_stats_h(net, turn, a, b):
    s = net[a:b]; t = turn[a:b]; n = len(s)
    mean = sum(s)/n if n else 0.0
    var = sum((x-mean)**2 for x in s)/max(n-1,1) if n > 1 else 0.0
    sh = mean/math.sqrt(var)*math.sqrt(BPY) if var > 0 else 0.0
    cum = sum(s); cs, peak, mdd = 0.0, -1e18, 0.0
    for x in s:
        cs += x; peak = max(peak, cs); mdd = max(mdd, peak-cs)
    return {"sharpe": round(sh,3), "ann": round(cum/n*BPY,4) if n else 0.0, "mdd": round(mdd,4), "cum": round(cum,4), "n": n, "turnover": round(sum(t)/n,6) if n else 0.0}

def eval_arm(coins, des, rets, n, fee, fund, freq):
    legs_net, legs_turn, per_coin = [], [], {}
    h2a = n - H2_LEN
    for c in coins:
        _pos = apply_gate(des[c], freq)
        _net, _turn, _flips, _entries = pnl_from_pos(_pos, rets[c], fee, fund)
        per_coin[c] = {"FULL": seg_stats_h(_net, _turn, 0, n), "H2": seg_stats_h(_net, _turn, h2a, n), "flips": _flips, "entries": _entries}
        legs_net.append(_net)
        legs_turn.append(_turn)
    w = 1.0/len(coins)
    net = [sum(legs_net[i][t]*w for i in range(len(coins))) for t in range(n)]
    turn = [sum(legs_turn[i][t]*w for i in range(len(coins))) for t in range(n)]
    return {"FULL": seg_stats_h(net, turn, 0, n), "H2": seg_stats_h(net, turn, n-H2_LEN, n), "flips": sum(v["flips"] for v in per_coin.values()), "entries": sum(v["entries"] for v in per_coin.values()), "per_coin": per_coin}
def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x6 start smoke=%s" % SMOKE + chr(10))
    coins = ["ETC", "TRX"] if SMOKE else list(COINS)
    bars, _ = common_1h(coins)
    n = len(bars["ETC"])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    log("common 1h bars n=%d coins=%s smoke=%s" % (n, coins, SMOKE))
    assert n > 2000, n
    assert n > H2_LEN, n
    mats = {c: build_sig_1h(bars[c]) for c in coins}
    log("signals built")
    des, rets = {}, {}
    for c in coins:
        raw, rt, sg = mats[c]
        spec = BASKET_SPECS[c]
        des[c] = desired_1h(raw, rt, sg, spec, spec.get("q", 0.3))
        rets[c] = rt[0].tolist()
    log("desired positions built")
    arms = {}
    for freq in ("hourly", "aligned_4h"):
        arms[freq] = eval_arm(coins, des, rets, n, FEE, FUND, freq)
        f, h = arms[freq]["FULL"], arms[freq]["H2"]
        log("%s FULL sh=%.3f dd=%.4f cum=%.4f to=%.5f flips=%d entries=%d | H2 sh=%.3f dd=%.4f" % (freq, f["sharpe"], f["mdd"], f["cum"], f["turnover"], arms[freq]["flips"], arms[freq]["entries"], h["sharpe"], h["mdd"]))
    fee2x = {}
    for freq in ("hourly", "aligned_4h"):
        r = eval_arm(coins, des, rets, n, FEE*2, FUND, freq)
        fee2x[freq] = {"FULL_sharpe": r["FULL"]["sharpe"], "H2_sharpe": r["H2"]["sharpe"], "FULL_cum": r["FULL"]["cum"]}
        log("%s fee2x FULL sh=%.3f cum=%.4f | H2 sh=%.3f" % (freq, r["FULL"]["sharpe"], r["FULL"]["cum"], r["H2"]["sharpe"]))
    fh, fa = arms["hourly"]["FULL"], arms["aligned_4h"]["FULL"]
    hh, ha = arms["hourly"]["H2"], arms["aligned_4h"]["H2"]
    to_ratio = (fh['turnover']/fa['turnover'] if fa['turnover'] > 1e-12 else 0.0)
    d_sharpe = round(fh["sharpe"]-fa["sharpe"], 3)
    d_sharpe_h2 = round(hh["sharpe"]-ha["sharpe"], 3)
    rec, reason = ('0', 'hourly sharpe gain >= +0.2') if d_sharpe >= 0.2 else ('1', 'hourly churn not paid (d=%+.3f, ratio=%.2f)' % (d_sharpe, to_ratio))
    log("compare dFULL=%+.3f dH2=%+.3f ratio=%.2f -> ALIGN=%s (%s)" % (d_sharpe, d_sharpe_h2, to_ratio, rec, reason))
    out = {
        "config": {"coins": coins, "weights": {c: round(1.0/len(coins),4) for c in coins}, "grid": "1h", "grid_bars": n, "h2_len": H2_LEN, "fee": FEE, "fund": FUND, "lev": LEV, "params_scaled_x4": ["cd","ts","vw"], "smoke": SMOKE},
        "arms": arms, "fee2x": fee2x,
        "compare": {"d_sharpe_FULL": d_sharpe, "d_sharpe_H2": d_sharpe_h2, "d_cum_FULL": round(fh["cum"]-fa["cum"],4), "d_turnover_FULL": round(fh["turnover"]-fa["turnover"],6), "turnover_ratio_hourly_over_4h": round(to_ratio,4), "d_flips": arms["hourly"]["flips"]-arms["aligned_4h"]["flips"], "d_entries": arms["hourly"]["entries"]-arms["aligned_4h"]["entries"]},
        "recommendation": {"Y1B_DECISION_ALIGN": rec, "reason": reason, "status": "shelved (executor default stays unset)"},
        "decision": "PENDING",
        "conclusion": "X6 1h replay hourly vs 4h-aligned dFULL=%+.3f ratio=%.2f ALIGN=%s (%s). PENDING; no live change." % (d_sharpe, to_ratio, rec, reason),
    }
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s" % OUT)


if __name__ == "__main__":
    main()
