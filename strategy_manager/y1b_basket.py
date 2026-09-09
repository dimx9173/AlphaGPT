"""Y1b basket live-signal module (E10 lock).

Offline-capable: builds Y1b ETC+TRX signals from 15m CSV bars aggregated x16
to 4h, mirroring research/run_paper2.py + run_aa.py leg_series engine:
quantile_mask_long q0.3 + cooldown + stops(time_stop24) + vol_scale(vtNone=1.0)
+ roll1. No orders placed here; runner consumes signals.
"""
from __future__ import annotations
import csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ETC, LOCKED_TRX, LEV, FUND, FEE

BASKET = {
    "ETC": dict(lth=LOCKED_ETC["lth"], sth=LOCKED_ETC["sth"], cd=LOCKED_ETC["cd"],
                sl=LOCKED_ETC["sl"], ts=LOCKED_ETC["ts"], vt=LOCKED_ETC["vt"],
                vw=LOCKED_ETC["vw"], q=LOCKED_ETC["q"]),
    "TRX": dict(lth=LOCKED_TRX["lth"], sth=LOCKED_TRX["sth"], cd=LOCKED_TRX["cd"],
                sl=LOCKED_TRX["sl"], ts=LOCKED_TRX["ts"], vt=LOCKED_TRX["vt"],
                vw=LOCKED_TRX["vw"], q=LOCKED_TRX["q"]),
}
WEIGHTS = {"ETC": 0.5, "TRX": 0.5}

def load_bars_4h(coin: str, path: str | None = None):
    p = path or f"data/data_15m_3y/{coin}.csv"
    rows = list(csv.DictReader(open(p)))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16:
            break
        bars.append((float(blk[0]["open"]), max(float(x["high"]) for x in blk),
                     min(float(x["low"]) for x in blk), float(blk[-1]["close"]),
                     sum(float(x["volume"]) for x in blk)))
    return bars

def _quantile_mask_long(sig, q):
    if q is None:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_position(bars, spec: dict):
    n = len(bars)
    raw = {"open": torch.tensor([[x[0] for x in bars]]),
           "high": torch.tensor([[x[1] for x in bars]]),
           "low": torch.tensor([[x[2] for x in bars]]),
           "close": torch.tensor([[x[3] for x in bars]]),
           "volume": torch.tensor([[x[4] for x in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] if i < n-1 else 0.0 for i in range(n)]
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=2190.0,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mask = _quantile_mask_long(sig, spec["q"])
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, torch.tensor([rets]))
    scale = bt._vol_scale(torch.tensor([rets]))
    lp = lp * scale
    sp = sp * scale
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    pos = (lp - sp)[0].tolist()
    out = []
    for v in pos:
        out.append(1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0))
    return out

def basket_signals(bars_map: dict | None = None):
    bars_map = bars_map or {c: load_bars_4h(c) for c in BASKET}
    n = min(len(b) for b in bars_map.values())
    sigs = {}
    for c, spec in BASKET.items():
        sigs[c] = leg_position(bars_map[c][:n], spec)
    return {"n": n, "signals": sigs, "weights": dict(WEIGHTS),
            "formula": list(FORMULA), "lev": LEV, "fee": FEE, "fund": FUND}

def latest_signals(bars_map: dict | None = None):
    r = basket_signals(bars_map)
    return {c: (s[-1] if s else 0.0) for c, s in r["signals"].items()}
