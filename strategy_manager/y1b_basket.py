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
from strategy_manager.config import FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS, LEV, FUND, FEE

BASKET = {
    "ETC": dict(lth=LOCKED_ETC["lth"], sth=LOCKED_ETC["sth"], cd=LOCKED_ETC["cd"],
                sl=LOCKED_ETC["sl"], ts=LOCKED_ETC["ts"], vt=LOCKED_ETC["vt"],
                vw=LOCKED_ETC["vw"], q=LOCKED_ETC["q"]),
    "TRX": dict(lth=LOCKED_TRX["lth"], sth=LOCKED_TRX["sth"], cd=LOCKED_TRX["cd"],
                sl=LOCKED_TRX["sl"], ts=LOCKED_TRX["ts"], vt=LOCKED_TRX["vt"],
                vw=LOCKED_TRX["vw"], q=LOCKED_TRX["q"]),
}
WEIGHTS = {"ETC": 0.5, "TRX": 0.5}

# Top5 basket (2026-09-12, gate PASS): ETC/TRX + ATOM/APT/KAS, equal 20%.
BASKET_5 = {
    "ETC": dict(lth=LOCKED_ETC["lth"], sth=LOCKED_ETC["sth"], cd=LOCKED_ETC["cd"],
                sl=LOCKED_ETC["sl"], ts=LOCKED_ETC["ts"], vt=LOCKED_ETC["vt"],
                vw=LOCKED_ETC["vw"], q=LOCKED_ETC["q"]),
    "TRX": dict(lth=LOCKED_TRX["lth"], sth=LOCKED_TRX["sth"], cd=LOCKED_TRX["cd"],
                sl=LOCKED_TRX["sl"], ts=LOCKED_TRX["ts"], vt=LOCKED_TRX["vt"],
                vw=LOCKED_TRX["vw"], q=LOCKED_TRX["q"]),
    "ATOM": dict(lth=LOCKED_ATOM["lth"], sth=LOCKED_ATOM["sth"], cd=LOCKED_ATOM["cd"],
                 sl=LOCKED_ATOM["sl"], ts=LOCKED_ATOM["ts"], vt=LOCKED_ATOM["vt"],
                 vw=LOCKED_ATOM["vw"], q=LOCKED_ATOM["q"]),
    "APT": dict(lth=LOCKED_APT["lth"], sth=LOCKED_APT["sth"], cd=LOCKED_APT["cd"],
                sl=LOCKED_APT["sl"], ts=LOCKED_APT["ts"], vt=LOCKED_APT["vt"],
                vw=LOCKED_APT["vw"], q=LOCKED_APT["q"]),
    "KAS": dict(lth=LOCKED_KAS["lth"], sth=LOCKED_KAS["sth"], cd=LOCKED_KAS["cd"],
                sl=LOCKED_KAS["sl"], ts=LOCKED_KAS["ts"], vt=LOCKED_KAS["vt"],
                vw=LOCKED_KAS["vw"], q=LOCKED_KAS["q"]),
}
WEIGHTS_5 = {"ETC": 0.2, "TRX": 0.2, "ATOM": 0.2, "APT": 0.2, "KAS": 0.2}

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

def _active_basket():
    import os as _os
    if _os.getenv("Y1B_TOP5", "").strip().lower() in {"1", "true", "yes"}:
        return BASKET_5, WEIGHTS_5
    return BASKET, WEIGHTS

def basket_signals(bars_map: dict | None = None, top5: bool | None = None):
    import os as _os
    use5 = top5 if top5 is not None else (_os.getenv("Y1B_TOP5", "").strip().lower() in {"1", "true", "yes"})
    basket, weights = (BASKET_5, WEIGHTS_5) if use5 else (BASKET, WEIGHTS)
    bars_map = bars_map or {c: load_bars_4h(c) for c in basket}
    n = min(len(b) for b in bars_map.values())
    sigs = {}
    for c, spec in basket.items():
        sigs[c] = leg_position(bars_map[c][:n], spec)
    import datetime as _dt
    # signal_age: hours since the last closed 4h bar (bars are 4h-aggregated).
    _now = _dt.datetime.now(_dt.timezone.utc)
    _age_h = round((_now.hour % 4) + _now.minute / 60.0 + _now.second / 3600.0, 2)
    return {"n": n, "signals": sigs, "weights": dict(weights),
            "formula": list(FORMULA), "lev": LEV, "fee": FEE, "fund": FUND,
            "signal_age_h": _age_h, "asof_utc": _now.isoformat()}

def latest_signals(bars_map: dict | None = None, top5: bool | None = None):
    r = basket_signals(bars_map, top5=top5)
    return {c: (s[-1] if s else 0.0) for c, s in r["signals"].items()}
