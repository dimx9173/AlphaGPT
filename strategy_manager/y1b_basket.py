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

# === 15m mode (2026-09-14): Y1B_BAR=15m reads data/data_1y/15m directly,
# no x16 aggregation. cd/ts/vw scale x16 (4h-bar units -> 15m-bar units),
# bars_per_year 2190 -> 35040. Default 4h (legacy, unchanged).
BAR_INTERVALS = ("4h", "15m")
BPY_MAP = {"4h": 2190.0, "15m": 35040.0}
AGG_MAP = {"4h": 16, "15m": 1}
DATA_DIR_MAP = {"4h": "data/data_15m_3y", "15m": "data/data_1y/15m"}


def bar_interval() -> str:
    import os as _os
    v = (_os.getenv("Y1B_BAR", "4h") or "4h").strip().lower()
    return v if v in BAR_INTERVALS else "4h"


def bars_per_year() -> float:
    return BPY_MAP[bar_interval()]


def _scale_spec(spec: dict) -> dict:
    """Scale bar-count params (cd/ts/vw) 4h->15m when in 15m mode."""
    if bar_interval() != "15m":
        return spec
    out = dict(spec)
    for k in ("cd", "ts", "vw"):
        try:
            if out.get(k) is not None:
                out[k] = int(out[k]) * 16
        except (TypeError, ValueError):
            pass
    return out


def load_bars_4h(coin: str, path: str | None = None):
    bi = bar_interval()
    if path is None:
        p = f"{DATA_DIR_MAP[bi]}/{coin}.csv"
    else:
        p = path
    rows = list(csv.DictReader(open(p)))
    agg = AGG_MAP[bi]
    bars = []
    for i in range(0, len(rows), agg):
        blk = rows[i:i+agg]
        if len(blk) < agg:
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
    spec = _scale_spec(spec)
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
                      cooldown_bars=spec["cd"], bars_per_year=bars_per_year(),
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
    try:
        _sg_last = float(sg[0][-1])
    except Exception:
        _sg_last = 0.5
    return out, _sg_last

def _active_basket():
    import os as _os
    if _os.getenv("Y1B_TOP5", "").strip().lower() in {"1", "true", "yes"}:
        return BASKET_5, WEIGHTS_5
    return BASKET, WEIGHTS

# === P1-1 inverse-vol weighting (E06/E07, default OFF/equal) ===
# Y1B_WEIGHT_MODE=equal|invvol|invvol_cap (default equal).
# invvol: trailing 60x4h realized-vol reciprocal, per-coin clip 10~35%.
# invvol_cap: invvol weights + portfolio vol target (Y1B_VOL_TARGET, default
# 0.35 ann.) rescaling total leverage. P0-3 permutation FAIL => verdict
# PENDING: research contrast only, live default stays equal.
WEIGHT_MODES = ("equal", "invvol", "invvol_cap")
Y1B_VOL_WINDOW = 60
Y1B_W_MIN, Y1B_W_MAX = 0.10, 0.35
Y1B_VOL_TARGET_DEFAULT = 0.35
Y1B_LEV_SCALE_MIN, Y1B_LEV_SCALE_MAX = 0.25, 2.0


def weight_mode():
    import os as _os
    m = (_os.getenv("Y1B_WEIGHT_MODE", "equal") or "equal").strip().lower()
    return m if m in WEIGHT_MODES else "equal"


def vol_target():
    import os as _os
    try:
        return float(_os.getenv("Y1B_VOL_TARGET", str(Y1B_VOL_TARGET_DEFAULT)))
    except (TypeError, ValueError):
        return Y1B_VOL_TARGET_DEFAULT


def realized_vol(closes, window=Y1B_VOL_WINDOW):
    """Trailing annualized realized vol of simple returns (4h grid)."""
    import math as _m
    r = [(closes[i + 1] - closes[i]) / closes[i] for i in range(len(closes) - 1)
         if closes[i]]
    r = r[-window:] if len(r) >= window else r
    if len(r) < 2:
        return 0.0
    m = sum(r) / len(r)
    var = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    return _m.sqrt(max(var, 0.0)) * _m.sqrt(bars_per_year())


def invvol_weights(vols: dict, lo=Y1B_W_MIN, hi=Y1B_W_MAX):
    """Inverse-vol weights with exact box-simplex projection (cap clip).

    Target raw = inv-vol normalized; project onto {sum=1, lo<=w<=hi} via
    water-filling: iteratively pin weights hitting bounds, rescale the rest.
    Falls back to uniform if infeasible (e.g. 2-coin basket vs 35% cap).
    """
    coins = list(vols)
    n = len(coins)
    if n == 0:
        return {}
    if not (n * lo <= 1.0 <= n * hi):
        return {c: 1.0 / n for c in coins}
    floored = {c: max(float(vols[c]), 1e-9) for c in coins}
    if all(v <= 1e-9 for v in vols.values()):
        return {c: 1.0 / n for c in coins}
    inv = {c: 1.0 / floored[c] for c in coins}
    tot = sum(inv.values())
    raw = {c: inv[c] / tot for c in coins}
    pinned = {}
    free = list(coins)
    remain = 1.0
    for _ in range(n + 1):
        if not free:
            break
        scale = remain / sum(raw[c] for c in free)
        trial = {c: raw[c] * scale for c in free}
        over = [c for c in free if trial[c] > hi]
        under = [c for c in free if trial[c] < lo]
        if not over and not under:
            for c in free:
                pinned[c] = trial[c]
            free = []
            break
        # pin the most-violated bound first for determinism
        if over and (not under or max(trial[c] - hi for c in over) >= max(lo - trial[c] for c in under)):
            c = max(over, key=lambda x: trial[x])
            pinned[c] = hi
            remain -= hi
            free.remove(c)
        else:
            c = min(under, key=lambda x: trial[x])
            pinned[c] = lo
            remain -= lo
            free.remove(c)
    else:
        return {c: 1.0 / n for c in coins}
    if free:
        for c in free:
            pinned[c] = raw[c] * (remain / sum(raw[x] for x in free))
    s = sum(pinned.values())
    return {c: pinned[c] / s for c in coins}


def weights_for_bars(bars_map: dict, mode: str | None = None,
                     window=Y1B_VOL_WINDOW, target: float | None = None):
    """Static live weights from trailing-window vols. Returns (weights, meta).

    Default equal path returns the locked static weights untouched.
    """
    import math as _m
    coins = list(bars_map)
    n = len(coins)
    mode = (mode or weight_mode()).strip().lower()
    if mode not in WEIGHT_MODES:
        mode = "equal"
    if mode == "equal" or n == 0:
        return {c: 1.0 / n for c in coins}, {"mode": "equal", "lev_scale": 1.0}
    closes = {c: [b[3] for b in bars_map[c][-window - 1:]] for c in coins}
    vols = {c: realized_vol(closes[c], window) for c in coins}
    w = invvol_weights(vols)
    meta = {"mode": mode, "vols": vols, "lev_scale": 1.0,
            "vol_target": target if target is not None else vol_target()}
    if mode == "invvol_cap":
        rets = {}
        for c in coins:
            cc = closes[c]
            rets[c] = [(cc[i + 1] - cc[i]) / cc[i] for i in range(len(cc) - 1)
                       if cc[i]]
        m = min(len(rets[c]) for c in coins)
        if m >= 20:
            mat = [[rets[c][-m + k] for k in range(m)] for c in coins]
            means = [sum(row) / m for row in mat]
            pv = 0.0
            for i in range(n):
                for j in range(n):
                    cov = sum((mat[i][k] - means[i]) * (mat[j][k] - means[j])
                              for k in range(m)) / (m - 1)
                    pv += w[coins[i]] * w[coins[j]] * cov
            port_vol = _m.sqrt(max(pv, 0.0)) * _m.sqrt(bars_per_year())
            tgt = meta["vol_target"]
            scale = tgt / port_vol if port_vol > 1e-9 else 1.0
            scale = min(max(scale, Y1B_LEV_SCALE_MIN), Y1B_LEV_SCALE_MAX)
            meta["lev_scale"] = round(scale, 4)
            meta["port_vol"] = round(port_vol, 4)
        else:
            meta["lev_scale"] = 1.0
    return w, meta

def basket_signals(bars_map: dict | None = None, top5: bool | None = None):
    import os as _os
    use5 = top5 if top5 is not None else (_os.getenv("Y1B_TOP5", "").strip().lower() in {"1", "true", "yes"})
    basket, weights = (BASKET_5, WEIGHTS_5) if use5 else (BASKET, WEIGHTS)
    bars_map = bars_map or {c: load_bars_4h(c) for c in basket}
    n = min(len(b) for b in bars_map.values())
    sigs = {}
    sg_last = {}
    for c, spec in basket.items():
        _pos, _sg = leg_position(bars_map[c][:n], spec)
        sigs[c] = _pos
        sg_last[c] = _sg
    import datetime as _dt
    # signal_age: hours since the last closed 4h bar (bars are 4h-aggregated).
    _now = _dt.datetime.now(_dt.timezone.utc)
    _age_h = round((_now.hour % 4) + _now.minute / 60.0 + _now.second / 3600.0, 2)
    _mode = weight_mode()
    _wout, _wmeta = dict(weights), {"mode": "equal", "lev_scale": 1.0}
    if _mode != "equal":
        try:
            _wout, _wmeta = weights_for_bars(
                {c: bars_map[c][:n] for c in basket}, mode=_mode)
        except Exception:
            _wout, _wmeta = dict(weights), {"mode": _mode, "fallback": "equal"}
    return {"n": n, "signals": sigs, "sg_last": sg_last, "weights": _wout,
            "weight_mode": _mode, "weight_meta": _wmeta,
            "formula": list(FORMULA), "lev": LEV, "fee": FEE, "fund": FUND,
            "signal_age_h": _age_h, "asof_utc": _now.isoformat()}

def latest_signals(bars_map: dict | None = None, top5: bool | None = None):
    r = basket_signals(bars_map, top5=top5)
    return {c: (s[-1] if s else 0.0) for c, s in r["signals"].items()}
# --- P1-2 drawdown three-layer brake (E10/E11). Research params; ALL default OFF.
# P0-3 = FAIL (permutation p>=0.05) => P1-2 conclusion PENDING; live chain
# unchanged unless env flags are explicitly turned ON. E10 FORMULA untouched.
# NOTE: Y1B_VOL_TARGET is shared with P1-1 invvol_cap (default 0.35 there);
# P1-2 brake reads its own Y1B_TS_VOL_TARGET (default 0.7) so the two steps
# never fight over one knob.
DD_BRAKE_DEFAULTS = {
    "vol_target": 0.7,   # B1: per-coin trailing-60-bar ann vol target
    "vol_min": 0.25,     # B1: per-coin scale floor
    "port_cap": 1.0,     # B2: portfolio trailing-60-bar ann vol cap
    "ts_n": 24,          # B3a: holding >= N bars with no profit -> halve
    "trail_dd": 0.2,     # B3b: portfolio trailing peak-dd > trigger -> halve
    "half": 0.5,
}
DD_BRAKE_WINDOW = 60
DD_BRAKE_BPY = None  # resolved via bars_per_year() at call time


def _brake_env_on(name: str) -> bool:
    import os as _os
    return (_os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes"}


def _brake_env_float(name: str, default: float) -> float:
    import os as _os
    try:
        return float(_os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def brake_b1_on() -> bool:
    """B1 single-coin slow vol targeting (60x4h). Env Y1B_VOL_TS, default OFF."""
    return _brake_env_on("Y1B_VOL_TS")


def brake_b2_on() -> bool:
    """B2 portfolio vol cap (proportional downscale). Env Y1B_PORT_CAP, default OFF."""
    return _brake_env_on("Y1B_PORT_CAP")


def brake_b3_on() -> bool:
    """B3 time-stop / portfolio trailing. Env Y1B_TIMESTOP, default OFF."""
    return _brake_env_on("Y1B_TIMESTOP")


def brakes_all_off() -> bool:
    return not (brake_b1_on() or brake_b2_on() or brake_b3_on())


def trailing_vol(rets: list, window: int = DD_BRAKE_WINDOW,
                 bpy: float | None = None) -> list:
    """Trailing sample-stdev annualized (causal window ending at t)."""
    bpy = bpy or bars_per_year()
    import statistics as _st
    out = []
    for t in range(len(rets)):
        w = [float(x) for x in rets[max(0, t - window + 1):t + 1]]
        if len(w) < 2:
            out.append(0.0)
            continue
        sd = _st.stdev(w)
        out.append(sd * (bpy ** 0.5) if sd > 0 else 0.0)
    return out


def vol_target_scale(vol: float, target: float = 0.7, smin: float = 0.25) -> float:
    """B1 scale: min(1, target/vol), floored at smin. vol<=0 -> 1.0 (no info)."""
    try:
        v = float(vol)
    except (TypeError, ValueError):
        return 1.0
    if v <= 1e-12:
        return 1.0
    return max(float(smin), min(1.0, float(target) / v))


def port_cap_scale(port_vol: float, cap: float = 1.0) -> float:
    """B2 scale: min(1, cap/port_vol). port_vol<=0 -> 1.0."""
    try:
        v = float(port_vol)
    except (TypeError, ValueError):
        return 1.0
    if v <= 1e-12:
        return 1.0
    return min(1.0, float(cap) / v)


def timestop_scales(pos: list, rets: list, n: int = 24,
                    factor: float = 0.5) -> list:
    """B3a: per-bar scale; holding the same nonzero side >= n bars with
    cumulative excursion <= 0 -> factor (default half), else 1.0."""
    out = []
    cur = 0.0
    hold = 0
    exc = 0.0
    for t in range(len(pos)):
        v = float(pos[t]) if t < len(pos) else 0.0
        w = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        r = float(rets[t]) if t < len(rets) else 0.0
        if w != cur:
            cur, hold, exc = w, 0, 0.0
        if cur != 0.0:
            hold += 1
            exc += cur * r
            out.append(float(factor) if (hold >= int(n) and exc <= 0) else 1.0)
        else:
            out.append(1.0)
    return out


def trailing_dd_scales(port_net: list, trigger: float = 0.2,
                       factor: float = 0.5) -> list:
    """B3b: per-bar scale; trailing peak-minus-cum drawdown > trigger -> factor."""
    out = []
    cs = 0.0
    peak = None
    for x in port_net:
        cs += float(x)
        peak = cs if peak is None else max(peak, cs)
        out.append(float(factor) if (peak - cs) > float(trigger) else 1.0)
    return out


def brake_open_scale(coin: str) -> float:
    """P1-2 live open-size scale (env-gated, default 1.0 = OFF).

    B1 (Y1B_VOL_TS): trailing-60-bar coin vol vs Y1B_TS_VOL_TARGET (def 0.7),
      floor Y1B_VOL_MIN (def 0.25). B2 (Y1B_PORT_CAP): Y1B_PORT_VOL reading
      vs Y1B_PORT_VOL_CAP (def 1.0) — offline research feeds port vol via env.
      B3 (Y1B_TIMESTOP): time-stop on the coin's own signal history +
      Y1B_PORT_DD reading vs Y1B_TRAIL_DD (def 0.2) halves. Any flag OFF skips
      that layer. All OFF -> 1.0 (formula/positions untouched).
    """
    import os as _os
    if brakes_all_off():
        return 1.0
    sc = 1.0
    if brake_b1_on():
        try:
            b = load_bars_4h(coin)[-240:]
            cl = [x[3] for x in b]
            r = [(cl[i + 1] - cl[i]) / cl[i] if cl[i] else 0.0
                 for i in range(len(cl) - 1)] + [0.0]
            v = trailing_vol(r)[-1] if r else 0.0
            sc *= vol_target_scale(
                v, _brake_env_float("Y1B_TS_VOL_TARGET", DD_BRAKE_DEFAULTS["vol_target"]),
                _brake_env_float("Y1B_VOL_MIN", DD_BRAKE_DEFAULTS["vol_min"]))
        except Exception:
            pass
    if brake_b2_on():
        try:
            pv = float(_os.getenv("Y1B_PORT_VOL", "") or 0.0)
            if pv > 0:
                sc *= port_cap_scale(
                    pv, _brake_env_float("Y1B_PORT_VOL_CAP", DD_BRAKE_DEFAULTS["port_cap"]))
        except (TypeError, ValueError):
            pass
    if brake_b3_on():
        try:
            basket = BASKET_5 if coin in BASKET_5 else BASKET
            spec = basket.get(coin)
            if spec is not None:
                b = load_bars_4h(coin)[-240:]
                sig, _ = leg_position(b, spec)
                cl = [x[3] for x in b]
                r = [(cl[i + 1] - cl[i]) / cl[i] if cl[i] else 0.0
                     for i in range(len(cl) - 1)] + [0.0]
                tsn = int(_brake_env_float("Y1B_TS_N", float(DD_BRAKE_DEFAULTS["ts_n"])))
                sc *= float(timestop_scales(sig, r, tsn)[-1]) if sig else 1.0
            pdd = float(_os.getenv("Y1B_PORT_DD", "") or 0.0)
            trg = _brake_env_float("Y1B_TRAIL_DD", DD_BRAKE_DEFAULTS["trail_dd"])
            if pdd > trg:
                sc *= DD_BRAKE_DEFAULTS["half"]
        except Exception:
            pass
    return max(0.0, min(1.0, float(sc)))
