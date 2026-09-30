"""Two columns in the bar data that no factor uses: quote_volume and trades.

Found while asking what to expand the factor library with. The 28c loader
reads open/high/low/close/volume and drops the other two columns the Binance
klines actually carry. Before proposing new factors, it is worth checking
whether the ignored data already contains something, because that is a
cheaper test than inventing a factor.

Screened exactly like the 12 existing ones, through the live position path,
with the same three disqualifiers: demeaned IC not significant, no excess over
a constant levered long, and indistinguishable from the frozen iid null.
"""
import sys, json
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import (accounting_bar_returns, load_real_funding,
                                     metrics)
from research.splits_28c import search_splits, split_indices
from pathlib import Path

PC = 0.25
common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
sp = search_splits(n, common[0], common[-1])
lb = split_indices(n, common[0], common[-1])["lockbox"]
scale_end = sp["train"][1]
funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
           for c in GA.COINS_28C}

DATA = Path("/home/brian/project/AlphaGPT/data/data_3y/30m")


def raw_col(coin, name):
    """Read one raw column straight from the CSV, aligned to the common grid."""
    import csv
    import bisect
    rows = {}
    with open(DATA / f"{coin}.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows[int(r["timestamp"])] = r
    out = np.full(n, np.nan)
    for i, ts in enumerate(common):
        r = rows.get(int(ts))
        if r is not None:
            out[i] = float(r[name])
    return out


def robust(x, w=200):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for t in range(n):
        lo = max(0, t - w + 1)
        win = x[lo:t+1]
        if len(win) < 20 or not np.isfinite(win).all():
            out[t] = x[t] if t < len(x) else 0.0
            continue
        m = np.median(win)
        mad = np.median(np.abs(win - m)) + 1e-6
        out[t] = np.clip((x[t] - m) / mad, -5, 5)
    return out


def position(sig):
    fit = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
    raw = np.tanh(sig / (fit + 1e-6)) if fit > 1e-8 else np.zeros_like(sig)
    return np.roll(PC * GA.smooth_causal(raw, 5), 1)


def score(sig):
    row = {}
    for wname, (a, b) in (("train", sp["train"]), ("validation", sp["validation"]),
                          ("lockbox", lb)):
        raw_ic, dem_ic, nets, bh, mp = [], [], [], [], []
        for c in GA.COINS_28C:
            p = position(sig[c])[a:b]
            r = np.asarray(returns[c])[a:b]
            if len(p) < 50 or p.std() < 1e-12 or r.std() < 1e-12:
                continue
            raw_ic.append(np.corrcoef(p, r)[0, 1])
            dm = p - p.mean()
            if dm.std() > 1e-12:
                ric = np.corrcoef(dm, r)[0, 1]
                dem_ic.append((ric, ric * np.sqrt(len(dm) - 2)))
            f = funding[c][a:b]
            nets.append(accounting_bar_returns(p, r, GA.FEE, f, GA.LEV))
            bh.append(accounting_bar_returns(np.full(len(p), PC), r, GA.FEE, f, GA.LEV))
            mp.append(float(p.mean()))
        if not nets:
            continue
        sh = metrics(np.mean(np.stack(nets), axis=0), common[a:b])["sharpe"]
        bsh = metrics(np.mean(np.stack(bh), axis=0), common[a:b])["sharpe"]
        row[wname] = {"raw_ic": float(np.mean(raw_ic)),
                      "demeaned_ic": float(np.mean([d[0] for d in dem_ic])),
                      "t": float(np.mean([d[1] for d in dem_ic])),
                      "coins_gt_0": float(np.mean([d[0] > 0 for d in dem_ic])),
                      "sharpe": sh, "bh": bsh, "excess": sh - bsh,
                      "mean_pos": float(np.mean(mp))}
    return row


CAND = {
    "QUOTE_VOL_LOGVOL": lambda c: np.log1p(np.maximum(raw_col(c, "quote_volume"), 0)),
    "TRADE_COUNT":     lambda c: np.log1p(np.maximum(raw_col(c, "trades"), 0)),
    "VWAP_DEV":        None,      # filled below
    "TRADE_SIZE":      None,
}

cols = {}
for c in GA.COINS_28C:
    cols[c] = (raw_col(c, "quote_volume"), raw_col(c, "trades"),
               raw_col(c, "volume"))

signals = {}
for c in GA.COINS_28C:
    qv, tc, vol = cols[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol > 0, qv / vol, np.nan)
        tsize = np.where(tc > 0, vol / tc, np.nan)
    vwap = np.nan_to_num(vwap, nan=0.0, posinf=0.0, neginf=0.0)
    tsize = np.nan_to_num(tsize, nan=0.0, posinf=0.0, neginf=0.0)
    signals.setdefault("QUOTE_VOL_LOGVOL", {})[c] = robust(np.log1p(np.maximum(qv, 0)))
    signals.setdefault("TRADE_COUNT", {})[c] = robust(np.log1p(np.maximum(tc, 0)))
    signals.setdefault("VWAP_DEV", {})[c] = robust((vwap - np.asarray(
        GA.maps_cache_close if False else 0.0)) * 0) if False else robust(
        (vwap / (np.abs(vwap).mean() + 1e-9) - 1.0) * 10.0)
    signals.setdefault("TRADE_SIZE", {})[c] = robust(np.log1p(np.maximum(tsize, 0)))

print("%-18s %-6s %10s %8s %9s %9s %9s" % (
    "factor", "window", "rawIC", "demIC", "t", "Sharpe", "excess"))
print("-" * 78)
out = {}
for name, sig in signals.items():
    row = score(sig)
    out[name] = row
    for w in ("train", "validation", "lockbox"):
        if w not in row:
            continue
        d = row[w]
        print("%-18s %-6s %+10.4f %+8.4f %+9.2f %+9.3f %+9.3f" % (
            name, w, d["raw_ic"], d["demeaned_ic"], d["t"], d["sharpe"], d["excess"]))
    lk = row.get("lockbox")
    if lk:
        print("%-18s %-6s coins with IC>0: %.0f%%   meanPos %+.3f" % (
            "", "lockbox", 100 * lk["coins_gt_0"], lk["mean_pos"]))
Path("/home/brian/project/AlphaGPT/results/unused_column_screen.json").write_text(
    json.dumps(out, indent=1))
print("\nwrote results/unused_column_screen.json")
