#!/usr/bin/env python3
"""Cross-validate legacy vs equity-compound-v2 accounting on identical returns.

Legacy  : bar Sharpe (annualised by sqrt(17520)), MDD on 1+cumsum(returns) path.
New     : daily-compounded Sharpe (UTC, sqrt(365)), MDD on compounded equity.

Run: .venv2/bin/python tools/crosscheck_accounting.py
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.accounting_28c import metrics as new_metrics, ACCOUNTING_VERSION

BPY = 17520.0
FAIL_TOL = 1e-9


def legacy_metrics(returns, equity_additive):
    n = len(returns)
    mean = float(np.mean(returns)) if n else 0.0
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe = mean / std * math.sqrt(BPY) if std > 1e-12 else 0.0
    peak = -float("inf"); mdd = 0.0
    for v in equity_additive:
        peak = max(peak, float(v))
        mdd = max(mdd, (peak - float(v)) / peak if peak > 0 else 0.0)
    return {"sharpe": sharpe, "mdd": mdd,
            "final_x": float(equity_additive[-1]) if len(equity_additive) else 1.0}


def both(returns, timestamps):
    legacy = legacy_metrics(returns, 1.0 + np.cumsum(returns))
    new = new_metrics(returns, timestamps)
    return legacy, new


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    return bool(cond)


def main():
    ts0 = 1726228800000  # 2024-09-10 00:00 UTC, same anchor as the 28c contract
    BAR = 1800000
    rng = np.random.default_rng(20260925)
    failures = []

    print("== 1. same data, different conventions ==")
    for label, n, drift in [("small-n", 300, 0.0), ("multi-month", 9000, 0.0),
                            ("drifting", 9000, 8e-5), ("heavy-tail", 6000, 0.0)]:
        r = rng.normal(drift, 0.004, n)
        if label == "heavy-tail":
            r += rng.standard_t(3, n) * 0.002
        ts = ts0 + np.arange(n) * BAR
        lg, nw = both(r, ts)
        d = nw["sharpe"] - lg["sharpe"]
        print(f"  {label:11} n={n:5} legacy_sharpe={lg['sharpe']:8.3f} new_sharpe={nw['sharpe']:8.3f} "
              f"delta={d:8.3f} | legacy_mdd={lg['mdd']:.4f} new_mdd={nw['mdd']:.4f}")
        if n > 200 and abs(d) < 1e-6:
            failures.append(f"{label}: Sharpe identical, convention likely not applied")

    print("\n== 2. invariants that MUST hold under v2 ==")
    n = 5000
    ts = ts0 + np.arange(n) * BAR

    r = rng.normal(0, 0.004, n)
    # 5000 30m bars span ~105 UTC days. Grouping must follow calendar days (~105),
    # not the bar index, even with a hole in the middle of the series.
    ts_gap = np.concatenate([ts0 + np.arange(700) * BAR,
                             ts0 + 900 * BAR + np.arange(4300) * BAR])
    r_gap = np.concatenate([rng.normal(0, .004, 700), rng.normal(0, .004, 4300)])
    m_gap = new_metrics(r_gap, ts_gap)
    expected_days = len({int(t // 86400000) for t in ts_gap})
    failures.append("gap") if not check(
        "daily grouping follows real UTC days, not bar index",
        m_gap["daily_return_count"] == expected_days,
        f"daily_count={m_gap['daily_return_count']} expected={expected_days}") else None

    bar_idx = np.arange(n)
    m_fake = new_metrics(r, bar_idx)
    failures.append("fake-ts") if not check(
        "bar-index timestamps collapse to a fake single day (why v1 was invalid)",
        m_fake["daily_return_count"] <= 2, f"daily_count={m_fake['daily_return_count']}") else None

    r = rng.normal(0, 0.004, n)
    m = new_metrics(r, ts)
    failures.append("mdd-range") if not check(
        "mdd is a fraction in [0,1]", 0.0 <= m["mdd"] <= 1.0, f"mdd={m['mdd']:.6f}") else None

    r = rng.normal(0, 0.004, n); r[500] = -1.5
    m = new_metrics(r, ts)
    failures.append("insolvency") if not check(
        "insolvency -> solvent=False and mdd pinned to 1.0",
        (not m["solvent"]) and abs(m["mdd"] - 1.0) < 1e-12,
        f"solvent={m['solvent']} mdd={m['mdd']}") else None

    r = np.zeros(n)
    m = new_metrics(r, ts)
    failures.append("flat") if not check(
        "flat series is solvent with zero sharpe",
        m["solvent"] and m["sharpe"] == 0.0 and abs(m["final_x"] - 1.0) < 1e-12) else None

    print("\n== 3. compounding is the real driver (additive vs multiplicative) ==")
    r = np.full(3000, 0.001)
    ts = ts0 + np.arange(3000) * BAR
    lg, nw = both(r, ts)
    legacy_x = 1.0 + 3000 * 0.001          # additive: 4.0
    compound_x = (1.001 ** 3000)            # multiplicative: 19.8
    print(f"  3000 bars of +0.1%: legacy_final_x={lg['final_x']:.4f} (additive) "
          f"new_final_x={nw['final_x']:.4f} (compound)")
    print(f"  additive gives {legacy_x:.4f}, compound gives {compound_x:.4f} "
          f"-> {compound_x/legacy_x:.2f}x difference, v2 is the correct one")
    failures.append("compound") if not check(
        "v2 final_x equals the multiplicative product",
        abs(nw["final_x"] - compound_x) < 1e-6) else None
    failures.append("additive") if not check(
        "legacy final_x equals the additive sum (documents the old inflation)",
        abs(lg["final_x"] - legacy_x) < 1e-9) else None

    print("\n== 4. determinism ==")
    a = new_metrics(r, ts); b = new_metrics(r, ts)
    failures.append("determinism") if not check(
        "same input -> identical output", a == b) else None

    print("\n== 5. GA v2 timestamp plumbing ==")
    sys.path.insert(0, str(ROOT))
    src = (ROOT / "research/ga_28c_30m_3y.py").read_text()
    failures.append("ga-plumbing") if not check(
        "GA evaluate() takes real bars, not bar indices",
        "scale_end, bars" in src and "np.array([start + i" not in src) else None

    print(f"\naccounting_version={ACCOUNTING_VERSION}")
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("RESULT: all cross-checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
