#!/usr/bin/env python3
"""Compare the fixed-12 and variable-length arms, real against null.

A higher number on real data is not the finding. Both arms optimise the
same thing on the same training window, and the variable arm searches a
larger space, so it will find a better fit to noise if noise is what is
there. The finding is the REAL-vs-NULL gap INSIDE each arm, and whether the
variable arm gap is wider than the fixed gap or merely equal to it.

The primary metric is holdout excess over buy-and-hold, because that is
what the acceptance gate tests. Absolute Sharpe is reported but is not
the decision: this universe drifted, so a positive Sharpe can still lose
to doing nothing.
"""

from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("12", "varlen")


def load(outdir):
    rows = []
    for p in sorted(Path(outdir).glob("*_s*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        arm = "varlen" if d.get("formula_len", [12, 12])[1] > 12 else "12"
        data = "null" if d.get("null_data", "none") != "none" else "real"
        h = d.get("holdout") or {}
        f = h.get("formula") or []
        rows.append({
            "arm": arm, "data": data, "seed": d.get("seed"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "sharpe": h.get("portfolio_sharpe"),
            "bh": h.get("buy_and_hold_sharpe"),
            "pos_coins": h.get("positive_coins"),
            "mean_pos": h.get("mean_position"),
            "n_factors": sum(1 for t2 in f if t2 < 12),
            "n_tokens": len(f),
            "folds": d.get("folds"),
        })
    return rows

def desc(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return "n=%d mean=%+.4f sd=%.4f median=%+.4f" % (
        len(xs), st.mean(xs), sd, st.median(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "results" / "varlen_ab")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = load(args.outdir)
    if not rows:
        print("no results in", args.outdir)
        return

    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["data"]), []).append(r)
    print("=" * 74)
    print("HOLDOUT EXCESS SHARPE vs BUY-AND-HOLD  (primary; this is what the gate tests)")
    print("=" * 74)
    for arm in ARMS:
        for data in ("real", "null"):
            g = by.get((arm, data), [])
            lab = ("fixed-12" if arm == "12" else "varlen-6..20") + " / " + data
            print("  %-26s %s" % (lab, desc([r["excess"] for r in g])))

    print()
    print("=" * 74)
    print("REAL minus NULL, inside each arm  (this gap is the only edge evidence)")
    print("=" * 74)
    gaps = {}
    for arm in ARMS:
        re_ = [r["excess"] for r in by.get((arm, "real"), [])]
        nu = [r["excess"] for r in by.get((arm, "null"), [])]
        g = st.mean(re_) - st.mean(nu) if re_ and nu else None
        gaps[arm] = g
        print("  %-12s real %+.4f  null %+.4f  gap %+.4f" % (
            ("fixed-12" if arm == "12" else "varlen"),
            st.mean(re_) if re_ else float("nan"),
            st.mean(nu) if nu else float("nan"), g if g is not None else float("nan")))
    print()
    print("=" * 74)
    print("VARLEN minus FIXED-12  (equal gains on real and null = multiple testing)")
    print("=" * 74)
    for data in ("real", "null"):
        a = [r["excess"] for r in by.get(("varlen", data), [])]
        b = [r["excess"] for r in by.get(("12", data), [])]
        print("  %-6s varlen %+.4f  fixed %+.4f  gain %+.4f" % (
            data, st.mean(a) if a else float("nan"),
            st.mean(b) if b else float("nan"),
            (st.mean(a) - st.mean(b)) if a and b else float("nan")))

    if gaps.get("12") is not None and gaps.get("varlen") is not None:
        d = gaps["varlen"] - gaps["12"]
        print()
        print("  difference of the real-null gaps (varlen - fixed) = %+.4f" % d)
        if d > 0:
            print("  -> varlen widens the real-null separation")
        elif d < 0:
            print("  -> varlen NARROWS it: a bigger search fitting noise better")
        else:
            print("  -> no change in separation")
    print()
    print("=" * 74)
    print("SUPPORTING (reported, not decisive)")
    print("=" * 74)
    for arm in ARMS:
        for data in ("real", "null"):
            g = by.get((arm, data), [])
            lab = ("fixed-12" if arm == "12" else "varlen") + "/" + data
            print("  %-18s absSharpe %s" % (lab, desc([r["sharpe"] for r in g])))
            print("  %-18s posCoins  %s" % ("", desc([r["pos_coins"] for r in g])))
            print("  %-18s nFactors %s" % ("", desc([r["n_factors"] for r in g])))
            print("  %-18s nTokens  %s" % ("", desc([r["n_tokens"] for r in g])))
            print("  %-18s meanPos  %s" % ("", desc([r["mean_pos"] for r in g])))
            print()

    if args.json:
        out = {"rows": rows, "real_null_gap": gaps}
        (args.outdir / "analysis.json").write_text(json.dumps(out, indent=2))
        print("wrote", args.outdir / "analysis.json")


if __name__ == "__main__":
    main()