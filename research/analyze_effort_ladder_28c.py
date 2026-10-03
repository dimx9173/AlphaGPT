#!/usr/bin/env python3
"""Dose-response of search effort on holdout excess Sharpe.

Read the SHAPE of the real and null curves, not their level. Three shapes
mean three different things and they need different fixes:

  both curves falling      -> overfitting. More search fits the training
                             window harder. Fix: less effort, or regularise.
  null falls, real flat    -> there IS structure and effort destroys it.
                             Fix: regularise, do not reduce effort.
  both flat                -> effort is not the variable. The factor set has
                             no conditional timing power to find at all.
"""

import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFORTS = (2, 5, 10, 20)


def load(outdir):
    rows = []
    for p in sorted(Path(outdir).glob("g*_s*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        h = d.get("holdout") or {}
        f = h.get("formula") or []
        rows.append({
            "gens": d.get("generations"),
            "data": ("null" if d.get("null_data", "none") != "none" else "real"),
            "seed": d.get("seed"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "sharpe": h.get("portfolio_sharpe"),
            "pos_coins": h.get("positive_coins"),
            "mean_pos": h.get("mean_position"),
            "n_factors": sum(1 for x in f if x < 12),
        })
    return rows

def desc(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return "n=%d %+.4f +/- %.4f" % (len(xs), st.mean(xs), sd / math.sqrt(len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "effort_ladder")
    args = ap.parse_args()
    rows = load(args.outdir)
    if not rows:
        print("no results in", args.outdir)
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["gens"], r["data"])].append(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "effort_ladder")
    args = ap.parse_args()
    rows = load(args.outdir)
    if not rows:
        print("no results in", args.outdir)
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["gens"], r["data"])].append(r)
    efforts = sorted({r["gens"] for r in rows})
    print("========================================================================")
    efforts = sorted({r["gens"] for r in rows})
    print("========================================================================")
    print("========================================================================")
    hdr = "  %-12s %-26s %s" % ("generations", "real", "null")
    print(hdr)
    for g in efforts:
        rr = by.get((g, "real"), [])
        nn = by.get((g, "null"), [])
        e = lambda rs: [r["excess"] for r in rs]
        print("  %-12s %-26s %s" % (g, desc(e(rr)), desc(e(nn))))
    print()
    print("  %-12s %s" % ("generations", "real minus null (higher = more edge)"))
    gaps = {}
    for g in efforts:
        rr = [r["excess"] for r in by.get((g, "real"), [])]
        nn = [r["excess"] for r in by.get((g, "null"), [])]
        gaps[g] = (st.mean(rr) - st.mean(nn)) if rr and nn else None
        s = ("%+.4f" % gaps[g]) if gaps[g] is not None else "n/a"
        print("  %-12s %s" % (g, s))
    print()
    print("SHAPE, which is the whole point of this ladder")
    print("========================================================================")
    e = lambda g, d: [r["excess"] for r in by.get((g, d), [])]
    try:
        r0, r1 = st.mean(e(efforts[0], "real")), st.mean(e(efforts[-1], "real"))
        n0, n1 = st.mean(e(efforts[0], "null")), st.mean(e(efforts[-1], "null"))
        dr, dn = r1 - r0, n1 - n0
        print("  real  %+.3f -> %+.3f   change %+.3f" % (r0, r1, dr))
        print("  null  %+.3f -> %+.3f   change %+.3f" % (n0, n1, dn))
        print()
        if dr < -0.2 and dn < -0.2:
            print("  BOTH FALL -> overfitting. More search fits the training")
            print("  window harder on real and on synthetic alike. Fix: less")
            print("  effort or regularisation. Do not expand the search space.")
        elif dn < -0.2 and dr >= -0.2:
            print("  NULL FALLS, REAL FLAT -> structure exists and effort is")
            print("  destroying it. Fix: regularise, keep the effort.")
        elif abs(dr) < 0.2 and abs(dn) < 0.2:
            print("  BOTH FLAT -> effort is not the variable. The factor set has")
            print("  no conditional timing power to find. Fix: change factors.")
        else:
            print("  MIXED -> read the two changes above before deciding.")
    except Exception as exc:
        print("  (could not classify: %s)" % exc)

        print("  (could not classify: %s)" % exc)

    print("SUPPORTING (reported, not decisive)")
    print("========================================================================")
    print("  gens  data    absSharpe            posCoins           nFactors          meanPos")
    for g in efforts:
        for d in ("real", "null"):
            rs = by.get((g, d), [])
            ex = lambda k: [r[k] for r in rs]
            print("  %-5s %-7s %-19s %-18s %-17s %s"
                  % (g, d, desc(ex("sharpe")), desc(ex("pos_coins")),
                     desc(ex("n_factors")), desc(ex("mean_pos"))))
    print()
    (args.outdir / "analysis.json").write_text(json.dumps({
        "rows": rows, "real_null_gap": {str(k): v for k, v in gaps.items()}}, indent=2))
    print("wrote", args.outdir / "analysis.json")


if __name__ == "__main__":
    main()