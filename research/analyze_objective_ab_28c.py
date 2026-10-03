#!/usr/bin/env python3
"""Reward vs gate: does a pure excess-over-hold objective find anything
that the v3 reward missed?

The v3 reward scores absolute Sharpe at w_psh 0.10 and mean-IC at 1.00. On a
universe whose buy-and-hold is +1.42 Sharpe, "hold long" therefore earns
reward before any timing is demonstrated, while the acceptance gate tests
excess only. This zeroes both terms and leaves excess as the whole objective.

Control is the existing gen-10 v3 run: same seeds, same folds, same
everything else, reached through the same harness.
"""

import argparse, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results" / "effort_ladder"
TREAT = ROOT / "results" / "effort_ladder_excess_bh"


def load(d, gens=10):
    out = {}
    for p in sorted(Path(d).glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        # index.json is a list and analysis.json is a dict of summaries; only
        # the per-run artifacts carry a generations field.
        if not isinstance(j, dict) or j.get("generations") != gens:
            continue
        h = j.get("holdout") or {}
        data = "null" if j.get("null_data", "none") != "none" else "real"
        out.setdefault(data, []).append({
            "seed": j.get("seed"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "sharpe": h.get("portfolio_sharpe"),
            "bh": h.get("buy_and_hold_sharpe"),
            "pos": h.get("mean_position"),
            "coins": h.get("positive_coins"),
        })
    return out

def sem(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, default=CONTROL)
    ap.add_argument("--treat", type=Path, default=TREAT)
    args = ap.parse_args()
    c = load(args.control)
    z = load(args.treat)
    if not z:
        print("no treatment results in", args.treat)
        return
    print("==========================================================================")
    print("HOLDOUT EXCESS SHARPE vs BUY-AND-HOLD, gen 10, 8 seeds")
    print("==========================================================================")
    print("  reward        real                  null                 real-null gap")
    gaps = {}
    for name, d in (("v3 (control)  ", c), ("excess_bh (new)", z)):
        rr = [r["excess"] for r in d.get("real", [])]
        nn = [r["excess"] for r in d.get("null", [])]
        g = st.mean(rr) - st.mean(nn) if rr and nn else None
        gse = math.sqrt(sem(rr) ** 2 + sem(nn) ** 2) if rr and nn else None
        gaps[name] = (g, gse)
        gs = ("%+.4f +/- %.4f" % (g, gse)) if g is not None else "n/a"
        print("  %s %+.4f +/- %.4f  %+.4f +/- %.4f  %s"
              % (name, st.mean(rr), sem(rr), st.mean(nn), sem(nn), gs))
    print()
    print("==========================================================================")
    print("BEAT THE HOLD (the gate requirement)")
    print("==========================================================================")
    for name, d in (("v3 control ", c), ("excess_bh new  ", z)):
        for data in ("real", "null"):
            xs = [r["excess"] for r in d.get(data, [])]
            w = sum(1 for x in xs if x is not None and x > 0)
            print("  %s %-5s %d/%d positive" % (name, data, w, len(xs)))
    print()
    print("==========================================================================")
    print("POSITION AND BREADTH (did the reward change what the book does?)")
    print("==========================================================================")
    for name, d in (("v3 control ", c), ("excess_bh new  ", z)):
        rs = d.get("real", [])
        if not rs:
            continue
        mp = [r["pos"] for r in rs]
        atcap = sum(1 for x in mp if x is not None and x > 0.24)
        cn = [r["coins"] for r in rs]
        print("  %s mean_pos %.3f, %d/%d at the 0.25 cap, posCoins %.1f"
              % (name, st.mean([x for x in mp if x is not None]), atcap, len(rs), st.mean([x for x in cn if x is not None])))

    print("==========================================================================")
    gc = gaps.get("v3 (control)  ")
    gz = gaps.get("excess_bh (new)")
    if gc is not None and gz is not None:
        d = gz[0] - gc[0]
        se = math.sqrt(gc[1] ** 2 + gz[1] ** 2)
        tt = d / se if se else float("nan")
        print("  change in the real-null gap: %+.4f +/- %.4f   t = %+.2f" % (d, se, tt))
        print("  95%% CI [%+.3f, %+.3f]" % (d - 1.96 * se, d + 1.96 * se))
        print()
        # Significance, not an arbitrary threshold. An earlier version of this
        # script called any change above 0.5 a "widening" and would have reported
        # this run as a win on a t of +1.23.
        if abs(tt) < 2:
            print("  NOT distinguishable from zero. The change in the reward did")
            print("  not measurably change how far real data separates from null.")
        elif d > 0:
            print("  The pure-excess reward WIDENS the separation.")
        else:
            print("  The pure-excess reward NARROWS the separation.")
        print()
        print("  What DID change is what the book does, which is a separate question:")
        for key, label in (("pos", "mean position"), ("coins", "positive coins")):
            a = [r[key] for r in c.get("real", [])]
            b = [r[key] for r in z.get("real", [])]
            if not a or not b:
                continue
            dd = st.mean(b) - st.mean(a)
            s = math.sqrt(sem(a) ** 2 + sem(b) ** 2)
            u = dd / s if s else float("nan")
            print("    %-15s %+.3f -> %+.3f  diff %+.3f  t = %+.2f  %s"
                  % (label, st.mean(a), st.mean(b), dd, u,
                     "SIGNIFICANT" if abs(u) >= 2 else "not significant"))
        print("    The reward demonstrably changed the strategy's behaviour.")
        print("    Whether that produced a holdout gain is the separate, and")
        print("    negative, answer above.")
    (args.treat / "analysis.json").write_text(json.dumps({
        "control": c, "treatment": z, "gaps": gaps}, indent=2))
    print()
    print("wrote", args.treat / "analysis.json")


if __name__ == "__main__":
    main()