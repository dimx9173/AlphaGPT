"""P0-3b permutation redesign (Top5-aware), combo-level block null.

Problem with research/run_plateau.py per-coin null: shuffling one leg while
the other leg stays real leaves half the combo intact, inflating null sharpe
(ETC null_mean 1.59, TRX 1.88 vs obs 2.06) so p>=0.05 is nearly guaranteed.

Redesign (in-sample only, read-only, no orders):
  (a) block-shuffle of the signal time axis (block=24 bars preserves
      autocorrelation) instead of full column shuffle;
  (b) combo-level test: BOTH legs shuffled in every null replicate (gated);
      single-leg shuffle kept as diagnostic only (not gated);
  (c) time-reversal null as a second, shuffle-free reference.
Metric: combo sharpe p-values for block sizes {1, 24, 96}.
Primary gate: block=24 both-legs null p < 0.05.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pathlib

import torch

from research.run_aa import build_mats, combo, leg_series, load_bars, stats
from research.run_plateau import _eval_combo

SEED = 7
COINS = ["ETC", "TRX"]
FEE = 0.0004
FUND = 0.0005
LOCKED = {"ETC": (0.88, 0.12, 18, None, 24), "TRX": (0.85, 0.12, 6, 0.05, 24)}
BASE_STH = 0.12
N_PERM = int(os.getenv("PERM2_N", "5000"))
BLOCK_SIZES = [1, 24, 96]
PRIMARY_BLOCK = 24
OUT_PATH = os.getenv("PERM2_OUT", "results/permutation2.json")


def block_perm(n, block, g):
    """Permute time-axis columns in contiguous blocks of size `block`.

    Within-block order is preserved (keeps autocorrelation); block order is
    shuffled. Returns a LongTensor index of length n. block=1 == full shuffle.
    """
    if block <= 1:
        return torch.randperm(n, generator=g)
    blocks = [list(range(s, min(s + block, n))) for s in range(0, n, block)]
    order = torch.randperm(len(blocks), generator=g).tolist()
    idx = [c for b in order for c in blocks[b]]
    return torch.tensor(idx, dtype=torch.long)


def reverse_idx(n):
    return torch.arange(n - 1, -1, -1, dtype=torch.long)


def _summ(nulls, obs):
    srt = sorted(nulls)
    m = sum(nulls) / len(nulls)
    var = sum((x - m) ** 2 for x in nulls) / max(len(nulls) - 1, 1)
    p = (1.0 + sum(1 for x in nulls if x >= obs)) / (1.0 + len(nulls))
    return {"n_perm": len(nulls), "observed": obs,
            "null_mean": round(m, 4), "null_std": round(var ** 0.5, 4),
            "null_median": round(srt[len(srt) // 2], 4),
            "null_max": round(max(nulls), 4), "null_min": round(min(nulls), 4),
            "p_value": round(p, 4), "gate_p_lt_0_05": bool(p < 0.05)}


def _both_leg_null(mats_is, idx_fn):
    pair = []
    for coin in COINS:
        raw, rt, sg = mats_is[coin]
        lth, _, cd, sl, ts = LOCKED[coin]
        net, _, _ = leg_series(raw, rt, sg[:, idx_fn(sg.shape[1])],
                               lth, BASE_STH, cd, sl, fee=FEE, fund=FUND)
        pair.append(net)
    return stats(combo(pair, [0.5, 0.5]))["sharpe"]


def _single_leg_null(mats_is, legs_obs, coin, idx_fn):
    raw, rt, sg = mats_is[coin]
    lth, _, cd, sl, ts = LOCKED[coin]
    net, _, _ = leg_series(raw, rt, sg[:, idx_fn(sg.shape[1])],
                           lth, BASE_STH, cd, sl, fee=FEE, fund=FUND)
    other = 1 - COINS.index(coin)
    pair = [net, legs_obs[other]] if COINS.index(coin) == 0 else [legs_obs[other], net]
    return stats(combo(pair, [0.5, 0.5]))["sharpe"]


def main():
    freeze = json.load(open("research/oos_freeze.json"))
    cut = int(freeze["cut"]["cut_index"])
    assert cut == 6580, "freeze cut moved: %s" % cut
    logp = pathlib.Path("logs/perm2.log")
    logp.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(logp, "a") as f:
            f.write(msg + "\n")

    open(logp, "w").write("perm2 start\n")
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    mats_is = {c: build_mats(full[c][0:cut]) for c in COINS}
    s_is, _, legs_is = _eval_combo(mats_is, BASE_STH, BASE_STH)
    obs = s_is["sharpe"]
    log("IS n=%d cut=%d obs combo sharpe=%.3f" % (n, cut, obs))

    blocks = {}
    diag = {}
    for b in BLOCK_SIZES:
        g = torch.Generator().manual_seed(SEED + b)
        fn = lambda nn, _b=b, _g=g: block_perm(nn, _b, _g)
        both = [_both_leg_null(mats_is, fn) for _ in range(N_PERM)]
        blocks[str(b)] = _summ(both, obs)
        log("block=%d BOTH null_mean=%.3f std=%.3f max=%.3f obs=%.3f p=%.4f => %s" % (
            b, blocks[str(b)]["null_mean"], blocks[str(b)]["null_std"],
            blocks[str(b)]["null_max"], obs, blocks[str(b)]["p_value"],
            "PASS" if blocks[str(b)]["gate_p_lt_0_05"] else "FAIL"))
        dg = {}
        for coin in COINS:
            g2 = torch.Generator().manual_seed(SEED + b + 1000 + COINS.index(coin))
            fn2 = lambda nn, _b=b, _g=g2: block_perm(nn, _b, _g)
            nulls = [_single_leg_null(mats_is, legs_is, coin, fn2) for _ in range(N_PERM)]
            s = _summ(nulls, obs)
            s["diagnostic_only"] = True
            dg[coin] = s
            log("block=%d DIAG %s null_mean=%.3f p=%.4f (not gated)" % (
                b, coin, s["null_mean"], s["p_value"]))
        diag[str(b)] = dg

    # (c) time-reversal null: flip time axis of both legs, deterministic.
    rev = _both_leg_null(mats_is, reverse_idx)
    reversal = {"combo_sharpe": round(rev, 4),
                "obs_minus_rev": round(obs - rev, 4),
                "obs_gt_rev": bool(obs > rev)}

    primary = blocks[str(PRIMARY_BLOCK)]
    out = {
        "config": {"in_sample": [0, cut], "block_sizes": BLOCK_SIZES,
                   "primary_block": PRIMARY_BLOCK, "n_perm": N_PERM, "seed": SEED,
                   "engine": "combo-level block shuffle both legs, locked params",
                   "oos_role": "not used; in-sample only"},
        "observed_combo_sharpe": obs,
        "blocks_both_legs": blocks,
        "single_leg_diagnostic": diag,
        "reversal": reversal,
        "verdict": {
            "combo_block24_pass": bool(primary["gate_p_lt_0_05"]),
            "all_blocks_pass": bool(all(v["gate_p_lt_0_05"] for v in blocks.values())),
            "reversal_obs_gt_rev": bool(obs > rev),
        },
    }
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    log("saved %s verdict=%s" % (OUT_PATH, json.dumps(out["verdict"])))


if __name__ == "__main__":
    main()
