"""Decompose the contrarian result: gross edge vs fee drag.

The probe reported Sharpe -13 for a lag-1 fade. That is extreme enough to
check for a bug before believing it. If the gross (zero-fee) result is mildly
positive and the fee is what destroys it, the number is real and it says the
reversion is too small to trade at 30m. If gross is also -13, there is a bug.
"""
import sys
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import accounting_bar_returns, load_real_funding, metrics
from research.splits_28c import split_indices

common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
a, b = split_indices(n, common[0], common[-1])["lockbox"]
funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64)) for c in GA.COINS_28C}

for lag in (1, 48):
    print("=" * 70)
    print("lag %d  -- gross vs net" % lag)
    print("=" * 70)
    for fee in (0.0, 0.0001, 0.0004, 0.0010):
        nets, bhs, turns = [], [], []
        for c in GA.COINS_28C:
            r_all = np.asarray(returns[c], float)
            r = r_all[a:b]
            sig = np.zeros(len(r)); sig[lag:] = -r_all[a:b][:-lag]
            sd = float(np.std(sig[:len(sig)//2]))
            pos = np.tanh(sig / (sd + 1e-9)) * GA.POSITION_CAP
            pos = np.roll(pos, 1); pos[0] = 0.0
            turn = float(np.abs(np.diff(pos, prepend=0.0)).mean())
            f = funding[c][a:b]
            nets.append(accounting_bar_returns(pos, r, fee, f, GA.LEV))
            bhs.append(accounting_bar_returns(np.full(len(pos), GA.POSITION_CAP), r, fee, f, GA.LEV))
            turns.append(turn)
        sh = metrics(np.mean(np.stack(nets), axis=0), common[a:b])["sharpe"]
        bsh = metrics(np.mean(np.stack(bhs), axis=0), common[a:b])["sharpe"]
        print("  fee %.4f  strategy %+8.3f   B&H %+7.3f   excess %+8.3f   turnover/bar %.4f"
              % (fee, sh, bsh, sh - bsh, np.mean(turns)))
    print()
