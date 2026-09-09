# STRATEGY BSD-1: baseline formula on BTC/SOL/DOGE/ASTER

## Config
- Formula: [3,2,7,2,7,11,15,4,4,6,6,10]
- Venue aster, perp 2x, fund 0.0005/bar, fee 0.0004 (aster default taker)
- 15m -> 4h bars (16x). Method: load_bars / build_mats / net_series_side / stats
  (same as run_grouptest678910.py).
- Grid per coin: lth in [0.85,0.88] x sth in [0.10,0.12,0.15] x cd in [3,6,12]
  x sl in [None,0.03,0.05] = 54 rows max. Select per-coin H1-best by H1 sharpe.

## Splits
- BTC/SOL/DOGE: 6570 4h bars, 85% train-cut (5584 train), OOS 986 bars,
  H1/H2 493/493.
- ASTER: 2107 4h bars only (2025-09-19 listing, short-history caveat).
  85% split: 1790 train / 317 OOS, H1/H2 158/159.
- Combo: equal-weight mean of 4 coins at per-coin H1-best params,
  positionally aligned to common end (overlap L=2107 4h bars).
  Combo H1/H2 windows follow the ASTER split (idx 1790:1948 / 1948:2107),
  which is OOS for all 4 coins (inside BTC/SOL/DOGE H2 calendar window).

## Per-coin H1-best params + verification (H2 / FULL)
| coin | best (lth,sth,cd,sl) | H1 sh | H2 sh | H2 ann | H2 mdd | FULL sh | FULL ann | FULL mdd |
|---|---|---|---|---|---|---|---|---|
| BTC | (0.85,0.12,12,0.05) | 2.922 | -0.329 | -0.210 | 0.442 | 0.913 | 0.676 | 1.032 |
| SOL | (0.85,0.10,3,None) | 2.481 | -0.869 | -0.885 | 0.635 | 0.320 | 0.471 | 3.209 |
| DOGE | (0.85,0.15,12,None) | 2.573 | 0.507 | 0.465 | 0.593 | 0.248 | 0.381 | 3.390 |
| ASTER | (0.85,0.10,3,None) | 7.309 | -2.830 | -4.877 | 0.482 | 0.985 | 2.104 | 2.511 |

## Combo (equal-weight, end-aligned overlap)
| seg | sharpe | ann | mdd | n |
|---|---|---|---|---|
| H1 (idx 1790:1948) | 6.144 | 2.268 | 0.061 | 158 |
| H2 (idx 1948:2107) | -5.859 | -4.623 | 0.424 | 159 |
| FULL (overlap) | 2.831 | 2.286 | 0.515 | 2107 |

## H2 pairwise PnL corr (n=159, aligned tails)
- BTC_SOL 0.605, BTC_DOGE 0.614, SOL_DOGE 0.592
- BTC_ASTER -0.032, SOL_ASTER -0.133, DOGE_ASTER -0.063
- Majors correlate ~0.6 with each other; ASTER ~uncorrelated.

## Read
- H1-best params do NOT verify on H2 for BTC/SOL/ASTER (H2 sharpe <= 0);
  only DOGE holds positive H2 (0.507). Combo H2 is deeply negative (-5.86),
  driven by ASTER H1->H2 sign flip (7.31 -> -2.83).
- ASTER short-history caveat: 158/159-bar H1/H2 on a 2107-bar listing;
  H1 sharpe 7.3 is an overfit-prone micro-sample. Do not trust stand-alone.
- Combo H1 sharpe 6.14 on n=158 is also micro-sample; H2 shows no
  diversification benefit (majors correlated, ASTER uncorrelated but negative).
- Next (BSD-2): regime filter / long-only / vol-target variants before sizing.

## Artifacts
- backtest_BSD1_baseline.json (per-coin best + H1/H2/FULL + combo + corr)
- run_bsd1.py (this BSD-1 script)
