"""Causal 12-factor features and formula evaluation for offline research."""
from __future__ import annotations
import math
import numpy as np

OPS = {
    12: ("ADD", 2), 13: ("SUB", 2), 14: ("MUL", 2), 15: ("DIV", 2),
    16: ("NEG", 1), 17: ("ABS", 1), 18: ("SIGN", 1), 19: ("GATE", 3),
    20: ("JUMP", 1), 21: ("DECAY", 1), 22: ("DELAY1", 1), 23: ("MAX3", 1),
    24: ("ZSCORE", 1), 25: ("RANK", 1), 26: ("TS_RANK", 1),
    27: ("DELTA", 1), 28: ("CORR", 2),
}
ROBUST_IDS = {0, 3, 4, 5, 6, 8, 9, 11}
NORM_WINDOW = 200


def _windows(x, window):
    x=np.asarray(x,dtype=np.float64)
    pad=np.pad(x,(window-1,0),mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad,window)[:len(x)]


def _trailing_mean(x, window):
    return _windows(x,window).mean(axis=1)


def _trailing_robust(x, window=NORM_WINDOW):
    x=np.asarray(x,dtype=np.float64)
    wins=_windows(x,window)
    med=np.median(wins,axis=1)
    mad=np.median(np.abs(wins-med[:,None]),axis=1)+1e-6
    return np.clip((x-med)/mad,-5,5)


def _ret(close):
    out=np.zeros(len(close),dtype=np.float64)
    out[1:]=np.log(np.maximum(close[1:],1e-12)/np.maximum(close[:-1],1e-12))
    return out


def _rolling_sum(x,w):
    out=np.zeros(len(x),dtype=np.float64)
    for t in range(len(x)):
        out[t]=np.sum(x[max(0,t-w+1):t+1])
    return out


def causal_features(data: dict) -> np.ndarray:
    close=np.asarray(data['close'],dtype=np.float64)
    open_=np.asarray(data['open'],dtype=np.float64)
    high=np.asarray(data['high'],dtype=np.float64)
    low=np.asarray(data['low'],dtype=np.float64)
    volume=np.asarray(data['volume'],dtype=np.float64)
    n=len(close); ret=_ret(close)
    prev_close=np.concatenate(([close[0]],close[:-1]))
    prev_vol=np.concatenate(([volume[0]],volume[:-1]))
    # 0 RET
    f0=_trailing_robust(ret)
    # 1 liquidity score, via Amihud illiquidity
    #
    # This used to read 'liquidity' and 'fdv' and fall back to fixed constants
    # when they were absent. The 28c CSVs never carried those columns, so the
    # factor was a constant 0.4 on every bar of every coin: a silent placeholder
    # that still occupied a slot in the formula grammar, and still satisfied
    # every "is this factor well formed" check because 0.4 is a legal number.
    #
    # Neither input is recoverable. Binance publishes no historical order book
    # depth and no historical fully-diluted valuation, so the original
    # definition cannot be evaluated at this horizon on any venue.
    #
    # Amihud illiquidity needs only close and volume, which the contract does
    # have: ILLIQ = |return| / dollar_volume, the price impact per dollar
    # traded. A deeper book moves less price per dollar, so lower ILLIQ means
    # more liquid and the sign is flipped so that higher still means "more
    # liquid", matching what the old factor claimed to mean. The trailing
    # median/MAD normalisation is the same one the other robust factors use and
    # keeps the factor strictly causal.
    # The log is load-bearing. Amihud is a ratio spanning several orders of
    # magnitude (BTC sits near 7e-12 with a p99 about 4x the median), so the
    # raw ratio's median/MAD is tiny and the normalised factor collapses to
    # ~1e-5 of variation -- technically non-constant, practically dead. Taking
    # the log first puts the factor on the same [-5,5] footing as its peers.
    dollar_vol=volume*np.maximum(close,1e-12)
    illiq=np.log(np.maximum(np.abs(ret),1e-12)/np.maximum(dollar_vol,1e-12))
    f1=_trailing_robust(-illiq)
    # 2 pressure
    f2=np.tanh(((close-open_)/(high-low+1e-9))*3)
    # 3 FOMO acceleration
    vol_chg=(volume-prev_vol)/(prev_vol+1.0)
    fomo=np.zeros(n); fomo[1:]=vol_chg[1:]-vol_chg[:-1]
    f3=_trailing_robust(np.clip(fomo,-5,5))
    # 4 pump deviation
    ma=_trailing_mean(close,20)
    f4=_trailing_robust((close-ma)/(ma+1e-9))
    # 5 log volume
    f5=_trailing_robust(np.log1p(np.maximum(volume,0)))
    # 6 volatility clustering
    f6=_trailing_robust(np.sqrt(_trailing_mean(ret*ret,10)+1e-9))
    # 7 momentum reversal
    mom=_rolling_sum(ret,5)
    prev_mom=np.concatenate(([mom[0]],mom[:-1]))
    f7=((mom*prev_mom)<0).astype(np.float64)
    # 8 relative strength / RSI
    delta=close-prev_close
    gains=np.maximum(delta,0); losses=np.maximum(-delta,0)
    avg_gain=_trailing_mean(gains,14); avg_loss=_trailing_mean(losses,14)
    rsi=100-100/(1+(avg_gain+1e-9)/(avg_loss+1e-9))
    f8=_trailing_robust((rsi-50)/50)
    # 9 high-low range
    f9=_trailing_robust((high-low)/(close+1e-9))
    # 10 close position
    f10=(close-low)/(high-low+1e-9)
    # 11 volume trend
    f11=_trailing_robust((volume-prev_vol)/(prev_vol+1.0))
    return np.stack([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11],axis=0)


def _delay(x,d):
    out=np.zeros_like(x)
    if d<len(x): out[d:]=x[:-d]
    return out


def _zscore(x,window=60):
    wins=_windows(x,window)
    return np.clip((x-wins.mean(axis=1))/(wins.std(axis=1)+1e-6),-5,5)


def _rank(x,window=60):
    wins=_windows(x,window)
    return (wins<=x[:,None]).mean(axis=1)*2-1


def _ts_rank(x,window=20):
    return _rank(x,window)


def _rolling_sum(x, window):
    """Sum over each trailing window, matching _windows(x, window).sum(axis=1).

    _windows edge-pads on the left, so the first window-1 rows repeat x[0].
    Reproducing that padding here keeps the rolling moments identical to the
    windowed form instead of merely similar.
    """
    pad=np.pad(np.asarray(x,dtype=np.float64),(window-1,0),mode="edge")
    c=np.concatenate(([0.0],np.cumsum(pad)))
    return c[window:window+len(x)]-c[:len(x)]


def _corr(x,y,window=60):
    """Rolling Pearson correlation in O(n) rather than O(n*window).

    The windowed form materialises an (n, window) view and reduces along it,
    which made _corr 88% of search time: the formula grammar makes CORR a
    terminal token, so every evaluation pays for it on every coin.

    The rolling moments below compute the same quantities from prefix sums.
    Two details are load-bearing for equivalence: the left edge padding must
    match _windows, and the correlation must be the population form (ddof=0),
    which is what ndarray.std returns on the centred window.
    """
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    n=len(x)
    if n<window:
        return np.zeros(n,dtype=np.float64)
    sx=_rolling_sum(x,window); sy=_rolling_sum(y,window)
    sxx=_rolling_sum(x*x,window); syy=_rolling_sum(y*y,window); sxy=_rolling_sum(x*y,window)
    inv=1.0/window
    mx=sx*inv; my=sy*inv
    cov=sxy*inv-mx*my
    varx=sxx*inv-mx*mx; vary=syy*inv-my*my
    # Rounding can push a true-zero variance a hair below zero.
    stdx=np.sqrt(np.maximum(varx,0.0)); stdy=np.sqrt(np.maximum(vary,0.0))
    return np.clip(cov/(stdx*stdy+1e-6),-1,1)


def _apply(op,args):
    if op=='ADD': return args[0]+args[1]
    if op=='SUB': return args[0]-args[1]
    if op=='MUL': return args[0]*args[1]
    if op=='DIV': return args[0]/(args[1]+1e-6)
    if op=='NEG': return -args[0]
    if op=='ABS': return np.abs(args[0])
    if op=='SIGN': return np.sign(args[0])
    if op=='GATE': return np.where(args[0]>0,args[1],args[2])
    if op=='JUMP': return np.maximum(_zscore(args[0])-3,0)
    if op=='DECAY': return args[0]+0.8*_delay(args[0],1)+0.6*_delay(args[0],2)
    if op=='DELAY1': return _delay(args[0],1)
    if op=='MAX3': return np.maximum.reduce([args[0],_delay(args[0],1),_delay(args[0],2)])
    if op=='ZSCORE': return _zscore(args[0])
    if op=='RANK': return _rank(args[0])
    if op=='TS_RANK': return _ts_rank(args[0])
    if op=='DELTA': return args[0]-_delay(args[0],1)
    if op=='CORR': return _corr(args[0],args[1])
    raise ValueError(op)


def evaluate_formula(formula, features):
    stack=[]
    for token in formula:
        token=int(token)
        if token<12: stack.append(features[token])
        else:
            if token not in OPS: return None
            op,arity=OPS[token]
            if len(stack)<arity: return None
            args=[stack.pop() for _ in range(arity)][::-1]
            stack.append(_apply(op,args))
    if len(stack)!=1: return None
    return np.nan_to_num(stack[0],nan=0.0,posinf=5.0,neginf=-5.0)


def causal_signal(formula,data):
    return evaluate_formula(formula,causal_features(data))


# ---------------------------------------------------------------------------
# Fail loud on degenerate factors
# ---------------------------------------------------------------------------
#
# Two separate defects in this project were the same defect: a missing input
# silently became a constant, and a constant is a perfectly legal number, so
# nothing complained.
#
#   funding     flat +0.0005 at every settlement      -> short book credited
#   LIQ_SCORE   data.get('liquidity', 1e7)            -> constant 0.4
#
# The funding one cost 1.10 Sharpe on the v3c lockbox and flipped its sign. A
# placeholder that shifts every bar by the same amount is invisible to a Sharpe
# check, a drawdown check, and a formula-grammar check. It is only visible if
# something looks at the input instead of the output.
#
# So look at the input. A factor with no variance carries no information, and a
# search that is free to use it will happily build a formula around it.

FEATURE_NAMES = ("RET", "LIQ_SCORE", "PRESSURE", "FOMO", "PUMP_DEV", "LOG_VOL",
                 "VOL_CLUST", "MOM_REV", "DELTA_RSI", "HL_RANGE", "CLOSE_POS",
                 "VOL_TREND")

# A factor that varies by less than this over the whole series is treated as
# constant. It is deliberately far above float noise and far below any real
# factor's variation, so it catches placeholders and not rounding.
CONSTANT_TOL = 1e-9

# Factors that read inputs the offline contract does not carry, and what to
# fetch if one is ever needed. Kept here so the next person does not have to
# rediscover it by reading a fall-through default.
FACTOR_INPUTS = {
    "LIQ_SCORE": ("close", "volume"),
    "FUNDING": ("funding history",),
}


def constant_features(features: np.ndarray, tol: float = CONSTANT_TOL) -> list:
    """Names of factors that carry no information on this series."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"expected a (n_factors, n_bars) array, got {features.shape}")
    out = []
    for i in range(features.shape[0]):
        row = features[i]
        if row.size == 0:
            continue
        if float(np.ptp(row)) <= tol or float(np.std(row)) <= tol:
            name = (FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i}")
            out.append(name)
    return out


def assert_features_vary(features: np.ndarray, tol: float = CONSTANT_TOL,
                         context: str = "") -> None:
    """Raise if any factor is constant.

    A constant factor is either a placeholder for data that was never loaded or
    a bug in the factor itself. Both invalidate a search that used it, so this
    is a hard stop rather than a warning: continuing produces a number that
    looks like a result and is not one.
    """
    bad = constant_features(features, tol)
    if not bad:
        return
    detail = ", ".join(
        f"{name} (needs {FACTOR_INPUTS[name]})" if name in FACTOR_INPUTS else name
        for name in bad)
    where = f" for {context}" if context else ""
    raise ValueError(
        f"constant feature(s){where}: {detail}. A factor with no variance "
        f"carries no information and usually means an input was never loaded "
        f"and a default was substituted. Refusing to evaluate a search that "
        f"used it."
    )
