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
    # 1 liquidity score (raw data has synthetic liquidity in training)
    liq=np.asarray(data.get('liquidity',np.full(n,1e7)),dtype=np.float64)
    fdv=np.asarray(data.get('fdv',np.full(n,1e8)),dtype=np.float64)
    f1=np.clip((liq/(fdv+1e-6))*4,0,1)
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


def _corr(x,y,window=60):
    a=_windows(x,window); b=_windows(y,window)
    a=a-a.mean(axis=1,keepdims=True); b=b-b.mean(axis=1,keepdims=True)
    cov=(a*b).mean(axis=1)
    den=a.std(axis=1)*b.std(axis=1)+1e-6
    return np.clip(cov/den,-1,1)


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
