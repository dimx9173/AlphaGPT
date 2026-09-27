#!/usr/bin/env python3
"""Train-only threshold fit for top 28c rank-IC baseline factors."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.universe_28c import COINS_28C
from research.splits_28c import search_splits, split_indices
from research.accounting_28c import scheduled_funding_rates, position_from_signal, accounting_bar_returns, account_portfolio, compound_equity, max_drawdown, daily_sharpe, metrics, ACCOUNTING_VERSION
DATA=ROOT/'data'/'data_3y'/'30m'; CACHE=ROOT/'data'/'data_3y'/'30m_causal'
FEATURES={'HL_RANGE':9,'LOG_VOL':5,'FOMO':3,'VOL_TREND':11}
BPY=17520.; LEV=2.; FEE=.0004; FUND=.0005
GRID=[0.0,0.25,0.5,0.75,1.0,1.25]

def load():
    maps={}; rets={}; ts=None
    for c in COINS_28C:
        with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
        tset={int(r['timestamp']) for r in rows}; ts=tset if ts is None else ts&tset
        cache=CACHE/f'{c}.npy'
        if cache.exists(): maps[c]=np.load(cache)
        else:
            from research.causal_12f import causal_features
            maps[c]=causal_features({k:[float(r[k]) for r in rows] for k in ('open','high','low','close','volume')})
    ts=sorted(ts); n=len(ts)
    for c in COINS_28C:
        with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
        idx={int(r['timestamp']):i for i,r in enumerate(rows)}
        maps[c]=maps[c][:,[idx[t] for t in ts]]
    out={}
    for c in COINS_28C:
        with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
        idx={int(r['timestamp']):i for i,r in enumerate(rows)}; close=np.array([float(rows[idx[t]]['close']) for t in ts]); r=np.zeros(n); r[1:]=close[1:]/close[:-1]-1; out[c]=r
    return maps,out,ts

def stats(leg):
    x=np.mean(np.stack(leg),axis=0); sd=x.std(ddof=1) if len(x)>1 else 0.; sh=float(x.mean()/(sd+1e-9)*np.sqrt(BPY)) if sd>1e-9 else 0.; eq=np.cumsum(x); peak=np.maximum.accumulate(np.r_[0.,eq])[1:]; mdd=float(((peak-eq)/np.maximum(peak,1e-9)).max()); shs=[]
    for z in leg:
        s=z.std(ddof=1) if len(z)>1 else 0.; shs.append(float(z.mean()/(s+1e-9)*np.sqrt(BPY)) if s>1e-9 else 0.)
    return {'sharpe':sh,'mdd':mdd,'positive_coins':sum(v>0 for v in shs),'leg_sharpes':shs,'net_sum':float(x.sum())}

def eval_factor(feature, long_thr, short_thr, maps, returns, timestamps, start, end):
    legs = []; leg_sharpes = []; leg_mdds = []; leg_solvents = []
    for c in COINS_28C:
        raw = maps[c][feature].astype(float)
        scale_end = max(start, min(len(raw), 200))
        mean = float(np.mean(raw[:scale_end]))
        sd = float(np.std(raw[:scale_end]) + 1e-9)
        z = (raw - mean) / sd
        p = np.where(z > long_thr, 1., np.where(z < short_thr, -1., 0.))
        p = np.roll(p, 1); p[0] = 0.
        r = returns[c][start:end]
        pos = p[start:end]
        turn = np.abs(np.diff(pos, prepend=0.))
        # Use shared accounting for this leg. Funding is a real scheduled cost
        # at 00:00/08:00/16:00 UTC; passing a zero array silently modelled a
        # funding-free perpetual, which overstated every long threshold.
        fnd = scheduled_funding_rates(timestamps[start:end], FUND)
        net = accounting_bar_returns(pos, r, FEE, fnd, LEV)
        legs.append(net)
        # Compute metrics for this leg using shared functions
        m = metrics(net, timestamps[start:end])
        leg_sharpes.append(m['sharpe'])
        leg_mdds.append(m['mdd'])
        leg_solvents.append(m['solvent'])
    
    # Portfolio returns (equal-weighted)
    x = np.mean(np.stack(legs), axis=0)
    # Compute portfolio metrics using shared accounting
    m = metrics(x, timestamps[start:end])
    sh = m['sharpe']
    mdd = m['mdd']
    solvent = m['solvent']
    shs = leg_sharpes
    return {'sharpe': sh, 'mdd': mdd, 'solvent': solvent, 'positive_coins': sum(v > 0 for v in shs), 'leg_sharpes': shs, 'leg_mdds': leg_mdds, 'leg_solvents': leg_solvents, 'net_sum': float(x.sum())}
def main():
    maps,returns,ts=load(); n=len(ts); contract_splits=split_indices(n,ts[0],ts[-1]); ranges=search_splits(n,ts[0],ts[-1]); lockbox_range=contract_splits['lockbox']; out={'status':'threshold_fit_paper_only','history_years':2.03,'bars':n,'splits':{'train':ranges['train'],'validation':ranges['validation'],'lockbox':lockbox_range},'results':{}}
    for name,fi in FEATURES.items():
        candidates=[]
        for lt in GRID:
            for st in [-x for x in GRID]:
                if st>=lt: continue
                trn=eval_factor(fi,lt,st,maps,returns,ts,*ranges['train']); val=eval_factor(fi,lt,st,maps,returns,ts,*ranges['validation']); candidates.append({'long':lt,'short':st,'train':trn,'validation':val})
        candidates.sort(key=lambda x:(x['train']['sharpe'],x['validation']['sharpe']),reverse=True); best=candidates[0]; oos=eval_factor(fi,best['long'],best['short'],maps,returns,ts,*lockbox_range); out['results'][name]={'best':best,'oos':oos,'candidate_count':len(candidates)}
    p=ROOT/'results'/'threshold_fit_28c_3y.json'; p.write_text(json.dumps(out,indent=1)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
