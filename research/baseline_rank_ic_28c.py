#!/usr/bin/env python3
"""Rank/IC baseline scan for the 28-coin 3y causal universe."""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.universe_28c import COINS_28C
from research.causal_12f import evaluate_formula
DATA=ROOT/'data'/'data_3y'/'30m'; CACHE=ROOT/'data'/'data_3y'/'30m_causal'
NAMES=['RET','LIQ_SCORE','PRESSURE','FOMO','DEV','LOG_VOL','VOL_CLUSTER','MOM_REV','REL_STRENGTH','HL_RANGE','CLOSE_POS','VOL_TREND']


def load():
    maps={}; returns={}; common=None
    for c in COINS_28C:
        cache=CACHE/f'{c}.npy'
        if cache.exists(): maps[c]=np.load(cache)
        else:
            with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
            from research.causal_12f import causal_features
            d={k:[float(r[k]) for r in rows] for k in ('open','high','low','close','volume')}; maps[c]=causal_features(d)
        with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
        ts={int(r['timestamp']) for r in rows}; common=ts if common is None else common&ts
    common=sorted(common); n=len(common); outret={}
    for c in COINS_28C:
        with (DATA/f'{c}.csv').open() as f: rows=list(csv.DictReader(f))
        idx={int(r['timestamp']):i for i,r in enumerate(rows)}; close=np.array([float(rows[idx[t]]['close']) for t in common]); r=np.zeros(n); r[1:]=close[1:]/close[:-1]-1; outret[c]=r
    return maps,outret,n


def ic(a,b):
    if len(a)<20 or a.std()<1e-12 or b.std()<1e-12: return 0.0
    return float(np.corrcoef(a,b)[0,1])


def main():
    maps,returns,n=load(); tr=int(n*.6); va=int(n*.8); emb=200
    ranges={'train':(0,tr-emb),'validation':(tr+emb,va-emb),'oos':(va+emb,n)}
    out={'status':'baseline_only','data_dir':str(DATA),'bars':n,'splits':ranges,'features':{}}
    for fi,name in enumerate(NAMES):
        row={'per_coin':{}}
        for split,(a,b) in ranges.items():
            vals={c:ic(maps[c][fi][a:b-1],returns[c][a+1:b]) for c in COINS_28C}
            row[split]={'mean_ic':float(np.mean(list(vals.values()))),'median_ic':float(np.median(list(vals.values()))),'positive_coins':sum(x>0 for x in vals.values()),'per_coin':vals}
        out['features'][name]=row
    rank=sorted(out['features'].items(),key=lambda kv:kv[1]['validation']['mean_ic'],reverse=True)
    out['ranking_by_validation_ic']=[x[0] for x in rank]
    out['top_validation']=[{'feature':k,'mean_ic':v['validation']['mean_ic'],'oos_mean_ic':v['oos']['mean_ic'],'oos_positive':v['oos']['positive_coins']} for k,v in rank[:8]]
    p=ROOT/'results'/'baseline_rank_ic_28c_3y.json'; p.write_text(json.dumps(out,indent=2)); print(json.dumps(out['top_validation'],indent=2))
if __name__=='__main__': main()
