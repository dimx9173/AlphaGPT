#!/usr/bin/env python3
"""Download/validate 3 years of Binance USDT-M 30m klines for COINS_28C.

Writes data/data_3y/30m/<COIN>.csv. Resume-safe; never overwrites 1y data.
"""
from __future__ import annotations
import csv, json, math, os, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.universe_28c import COINS_28C

OUT=ROOT/'data'/'data_3y'/'30m'
STEP=30*60*1000
LIMIT=1000
BINANCE_SYMBOLS={c:f'{c}USDT' for c in COINS_28C}
BINANCE_SYMBOLS['PEPE']='1000PEPEUSDT'
START= int((time.time()-3*365*24*3600)*1000)


def get_klines(symbol,start):
    url=(f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}'
         f'&interval=30m&limit={LIMIT}&startTime={start}')
    req=urllib.request.Request(url,headers={'User-Agent':'AlphaGPT/3y-30m'})
    with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read())


def normalize(rows):
    out=[]; seen=set()
    for k in rows:
        ts=int(k[0])
        if ts in seen: continue
        seen.add(ts)
        out.append([str(ts),k[1],k[2],k[3],k[4],k[5],k[7],str(k[8])])
    out.sort(key=lambda x:int(x[0])); return out


def download(coin):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'{coin}.csv'; symbol=BINANCE_SYMBOLS[coin]
    existing=[]
    if path.exists():
        with path.open() as f: existing=list(csv.reader(f))
    if len(existing)>1:
        header=existing[0]; old=existing[1:]
        last=int(float(old[-1][0])); start=max(START,last+STEP)
        print(f'{coin}: resume {len(old)} rows',flush=True)
    else:
        header=['timestamp','open','high','low','close','volume','quote_volume','trades']; old=[]; start=START
    new=[]; cur=start; now=int(time.time()*1000)
    while cur<now:
        for attempt in range(5):
            try:
                data=get_klines(symbol,cur); break
            except Exception as exc:
                if attempt==4: raise
                print(f'{coin}: retry {attempt+1}: {exc}',flush=True); time.sleep(2**attempt)
        if not data: break
        new.extend(data)
        nxt=int(data[-1][0])+STEP
        if nxt<=cur: raise RuntimeError(f'{coin}: non-advancing Binance cursor')
        cur=nxt
        if len(data)<LIMIT: break
        time.sleep(0.12)
    new_rows=normalize(new)
    have={int(float(r[0])) for r in old}
    merged=old+[r for r in new_rows if int(r[0]) not in have]
    merged.sort(key=lambda r:int(r[0]))
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(merged)
    return len(merged),merged[0][0] if merged else None,merged[-1][0] if merged else None


def validate(coin):
    path=OUT/f'{coin}.csv'
    with path.open() as f: rows=list(csv.DictReader(f))
    ts=[int(float(r['timestamp'])) for r in rows]
    gaps=sum(1 for a,b in zip(ts,ts[1:]) if b-a!=STEP)
    bad=0
    for r in rows:
        try:
            vals=[float(r[k]) for k in ('open','high','low','close','volume')]
            if not all(math.isfinite(x) for x in vals) or min(vals[:4])<=0 or vals[4]<0: bad+=1
        except Exception: bad+=1
    return {'coin':coin,'rows':len(rows),'start':ts[0] if ts else None,'end':ts[-1] if ts else None,'gaps':gaps,'bad_rows':bad}

if __name__=='__main__':
    only=sys.argv[1:] or COINS_28C
    results=[]
    for c in only:
        if c not in COINS_28C: raise SystemExit(f'not in 28c manifest: {c}')
        try:
            n,s,e=download(c); v=validate(c); results.append(v); print(json.dumps(v),flush=True)
        except Exception as exc:
            results.append({'coin':c,'error':f'{type(exc).__name__}: {exc}'})
            print(json.dumps(results[-1]),flush=True)
    print('SUMMARY',json.dumps(results),flush=True)
