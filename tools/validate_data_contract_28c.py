#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.data_contract_28c import *

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def inspect(path):
 with path.open(newline='') as f: rows=list(csv.DictReader(f))
 ts=[int(float(r['timestamp'])) for r in rows]; bad=0
 for r in rows:
  try:
   vals=[float(r[k]) for k in ('open','high','low','close','volume')]
   if not all(math.isfinite(x) for x in vals) or min(vals[:4])<=0 or vals[4]<0: bad+=1
  except Exception: bad+=1
 return {'rows':len(rows),'start_ms':ts[0] if ts else None,'end_ms':ts[-1] if ts else None,'sorted_unique':ts==sorted(set(ts)),'gaps':sum(1 for a,b in zip(ts,ts[1:]) if b-a!=STEP_MS),'bad_rows':bad,'sha256':sha256(path)}

def main():
 data=ROOT/'data'/'data_3y'/'30m'; out={}
 sets=[]
 for c in COINS_28C:
  p=data/f'{c}.csv'; x=inspect(p); x['coin']=c; out[c]=x; sets.append(set())
  with p.open() as f: sets[-1]={int(float(r['timestamp'])) for r in csv.DictReader(f)}
 common=sorted(set.intersection(*sets)); start,end=common[0],common[-1]
 full=[c for c in COINS_28C if c in FULL_3Y_25 and out[c]['rows']==52560]
 contract={'version':'data-contract-28c-v1','step_ms':STEP_MS,'coins':COINS_28C,'full_3y_25':full,'late_start':{c:out[c]['start_ms'] for c in ('KAS','RENDER','POL')},'common':{'bars':len(common),'start_ms':start,'end_ms':end,'history_years':HISTORY_YEARS_COMMON_28},'lockbox_start_ms':lockbox_start_ms(end),'embargo_bars':EMBARGO_BARS,'files':out}
 failures=[]
 if len(common)!=COMMON_28_BARS: failures.append(f'common bars {len(common)} != {COMMON_28_BARS}')
 if len(full)!=25: failures.append(f'full 3y coins {len(full)} != 25')
 for c,x in out.items():
  if x['gaps'] or x['bad_rows'] or not x['sorted_unique']: failures.append(f'{c}: gaps/bad/sorted failure')
  if len(x.get('sha256','')) != 64: failures.append(f'{c}: invalid sha256')
 payload=json.dumps(contract,indent=2); p=ROOT/'results'/'data_contract_28c.v1.json'; p.write_text(payload); fixture=ROOT/'research'/'fixtures'/'data_contract_28c.v1.json';
 if not failures: fixture.write_text(payload)
 print(json.dumps({'common':contract['common'],'full_3y_count':len(full),'lockbox_start_ms':contract['lockbox_start_ms'],'output':str(p),'fixture':str(fixture),'failures':failures},indent=2))
 if failures: raise SystemExit(1)
if __name__=='__main__': main()
