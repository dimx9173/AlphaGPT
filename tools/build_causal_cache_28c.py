#!/usr/bin/env python3
"""Build versioned causal feature cache for the 28-coin/3y dataset."""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.universe_28c import COINS_28C
from research.causal_12f import causal_features
DATA=ROOT/'data'/'data_3y'/'30m'; CACHE=ROOT/'data'/'data_3y'/'30m_causal'; CACHE.mkdir(parents=True,exist_ok=True)
VERSION='causal-v2-vectorized'
meta={}
for coin in COINS_28C:
    src=DATA/f'{coin}.csv'; out=CACHE/f'{coin}.npy'
    with src.open() as f: rows=list(csv.DictReader(f))
    d={k:[float(r[k]) for r in rows] for k in ('open','high','low','close','volume')}
    x=causal_features(d)
    np.save(out,x)
    st=src.stat(); meta[coin]={'source_size':st.st_size,'source_mtime_ns':st.st_mtime_ns,'rows':len(rows),'shape':list(x.shape),'version':VERSION}
    print(coin,x.shape,flush=True)
(CACHE/'metadata.json').write_text(json.dumps({'version':VERSION,'coins':COINS_28C,'meta':meta},indent=2))
print('cache_complete',CACHE)
