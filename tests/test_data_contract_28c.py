from pathlib import Path
import json
from research.data_contract_28c import *
from research.data_contract_28c import lockbox_start_ms
from research.splits_28c import split_indices, search_splits
ROOT=Path(__file__).resolve().parents[1]

def test_contract_25_vs_28():
 d=json.loads((ROOT/'research'/'fixtures'/'data_contract_28c.v1.json').read_text())
 assert len(d['full_3y_25'])==25
 assert d['common']['bars']==COMMON_28_BARS
 assert d['common']['history_years'] == 2.03
 assert d['embargo_bars'] == 200
 assert d['lockbox_start_ms'] == lockbox_start_ms(d['common']['end_ms'])
 from datetime import datetime, timezone
 end_dt=datetime.fromtimestamp(d['common']['end_ms']/1000,tz=timezone.utc)
 lock_dt=datetime.fromtimestamp(d['lockbox_start_ms']/1000,tz=timezone.utc)
 assert 182 <= (end_dt-lock_dt).days <= 185
 assert (end_dt.year-lock_dt.year)*12 + (end_dt.month-lock_dt.month) == 6
 assert set(d['late_start']) == {'KAS','RENDER','POL'}
 assert d['files']['KAS']['rows'] == 50021 and d['files']['RENDER']['rows'] == 37909 and d['files']['POL']['rows'] == 35553
 assert d['late_start']['KAS'] == d['files']['KAS']['start_ms']
 assert d['late_start']['RENDER'] == d['files']['RENDER']['start_ms']
 assert d['late_start']['POL'] == d['files']['POL']['start_ms']
 assert all('sha256' in d['files'][c] and d['files'][c]['sha256'] for c in COINS_28C)

def test_lockbox_is_after_validation():
 d=json.loads((ROOT/'research'/'fixtures'/'data_contract_28c.v1.json').read_text())
 s=split_indices(d['common']['bars'],d['common']['start_ms'],d['common']['end_ms'])
 assert s['train'][1] < s['validation'][0] <= s['validation'][1] <= s['lockbox'][0]
 assert s['lockbox'][0] < s['lockbox'][1]
 assert 'lockbox' not in search_splits(d['common']['bars'],d['common']['start_ms'],d['common']['end_ms'])

def test_common_alignment_and_no_gaps():
 d=json.loads((ROOT/'research'/'fixtures'/'data_contract_28c.v1.json').read_text())
 assert d['common']['bars']==35553
 assert (d['common']['end_ms']-d['common']['start_ms'])//STEP_MS+1 == d['common']['bars']
 assert all(x['gaps']==0 and x['bad_rows']==0 and x['sorted_unique'] for x in d['files'].values())


def test_trim_smoke_uses_loaded_coin_set(monkeypatch):
    monkeypatch.setenv("TRAIN_12F_SMOKE", "1")
    from research.train_12f_30m import trim_28c_to_train
    bars={c:[{"timestamp":i,"open":1,"high":1,"low":1,"close":1,"volume":1} for i in range(10)] for c in ("ETC","TRX")}
    out=trim_28c_to_train(bars)
    assert set(out)==set(bars)
    assert len(out["ETC"])==10
