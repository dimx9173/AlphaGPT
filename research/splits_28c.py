from __future__ import annotations
from .data_contract_28c import EMBARGO_BARS, COMMON_28_BARS, lockbox_start_ms

def split_indices(n:int, common_start_ms:int, common_end_ms:int, embargo_bars:int=EMBARGO_BARS)->dict:
    if n != COMMON_28_BARS: raise ValueError(f"expected common bars {COMMON_28_BARS}, got {n}")
    train_end=int(n*0.60); lock_start=lockbox_start_ms(common_end_ms)
    # Convert timestamp to common index using the 30m grid.
    lock_idx=max(0, min(n, (lock_start-common_start_ms)//(30*60*1000)))
    train_end=max(embargo_bars, min(lock_idx-embargo_bars, train_end))
    return {"train":(0,train_end-embargo_bars),"validation":(train_end+embargo_bars,lock_idx-embargo_bars),"lockbox":(lock_idx,n),"embargo_bars":embargo_bars,"lockbox_start_ms":lock_start}

def search_splits(n:int, common_start_ms:int, common_end_ms:int, embargo_bars:int=EMBARGO_BARS):
    x=split_indices(n,common_start_ms,common_end_ms,embargo_bars)
    return {k:x[k] for k in ("train","validation")}
