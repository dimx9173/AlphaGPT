from __future__ import annotations
from .data_contract_28c import EMBARGO_BARS, COMMON_28_BARS, lockbox_start_ms

# The bar grid is a parameter, not a constant baked into the arithmetic. The
# 30m set is 35553 bars and the 1h set is 8760, and the bar count is checked
# against COMMON_28_BARS below, so both the grid and the expected count have to
# move together or a 1h run raises instead of splitting.
STEP_MS = 30 * 60 * 1000
EXPECTED_BARS = COMMON_28_BARS

def configure(step_ms:int=30*60*1000, expected_bars:int=COMMON_28_BARS,
              embargo_bars:int=EMBARGO_BARS)->None:
    """Point the splitter at a bar grid. Called once from main() after --bars."""
    global STEP_MS, EXPECTED_BARS, EMBARGO_BARS
    STEP_MS = step_ms
    EXPECTED_BARS = expected_bars
    EMBARGO_BARS = embargo_bars


def split_indices(n:int, common_start_ms:int, common_end_ms:int, embargo_bars:int=EMBARGO_BARS)->dict:
    if n != EXPECTED_BARS: raise ValueError(f"expected common bars {EXPECTED_BARS}, got {n}")
    train_end=int(n*0.60); lock_start=lockbox_start_ms(common_end_ms)
    # Convert timestamp to common index using the configured bar grid.
    lock_idx=max(0, min(n, (lock_start-common_start_ms)//STEP_MS))
    train_end=max(embargo_bars, min(lock_idx-embargo_bars, train_end))
    return {"train":(0,train_end-embargo_bars),"validation":(train_end+embargo_bars,lock_idx-embargo_bars),"lockbox":(lock_idx,n),"embargo_bars":embargo_bars,"lockbox_start_ms":lock_start}

def search_splits(n:int, common_start_ms:int, common_end_ms:int, embargo_bars:int=EMBARGO_BARS):
    x=split_indices(n,common_start_ms,common_end_ms,embargo_bars)
    return {k:x[k] for k in ("train","validation")}
