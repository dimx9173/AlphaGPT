"""Versioned data-window contract for the 28-coin Binance research sets.

30m is the contract every recorded batch was produced under. 1h and 4h are
backfilled to the same three-year span so that a bar-size comparison changes the
sampling and not the window; they are new profiles, not a revision of this one,
and an artifact must record which profile produced it.
"""
from __future__ import annotations
from datetime import datetime, timezone
STEP_MS=30*60*1000

# Bar-size profiles. `bars` is the count each profile produces after the 28-coin
# common-timestamp intersection, which is what the splitter asserts against; a
# gap anywhere makes that number smaller and the run raises rather than silently
# splitting on a hole.
BAR_PROFILES = {
    "30m": {"step_ms": 30 * 60 * 1000, "bpy": 17520.0, "dir": "30m"},
    "1h":  {"step_ms": 60 * 60 * 1000, "bpy": 8760.0, "dir": "1h"},
    "4h":  {"step_ms": 4 * 60 * 60 * 1000, "bpy": 2190.0, "dir": "4h"},
}
COINS_28C=["ADA","APT","ARB","ATOM","AVAX","BCH","BNB","BTC","DOGE","DOT","ETC","ETH","HBAR","ICP","KAS","LINK","LTC","NEAR","PEPE","POL","RENDER","AAVE","SOL","SUI","TRX","UNI","XLM","XRP"]
FULL_3Y_25=[c for c in COINS_28C if c not in {"KAS","POL","RENDER"}]
LATE_START_COINS={"KAS":1700186400000,"RENDER":1721988000000,"POL":1726228800000}
COMMON_28_BARS=35553
HISTORY_YEARS_FULL_3Y=3.0
HISTORY_YEARS_COMMON_28=2.03
LOCKBOX_MONTHS=6
EMBARGO_BARS=200

def _minus_months(dt:datetime, months:int)->datetime:
    m=dt.month-months
    y=dt.year+(m-1)//12; m=(m-1)%12+1
    day=min(dt.day,[31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return dt.replace(year=y,month=m,day=day)

def lockbox_start_ms(end_ms:int)->int:
    return int(_minus_months(datetime.fromtimestamp(end_ms/1000,tz=timezone.utc),LOCKBOX_MONTHS).timestamp()*1000)

def profile_metadata(common_start_ms:int,common_end_ms:int)->dict:
    return {"common_28":{"coins":COINS_28C,"bars":COMMON_28_BARS,"start_ms":common_start_ms,"end_ms":common_end_ms,"history_years":HISTORY_YEARS_COMMON_28},"full_3y_25":{"coins":FULL_3Y_25,"bars":52560,"history_years":HISTORY_YEARS_FULL_3Y},"lockbox":{"months":LOCKBOX_MONTHS,"start_ms":lockbox_start_ms(common_end_ms),"end_ms":common_end_ms},"embargo_bars":EMBARGO_BARS}
