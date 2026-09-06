import asyncio
from hyperliquid.info import Info
import time

async def main():
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    now = int(time.time()*1000)
    
    # Test 1: 1 month ago
    start_1m = now - 30*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_1m, now)
    print(f"1m range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 2: 3 months ago
    start_3m = now - 90*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_3m, now)
    print(f"3m range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 3: 6 months ago
    start_6m = now - 180*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_6m, now)
    print(f"6m range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 4: 1 year ago
    start_1y = now - 365*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_1y, now)
    print(f"1y range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 5: 2 years ago
    start_2y = now - 2*365*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_2y, now)
    print(f"2y range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 6: 3 years ago
    start_3y = now - 3*365*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_3y, now)
    print(f"3y range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 7: 5 years ago
    start_5y = now - 5*365*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_5y, now)
    print(f"5y range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
    
    # Test 8: 10 years ago
    start_10y = now - 10*365*24*3600*1000
    c = await asyncio.to_thread(info.candles_snapshot, "HYPE", "15m", start_10y, now)
    print(f"10y range: count={len(c)}, hours={(c[-1]['t']-c[0]['t'])/(3600*1000):.1f}")
asyncio.run(main())
