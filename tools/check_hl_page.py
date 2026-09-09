import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio, time
from hyperliquid.info import Info

DAY = 24*3600*1000

async def main():
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    now = int(time.time()*1000)
    c1 = await asyncio.to_thread(info.candles_snapshot, "BTC", "15m", 0, now)
    print(f"batch1 count={len(c1)} first={time.ctime(c1[0]['t']/1000)}")
    end2 = c1[0]['t'] - 1
    start2 = end2 - 90*DAY
    c2 = await asyncio.to_thread(info.candles_snapshot, "BTC", "15m", start2, end2)
    print(f"batch2 count={len(c2)} first={time.ctime(c2[0]['t']/1000) if c2 else 'EMPTY'} last={time.ctime(c2[-1]['t']/1000) if c2 else 'EMPTY'}")
    print(f"contiguous: {bool(c2) and c2[-1]['t'] < c1[0]['t']}")
    coins = ["BTC","ETH","XRP","SOL","BNB","DOGE","ADA","TRX","LINK","AVAX","SUI","XLM","TON","SHIB","LTC","HBAR","DOT","BCH","UNI","OM","NEAR","APT","ICP","ETC","POL","PEPE","ARB","RENDER","KAS","ATOM"]
    meta = info.meta()
    uni = meta.get("universe", []) if isinstance(meta, dict) else []
    names = {u["name"] for u in uni if isinstance(u, dict)}
    missing = [c for c in coins if c not in names]
    print(f"HL universe={len(names)} missing={missing}")
    print(f"HYPE in universe: {'HYPE' in names}")

asyncio.run(main())
