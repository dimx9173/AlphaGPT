import urllib.request, json, time
def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())

# which pairs exist for the problem coins?
coins = ["BTC","ETH","XRP","SOL","BNB","DOGE","ADA","TRX","LINK","AVAX","SUI","XLM","TON","SHIB","LTC","HBAR","DOT","BCH","UNI","OM","NEAR","APT","ICP","ETC","POL","PEPE","ARB","RENDER","KAS","ATOM"]
info = get("https://fapi.binance.com/fapi/v1/exchangeInfo")
syms = {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}
for c in coins:
    cands = [f"{c}USDT", f"{c}USDC", f"1000{c}USDT"]
    hit = next((x for x in cands if x in syms), None)
    print(f"{c}: {hit or 'MISSING'}")
# PEPE/SHIB start dates
for sym in ["PEPEUSDT", "1000PEPEUSDT", "SHIBUSDT", "1000SHIBUSDT"]:
    if sym in syms:
        d = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=15m&limit=1&startTime=0")
        print(f"{sym} first listing: {time.ctime(d[0][0]/1000)}")
