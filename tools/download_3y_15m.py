"""Download 3y 15m perp klines from Binance futures (public, no key).

30 coins -> data/data_15m_3y/{COIN}.csv  (timestamp,open,high,low,close,volume,quote_volume,trades)
Resume-safe: skips coins whose CSV already covers full range.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, os, time, urllib.request

OUT = "data/data_15m_3y"
INTERVAL_MS = 15 * 60 * 1000
LIMIT = 1000  # max per request

# coin -> Binance perp symbol (1000-contracts for PEPE/SHIB; TON/OM absent on Binance)
SYMBOLS = {
 "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "SOL": "SOLUSDT",
 "BNB": "BNBUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT", "TRX": "TRXUSDT",
 "LINK": "LINKUSDT", "AVAX": "AVAXUSDT", "SUI": "SUIUSDT", "XLM": "XLMUSDT",
 "LTC": "LTCUSDT", "HBAR": "HBARUSDT", "DOT": "DOTUSDT", "BCH": "BCHUSDT",
 "UNI": "UNIUSDT", "NEAR": "NEARUSDT", "APT": "APTUSDT", "ICP": "ICPUSDT",
 "ETC": "ETCUSDT", "POL": "POLUSDT", "ARB": "ARBUSDT", "RENDER": "RENDERUSDT",
 "KAS": "KASUSDT", "ATOM": "ATOMUSDT",
 "SHIB": "1000SHIBUSDT", "PEPE": "1000PEPEUSDT",
}
MISSING = ["TON", "OM"]  # no Binance perp; use HL spot/Aster fallback if needed

def get_klines(symbol, start_ms):
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
           f"&interval=15m&limit={LIMIT}&startTime={start_ms}")
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaGPT/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def download_coin(coin, symbol, years=3):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{coin}.csv")
    now = int(time.time() * 1000)
    start = now - years * 365 * 24 * 3600 * 1000
    # resume: continue from last saved candle
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            last_ts = int(float(rows[-1][0]))
            if now - last_ts < 30 * 60 * 1000:
                print(f"{coin}: up-to-date ({len(rows)-1} rows), skip")
                return len(rows) - 1
            start = last_ts + INTERVAL_MS
            print(f"{coin}: resume from {time.ctime(start/1000)}")
    rows, n = [], 0
    cur = start
    while True:
        try:
            data = get_klines(symbol, cur)
        except Exception as e:
            print(f"{coin}: request failed at {time.ctime(cur/1000)}: {e}; retry in 5s")
            time.sleep(5)
            continue
        if not data:
            break
        rows.extend(data)
        n += len(data)
        cur = data[-1][0] + INTERVAL_MS
        if len(data) < LIMIT or cur >= now:
            break
        time.sleep(0.25)  # respect weight limits
        if n % 10000 == 0:
            print(f"  {coin}: {n} rows ...")
    # dedup + sort
    seen, out = set(), []
    for k in rows:
        if k[0] not in seen:
            seen.add(k[0])
            out.append(k)
    out.sort(key=lambda k: k[0])
    # merge with existing file if resuming
    if os.path.exists(path) and start > now - years * 365 * 24 * 3600 * 1000:
        with open(path) as f:
            old = list(csv.reader(f))
        header, old_rows = old[0], old[1:]
        have = {int(float(r[0])) for r in old_rows}
        merged = old_rows + [[str(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], str(k[8])]
                             for k in out if k[0] not in have]
        merged.sort(key=lambda r: int(float(r[0])))
        out_rows = merged
    else:
        header = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
        out_rows = [[str(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], str(k[8])] for k in out]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"{coin}: DONE {len(out_rows)} rows "
          f"({time.ctime(int(float(out_rows[0][0]))/1000)} -> {time.ctime(int(float(out_rows[-1][0]))/1000)})")
    return len(out_rows)

if __name__ == "__main__":
    import sys
    coins = sys.argv[1:] or list(SYMBOLS)
    total = {}
    for coin in coins:
        if coin not in SYMBOLS:
            print(f"{coin}: no Binance perp mapping, skip (missing: TON/OM)")
            continue
        total[coin] = download_coin(coin, SYMBOLS[coin])
    print(f"\nSummary: {sum(total.values())} candles across {len(total)} coins")
    print(f"Missing on Binance (need HL/Aster fallback): {MISSING}")
