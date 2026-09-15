"""Download 1y klines 15m/30m/1h/4h from Binance futures (public, no key).

OUT: data/data_1y/{INTERVAL}/{COIN}.csv (timestamp,open,high,low,close,volume,quote_volume,trades)
Resume-safe per file. 29 coins x 4 intervals.
"""
import csv, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "data/data_1y"
INTERVALS = {"15m": 15*60*1000, "30m": 30*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000}
LIMIT = 1000
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
def get_klines(symbol, interval, start_ms):
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
           f"&interval={interval}&limit={LIMIT}&startTime={start_ms}")
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaGPT/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())
def download(coin, symbol, interval, ims, years=1):
    d = os.path.join(OUT, interval)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{coin}.csv")
    now = int(time.time() * 1000)
    start = now - years * 365 * 24 * 3600 * 1000
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            last_ts = int(float(rows[-1][0]))
            if now - last_ts < ims:
                return len(rows) - 1, "skip"
            start = last_ts + ims
    rows, cur = [], start
    while True:
        try:
            data = get_klines(symbol, interval, cur)
        except Exception as e:
            print(f"{coin}/{interval}: retry: {e}", flush=True)
            time.sleep(5)
            continue
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + ims
        if len(data) < LIMIT or cur >= now:
            break
        time.sleep(0.2)
    seen, out = set(), []
    for k in rows:
        if k[0] not in seen:
            seen.add(k[0])
            out.append(k)
    out.sort(key=lambda k: k[0])
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
    return len(out_rows), f"{len(out_rows)} rows"
if __name__ == "__main__":
    only = sys.argv[1:] or []
    ivs = [i for i in INTERVALS if (not only or i in only)]
    total, ndone = 0, 0
    for iv in ivs:
        for coin, sym in SYMBOLS.items():
            n, msg = download(coin, sym, iv, INTERVALS[iv])
            total += n
            ndone += 1
            print(f"[{ndone}/116] {coin}/{iv}: {msg}", flush=True)
    print(f"\nDone: {total} candles, {ndone} files -> {OUT}/{{15m,30m,1h,4h}}/<COIN>.csv")
