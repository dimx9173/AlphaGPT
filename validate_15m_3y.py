"""Validate data_15m_3y/*.csv: row counts, continuity, gaps, OHLC sanity."""
import csv, os, json
from datetime import datetime, timezone

D = "data_15m_3y"
STEP = 15 * 60 * 1000
report = {}
for fn in sorted(os.listdir(D)):
    if not fn.endswith(".csv"):
        continue
    coin = fn[:-4]
    path = os.path.join(D, fn)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    ts = [int(float(r["timestamp"])) for r in rows]
    o = [float(r["open"]) for r in rows]
    h = [float(r["high"]) for r in rows]
    l = [float(r["low"]) for r in rows]
    c = [float(r["close"]) for r in rows]
    v = [float(r["volume"]) for r in rows]
    gaps, dups = 0, 0
    seen = set()
    for i in range(1, len(ts)):
        d = ts[i] - ts[i-1]
        if d == STEP:
            continue
        elif d == 0:
            dups += 1
        elif d > STEP:
            gaps += d // STEP - 1
        else:
            gaps += 1  # out of order / overlap
    bad_ohlc = sum(1 for i in range(n) if not (h[i] >= max(o[i], c[i], l[i]) and l[i] <= min(o[i], c[i], h[i])))
    bad_px = sum(1 for x in o+h+l+c if not (x > 0))
    neg_vol = sum(1 for x in v if x < 0)
    report[coin] = {
        "rows": n,
        "first": datetime.fromtimestamp(ts[0]/1000, timezone.utc).strftime("%Y-%m-%d"),
        "last": datetime.fromtimestamp(ts[-1]/1000, timezone.utc).strftime("%Y-%m-%d"),
        "span_days": round((ts[-1]-ts[0])/86400000, 1),
        "missing_steps": gaps,
        "dup_steps": dups,
        "bad_ohlc": bad_ohlc,
        "bad_px": bad_px,
        "neg_vol": neg_vol,
    }
    print(f"{coin:6s} rows={n:7d} {report[coin]['first']}->{report[coin]['last']} "
          f"({report[coin]['span_days']}d) gaps={gaps} dups={dups} bad_ohlc={bad_ohlc} bad_px={bad_px}")
with open("validate_15m_3y.json", "w") as f:
    json.dump(report, f, indent=2)
tot = sum(r["rows"] for r in report.values())
print(f"\nTOTAL {tot} candles across {len(report)} coins -> validate_15m_3y.json")
