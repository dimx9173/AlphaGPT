from pathlib import Path
import csv
import numpy as np

from research.causal_12f import causal_signal
from research.universe_28c import COINS_28C, MANIFEST_28C

ROOT = Path(__file__).resolve().parents[1]
FORMULA = [5, 5, 6, 26, 20, 17, 22, 13, 16, 14, 21, 21]


def _load(path, limit=2000):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))[-limit:]
    return {
        "timestamp": [int(r["timestamp"]) for r in rows],
        "open": [float(r["open"]) for r in rows],
        "high": [float(r["high"]) for r in rows],
        "low": [float(r["low"]) for r in rows],
        "close": [float(r["close"]) for r in rows],
        "volume": [float(r["volume"]) for r in rows],
        "quote_volume": [float(r["quote_volume"]) for r in rows],
    }


def test_manifest_has_exact_28_and_data_files():
    assert len(COINS_28C) == 28
    assert set(MANIFEST_28C) == set(COINS_28C)
    for coin in COINS_28C:
        assert (ROOT / "data" / "data_3y" / "30m" / f"{coin}.csv").exists()


def test_timestamp_intersection_handles_phase_shifted_exports():
    timestamps = []
    for coin in COINS_28C:
        data = _load(ROOT / "data" / "data_3y" / "30m" / f"{coin}.csv", limit=17520)
        timestamps.append(set(data["timestamp"]))
    common = set.intersection(*timestamps)
    assert len(common) == 17520


def test_causal_signal_prefix_invariance():
    data = _load(ROOT / "data" / "data_3y" / "30m" / "ETC.csv", limit=2000)
    full = causal_signal(FORMULA, data)
    for cut in (100, 500, 1200, 1999):
        prefix = {k: v[:cut] for k, v in data.items()}
        got = causal_signal(FORMULA, prefix)
        assert np.allclose(got, full[:cut], atol=1e-10, rtol=0)


def test_future_shock_does_not_change_prefix():
    data = _load(ROOT / "data" / "data_3y" / "30m" / "BTC.csv", limit=2000)
    cut = 1200
    before = causal_signal(FORMULA, data)[:cut]
    shocked = {k: list(v) for k, v in data.items()}
    for key in ("open", "high", "low", "close", "volume", "quote_volume"):
        shocked[key][cut:] = [value * 1000.0 + 12345.0 for value in shocked[key][cut:]]
    after = causal_signal(FORMULA, shocked)[:cut]
    assert np.allclose(before, after, atol=1e-10, rtol=0)


def test_paper_runner_is_broker_free():
    source = (ROOT / "research" / "run_paper_28c_pit_30m.py").read_text()
    assert "execution.brokers" not in source
    assert "aiohttp" not in source
    assert "dotenv" not in source
    assert "market_open" not in source
    assert "Y1B_LIVE_ENABLED" not in source
