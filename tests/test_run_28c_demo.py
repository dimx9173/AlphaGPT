import asyncio
from pathlib import Path
import pytest

from research import run_28c_demo


def test_demo_runner_defaults_read_only():
    assert run_28c_demo.MAX_LEVERAGE == 2
    assert len(run_28c_demo.COINS_28C) == 28
    assert "SHIB" not in run_28c_demo.COINS_28C
    assert "AAVE" in run_28c_demo.COINS_28C
    assert run_28c_demo.OFFLINE_VENUE_SYMBOLS["PEPE"] == "1000PEPEUSDT"
    assert run_28c_demo.OFFLINE_VENUE_SYMBOLS["AAVE"] == "AAVEUSDT"


def test_arm_gate_is_hard_closed():
    with pytest.raises(RuntimeError):
        run_28c_demo.arm_gate()


def test_kline_parser_accepts_bybit_v5_lists_and_drops_open_bar(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    rows = []
    for i in range(210):
        ts = now_ms - (210 - i) * run_28c_demo.BAR_MS
        rows.append([str(ts), "1.0", "1.1", "0.9", "1.0", "100", "1000"])
    open_bar = int(__import__("time").time() * 1000) - run_28c_demo.BAR_MS // 2
    rows.append([str(open_bar), "1", "1.1", "0.9", "1.0", "50", "500"])
    data, dropped = run_28c_demo.parse_klines(rows)
    assert dropped == 1
    assert len(data["timestamp"]) >= 200
    assert data["timestamp"][-1] == now_ms - run_28c_demo.BAR_MS


def test_kline_parser_rejects_insufficient_history():
    now_ms = int(__import__("time").time() * 1000)
    rows = []
    for i in range(100):
        ts = now_ms - (100 - i) * run_28c_demo.BAR_MS
        rows.append([str(ts), "1.0", "1.1", "0.9", "1.0", "100", "1000"])
    open_bar = int(__import__("time").time() * 1000) - run_28c_demo.BAR_MS // 2
    rows.append([str(open_bar), "1", "1.1", "0.9", "1.0", "50", "500"])
    with pytest.raises(ValueError, match="insufficient closed klines: require >=200"):
        run_28c_demo.parse_klines(rows)


def test_kline_parser_rejects_insufficient_history():
    # Only 100 closed bars + 1 open = 101 total, should raise
    now_ms = int(__import__("time").time() * 1000)
    rows = []
    for i in range(100):
        ts = now_ms - (100 - i) * run_28c_demo.BAR_MS
        rows.append([str(ts), "1.0", "1.1", "0.9", "1.0", "100", "1000"])
    open_bar = int(__import__("time").time() * 1000) - run_28c_demo.BAR_MS // 2
    rows.append([str(int(__import__("time").time() * 1000) - run_28c_demo.BAR_MS // 2), "1", "1.1", "0.9", "1.0", "50", "500"])
    with pytest.raises(ValueError, match="insufficient closed klines: require >=200"):
        run_28c_demo.parse_klines(rows)


def test_demo_runner_source_has_no_order_sink():
    src=(Path(__file__).resolve().parents[1] / "research" / "run_28c_demo.py").read_text()
    assert "await broker.market_open" not in src
    assert "await broker.limit_open" not in src
    assert "await broker.set_leverage" not in src
    assert "orders_attempted" in src
