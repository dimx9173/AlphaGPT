"""Y1b once-CLI tests: refusal without confirm flag, dry-run by default."""
import json
import subprocess
import sys


def _run(*args):
    r = subprocess.run([sys.executable, "research/run_y1b_once.py", *args],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads(r.stdout)


def test_live_without_confirm_refused():
    d = _run("--live")
    assert d.get("refused") is True


def test_bad_notional_rejected_no_orders():
    import asyncio
    from unittest.mock import AsyncMock
    from strategy_manager.y1b_executor import build_plans
    from strategy_manager.risk import RiskEngine
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=20.0)
    b.market_open = AsyncMock(side_effect=AssertionError("bad notional must not order"))
    for bad in (-100.0, 0.0):
        plans = asyncio.run(build_plans(b, RiskEngine(), notional=bad))
        assert plans and all(p.gate_ok is False and p.size == 0.0 for p in plans)
    b.market_open.assert_not_awaited()

def test_oversize_notional_capped():
    import asyncio
    from unittest.mock import AsyncMock
    from strategy_manager.y1b_executor import build_plans
    from strategy_manager.risk import RiskEngine
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=1.0)
    plans = asyncio.run(build_plans(b, RiskEngine(), notional=99999.0))
    for p in plans:
        if p.want != 0:
            assert p.size * p.price <= 500.0 + 1e-6

def test_default_is_dry_run():
    d = _run()
    assert all(x.get("dry_run") for x in d["results"])
    assert {p["coin"] for p in d["plans"]} == {"ETC", "TRX"}


def test_print_config_no_compute():
    import json
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "research/run_y1b_once.py", "--print-config"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.loads(r.stdout)
    assert d["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["lev"] == 2.0
    assert d["symbols"] == {"ETC": "ETCUSDT", "TRX": "TRXUSDT"}
    assert d["dry_run_default"] is True
