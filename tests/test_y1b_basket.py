"""Y1b basket wiring test: locked params + offline signals, no orders."""
from strategy_manager.config import FORMULA, LOCKED_ETC, LOCKED_TRX, LEV, CHALLENGER_AA_H1_CD2
from strategy_manager.y1b_basket import latest_signals

def test_y1b_locked():
    assert FORMULA == [3,2,7,2,7,11,15,4,4,6,6,10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"]) == (0.88, 0.12, 18)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"]) == (0.85, 0.12, 6)
    assert LEV == 2.0

def test_y1b_signals_discrete():
    s = latest_signals()
    assert set(s) == {"ETC", "TRX"}
    assert all(v in (-1.0, 0.0, 1.0) for v in s.values())

def test_runner_y1b_once_shadow(monkeypatch, tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    from strategy_manager import runner as R
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.delenv("Y1B_LIVE_ENABLED", raising=False)
    r = R.StrategyRunner.__new__(R.StrategyRunner)
    import strategy_manager.risk as _rk
    r.risk = _rk.RiskEngine()
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=20.0)
    b.get_position = AsyncMock(return_value=None)
    b.market_open = AsyncMock(side_effect=AssertionError("no orders in shadow"))
    r.brokers = {"aster": b}
    plans, res, sync = asyncio.run(r.run_y1b_once(notional=50.0))
    assert {p.coin for p in plans} == {"ETC", "TRX"}
    assert all(x.get("dry_run") for x in res)
    b.market_open.assert_not_awaited()

def test_e13_guard_pinned():
    assert CHALLENGER_AA_H1_CD2 == dict(sth=0.10, etc_cd=17, trx_cd=11, vt=0.012, vw=12, ts=24, q=0.3)

def test_y1b_paper_mode_refusal():
    import asyncio
    from strategy_manager import runner as R
    r = R.StrategyRunner.__new__(R.StrategyRunner)
    r.paper_mode = True
    asyncio.run(r.run_loop())  # must return immediately, no orders
