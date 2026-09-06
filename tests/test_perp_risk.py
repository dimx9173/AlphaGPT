"""P4 Phase 4: perp risk gates (no chain)."""
import pytest

from strategy_manager.config import RiskConfig
from strategy_manager.risk import RiskEngine


def _eng(**kw):
    base = dict(perp_max_leverage=3, perp_max_notional_usdt=500,
                max_funding_rate=0.001)
    base.update(kw)
    rc = RiskConfig(
        daily_loss_pct=0.10, max_drawdown_pct=0.15,
        max_single_exposure_sol=1.0, blacklist=set(),
        velocity_max=100, velocity_window_sec=60, circuit_cooldown_sec=300,
        perp_max_leverage=base["perp_max_leverage"],
        perp_max_notional_usdt=base["perp_max_notional_usdt"],
        max_funding_rate=base["max_funding_rate"],
        venues_enabled={"solana", "hyperliquid", "aster"},
    )
    return RiskEngine(risk_config=rc, jupiter=None or __import__(
        "unittest.mock", fromlist=["AsyncMock"]).AsyncMock())


def test_check_perp_pass():
    eng = _eng()
    ok, reason = eng.check_perp("HYPE", 3, 400.0, 0.0005)
    assert ok and reason == ""


def test_check_perp_leverage_reject():
    eng = _eng()
    ok, reason = eng.check_perp("HYPE", 5, 100.0, 0.0)
    assert not ok and reason == "leverage"


def test_check_perp_notional_reject():
    eng = _eng()
    ok, reason = eng.check_perp("ASTERUSDT", 2, 600.0, 0.0)
    assert not ok and reason == "notional"


def test_check_perp_funding_reject():
    eng = _eng()
    ok, reason = eng.check_perp("BTC", 1, 100.0, 0.005)
    assert not ok and reason == "funding"


def test_check_perp_blacklist():
    eng = _eng()
    eng.risk_config = RiskConfig(
        daily_loss_pct=0.10, max_drawdown_pct=0.15,
        max_single_exposure_sol=1.0, blacklist={"BAD"},
        velocity_max=100, velocity_window_sec=60, circuit_cooldown_sec=300,
        perp_max_leverage=3, perp_max_notional_usdt=500,
        max_funding_rate=0.001, venues_enabled={"aster"},
    )
    ok, reason = eng.check_perp("BAD", 1, 10.0, 0.0)
    assert not ok and reason == "blacklist"


@pytest.mark.asyncio
async def test_check_safety_spot_unaffected_without_perp_params():
    from unittest.mock import AsyncMock

    eng = _eng()
    eng.jup = AsyncMock()
    eng.jup.get_quote = AsyncMock(return_value={"outAmount": "1000"})
    ok = await eng.check_safety("TokenA", 10_000.0)
    assert ok is True


@pytest.mark.asyncio
async def test_check_safety_perp_reject_wires():
    from unittest.mock import AsyncMock

    eng = _eng()
    eng.jup = AsyncMock()
    eng.jup.get_quote = AsyncMock(return_value={"outAmount": "1000"})
    ok = await eng.check_safety("HYPE", 10_000.0, leverage=10,
                                notional_usdt=100.0, funding_rate=0.0)
    assert ok is False


def test_risk_config_validate_ranges():
    import dataclasses

    rc = _eng().risk_config
    bad = dataclasses.replace(rc, perp_max_leverage=200)
    with pytest.raises(ValueError):
        bad.validate()
    bad2 = dataclasses.replace(rc, perp_max_notional_usdt=0)
    with pytest.raises(ValueError):
        bad2.validate()
