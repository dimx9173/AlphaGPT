import time
from unittest.mock import AsyncMock

import pytest

from strategy_manager.config import RiskConfig
from strategy_manager.risk import RiskEngine


def _eng(**overrides):
    rc = RiskConfig(
        daily_loss_pct=overrides.pop("daily_loss_pct", 0.10),
        max_drawdown_pct=overrides.pop("max_drawdown_pct", 0.15),
        max_single_exposure_sol=overrides.pop("max_single_exposure_sol", 1.0),
        blacklist=overrides.pop("blacklist", set()),
        velocity_max=overrides.pop("velocity_max", 3),
        velocity_window_sec=overrides.pop("velocity_window_sec", 60),
        circuit_cooldown_sec=overrides.pop("circuit_cooldown_sec", 60),
    )
    jup = AsyncMock()
    jup.get_quote = AsyncMock(return_value={"outAmount": "1000"})
    eng = RiskEngine(risk_config=rc, jupiter=jup)
    for k, v in overrides.items():
        setattr(eng, k, v)
    return eng


@pytest.mark.asyncio
async def test_daily_loss_circuit_blocks():
    eng = _eng(daily_loss_pct=0.10)
    assert await eng.check_safety("TokenA", 100000, daily_pnl=-0.11) is False
    blocked, reason = eng.check_circuit(daily_pnl=-0.11)
    assert reason in ("daily_loss", "circuit_cooldown")


@pytest.mark.asyncio
async def test_drawdown_circuit_blocks():
    eng = _eng(max_drawdown_pct=0.15)
    assert await eng.check_safety("TokenA", 100000, drawdown=0.20) is False


@pytest.mark.asyncio
async def test_exposure_cap_blocks():
    eng = _eng(max_single_exposure_sol=1.0)
    assert await eng.check_safety("TokenA", 100000, exposure_sol=1.5) is False


@pytest.mark.asyncio
async def test_blacklist_blocks():
    eng = _eng(blacklist={"BadToken"})
    assert await eng.check_safety("BadToken", 100000) is False
    assert await eng.check_safety("GoodToken", 100000) is True


@pytest.mark.asyncio
async def test_velocity_blocks():
    eng = _eng(velocity_max=3, velocity_window_sec=60)
    now = 1000.0
    assert eng.check_velocity("Tok", now=now) is True
    assert eng.check_velocity("Tok", now=now + 1) is True
    assert eng.check_velocity("Tok", now=now + 2) is True
    assert eng.check_velocity("Tok", now=now + 3) is False


@pytest.mark.asyncio
async def test_circuit_cooldown_blocks_subsequent_calls():
    eng = _eng(daily_loss_pct=0.10, circuit_cooldown_sec=300)
    assert await eng.check_safety("TokenA", 100000, daily_pnl=-0.20) is False
    assert eng._is_circuit_open() is True
    assert await eng.check_safety("TokenA", 100000) is False


def test_risk_config_validate():
    rc = RiskConfig(daily_loss_pct=0.10)
    rc.validate()
    bad = RiskConfig(daily_loss_pct=1.5)
    with pytest.raises(ValueError):
        bad.validate()


def test_position_size_capped_by_exposure():
    eng = _eng(max_single_exposure_sol=0.5)
    assert eng.calculate_position_size(10.0) == 0.5
