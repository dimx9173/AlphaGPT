import pytest
from unittest.mock import AsyncMock
from strategy_manager.risk import RiskEngine

@pytest.mark.asyncio
async def test_check_safety_low_liquidity_returns_false():
    eng = RiskEngine()
    assert await eng.check_safety("So11111111111111111111111111111111111111112", 100) is False
    assert await eng.check_safety("token", 4999) is False
    await eng.close()

@pytest.mark.asyncio
async def test_check_safety_success_with_mocked_jupiter():
    eng = RiskEngine()
    eng.jup.get_quote = AsyncMock(return_value={"outAmount": "1000"})
    assert await eng.check_safety("TokenA", 100000) is True
    await eng.close()

@pytest.mark.asyncio
async def test_check_safety_no_quote_returns_false():
    eng = RiskEngine()
    eng.jup.get_quote = AsyncMock(return_value=None)
    assert await eng.check_safety("TokenA", 100000) is False
    await eng.close()

@pytest.mark.asyncio
async def test_check_safety_exception_returns_false():
    eng = RiskEngine()
    eng.jup.get_quote = AsyncMock(side_effect=Exception("rpc down"))
    assert await eng.check_safety("TokenA", 100000) is False
    await eng.close()

def test_calculate_position_size_normal():
    eng = RiskEngine()
    assert eng.calculate_position_size(10.0) == eng.config.ENTRY_AMOUNT_SOL

def test_calculate_position_size_insufficient_balance():
    eng = RiskEngine()
    assert eng.calculate_position_size(0.05) == 0.0
    assert eng.calculate_position_size(1.0) == 0.0

def test_calculate_position_size_exact_threshold():
    eng = RiskEngine()
    size = eng.config.ENTRY_AMOUNT_SOL
    assert eng.calculate_position_size(size + 0.1) == size
    assert eng.calculate_position_size(size + 0.09) == 0.0

def test_calculate_position_size_zero():
    eng = RiskEngine()
    assert eng.calculate_position_size(0) == 0.0
