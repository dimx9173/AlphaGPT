import asyncio
import pytest
import aiohttp

from execution.jupiter import JupiterAggregator
from strategy_manager.runner import StrategyRunner
from unittest.mock import AsyncMock, MagicMock


class FakeResp:
    def __init__(self, status=200, json_data=None, text_data="", headers=None):
        self.status = status
        self._json = json_data if json_data is not None else {}
        self._text = text_data
        self.headers = headers or {}

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    def __init__(self, responses=None, raises=None):
        self.get_calls = []
        self.post_calls = []
        self._responses = list(responses or [])
        self._raises = list(raises or [])

    def _next(self, calls, method):
        calls.append(1)
        if self._raises:
            exc = self._raises.pop(0)
            if exc is not None:
                raise exc
        if self._responses:
            return self._responses.pop(0)
        return FakeResp(status=200, json_data={"outAmount": "1000000000"})

    def get(self, *a, **kw):
        return self._next(self.get_calls, "get")

    def post(self, *a, **kw):
        return self._next(self.post_calls, "post")

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_get_quote_retry_429_then_success(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=3, circuit_threshold=10)
    jup.session = FakeSession(responses=[
        FakeResp(status=429, text_data="rate limited", headers={"Retry-After": "0"}),
        FakeResp(status=200, json_data={"outAmount": "123"}),
    ])
    out = await jup.get_quote("IN", "OUT", 1000)
    assert out is not None and out["outAmount"] == "123"
    assert len(jup.session.get_calls) == 2
    assert jup._consecutive_fails == 0


@pytest.mark.asyncio
async def test_get_quote_retry_timeout_then_success(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=3, circuit_threshold=10)
    jup.session = FakeSession(
        responses=[FakeResp(status=200, json_data={"outAmount": "999"})],
        raises=[asyncio.TimeoutError("timeout"), None],
    )
    out = await jup.get_quote("IN", "OUT", 1000)
    assert out["outAmount"] == "999"
    assert len(jup.session.get_calls) == 2


@pytest.mark.asyncio
async def test_get_quote_retry_5xx_then_success(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=3, circuit_threshold=10)
    jup.session = FakeSession(responses=[
        FakeResp(status=500, text_data="oops"),
        FakeResp(status=200, json_data={"outAmount": "42"}),
    ])
    out = await jup.get_quote("IN", "OUT", 1000)
    assert out["outAmount"] == "42"


@pytest.mark.asyncio
async def test_get_quote_timeout_kwarg_forwarded(monkeypatch):
    captured = {}

    class CapSession:
        def get(self, *a, **kw):
            captured.update(kw)
            return FakeResp(status=200, json_data={"outAmount": "1"})
        async def close(self): pass

    jup = JupiterAggregator(max_retries=1, circuit_threshold=10, timeout=aiohttp.ClientTimeout(total=5))
    jup.session = CapSession()
    await jup.get_quote("IN", "OUT", 1)
    assert "timeout" in captured
    assert captured["timeout"].total == 5


@pytest.mark.asyncio
async def test_fallback_cache_when_jupiter_fails(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=1, circuit_threshold=10)
    jup.session = FakeSession(responses=[FakeResp(status=500, text_data="down")])
    key = jup._cache_key("IN", "OUT", 1000)
    jup._last_quote_cache[key] = {"outAmount": "777"}
    monkeypatch.setattr("data_pipeline.config.Config.BIRDEYE_API_KEY", "", raising=False)
    out = await jup.get_quote("IN", "OUT", 1000)
    assert out["outAmount"] == "777"


@pytest.mark.asyncio
async def test_circuit_breaker_short_circuits(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=1, circuit_threshold=2, circuit_cooldown=60)
    jup.session = FakeSession(responses=[
        FakeResp(status=500, text_data="err"),
        FakeResp(status=500, text_data="err"),
        FakeResp(status=200, json_data={"outAmount": "should-not-hit"}),
    ])
    monkeypatch.setattr("data_pipeline.config.Config.BIRDEYE_API_KEY", "", raising=False)
    assert await jup.get_quote("IN", "OUT", 1) is None
    assert await jup.get_quote("IN", "OUT", 2) is None
    assert jup._is_circuit_open()
    calls_before = len(jup.session.get_calls)
    out = await jup.get_quote("IN", "OUT", 1)
    assert out is None
    assert len(jup.session.get_calls) == calls_before


@pytest.mark.asyncio
async def test_circuit_breaker_uses_cache_when_open(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    jup = JupiterAggregator(max_retries=1, circuit_threshold=1, circuit_cooldown=60)
    jup.session = FakeSession(responses=[FakeResp(status=500, text_data="err")])
    monkeypatch.setattr("data_pipeline.config.Config.BIRDEYE_API_KEY", "", raising=False)
    assert await jup.get_quote("IN", "OUT", 9) is None
    assert jup._is_circuit_open()
    key = jup._cache_key("IN", "OUT", 9)
    jup._last_quote_cache[key] = {"outAmount": "555"}
    out = await jup.get_quote("IN", "OUT", 9)
    assert out["outAmount"] == "555"


@pytest.mark.asyncio
async def test_get_swap_tx_retry_and_timeout(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    monkeypatch.setattr("execution.config.ExecutionConfig.get_wallet_address", lambda: "FakeAddr")
    jup = JupiterAggregator(max_retries=3, circuit_threshold=10)
    jup.session = FakeSession(responses=[
        FakeResp(status=429, text_data="rate", headers={}),
        FakeResp(status=200, json_data={"swapTransaction": "B64TX"}),
    ])
    out = await jup.get_swap_tx({"x": 1})
    assert out == "B64TX"


@pytest.mark.asyncio
async def test_runner_fetch_live_price_sol_injected_jupiter(monkeypatch):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.trader = MagicMock()
    runner.trader.config.SOL_MINT = "So11111111111111111111111111111111111111112"
    runner.trader.rpc = MagicMock()
    fake_client = MagicMock()
    runner.trader.rpc.client = fake_client
    monkeypatch.setattr("strategy_manager.runner.get_mint_decimals", AsyncMock(return_value=6))

    class InjJup:
        _last_quote_cache = {}
        async def get_quote(self, **kw):
            return {"outAmount": "2000000000"}

    price = await StrategyRunner._fetch_live_price_sol(runner, "TOKEN", jupiter=InjJup())
    assert price == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_runner_fetch_raises_instead_of_zero(monkeypatch):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.trader = MagicMock()
    runner.trader.config.SOL_MINT = "So11111111111111111111111111111111111111112"
    runner.trader.rpc = MagicMock()
    runner.trader.rpc.client = MagicMock()
    runner.trader.jup = MagicMock()
    runner.trader.jup._last_quote_cache = {}
    monkeypatch.setattr("strategy_manager.runner.get_mint_decimals", AsyncMock(return_value=6))
    monkeypatch.setattr("data_pipeline.config.Config.BIRDEYE_API_KEY", "", raising=False)

    class FailJup:
        _last_quote_cache = {}
        async def get_quote(self, **kw):
            return None

    with pytest.raises(RuntimeError, match="Price unavailable"):
        await StrategyRunner._fetch_live_price_sol(runner, "TOKEN", jupiter=FailJup())


@pytest.mark.asyncio
async def test_runner_fetch_fallback_chain_jupiter_none_then_birdeye(monkeypatch):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.trader = MagicMock()
    runner.trader.config.SOL_MINT = "So11111111111111111111111111111111111111112"
    runner.trader.rpc = MagicMock()
    runner.trader.rpc.client = MagicMock()
    monkeypatch.setattr("strategy_manager.runner.get_mint_decimals", AsyncMock(return_value=6))
    monkeypatch.setattr("data_pipeline.config.Config.BIRDEYE_API_KEY", "KEY123", raising=False)

    class FailJup:
        _last_quote_cache = {}
        async def get_quote(self, **kw):
            return None

    class BirdeyeResp:
        status = 200
        async def json(self): return {"data": {"value": 1.5}}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class BirdeyeSess:
        def __init__(self, *a, **kw): pass
        def get(self, *a, **kw): return BirdeyeResp()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def close(self): pass

    import aiohttp as _aiohttp
    orig = _aiohttp.ClientSession
    monkeypatch.setattr(_aiohttp, "ClientSession", BirdeyeSess)

    price = await StrategyRunner._fetch_live_price_sol(runner, "TOKEN", jupiter=FailJup())
    assert price == pytest.approx(1.5)
    monkeypatch.setattr(_aiohttp, "ClientSession", orig)
