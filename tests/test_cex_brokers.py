"""P4 Phase 5: Binance / Bybit / OKX broker unit tests (no network, no keys)."""
import pytest

from execution.brokers.base import Side, Venue
from execution.brokers.binance import BinanceBroker
from execution.brokers.bybit import BybitBroker
from execution.brokers.okx import OkxBroker, _to_inst
from execution.brokers import make_venue_broker
from execution.brokers.cex_config import BinanceConfig, BybitConfig, OkxConfig


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status = payload, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._p

    async def text(self):
        return str(self._p)


class _FakeSession:
    """Maps (METHOD, url-substring) -> payload. Records calls."""

    def __init__(self, routes):
        self._routes = routes  # list of (method, substr, payload, status)
        self.calls = []

    def _match(self, method, url):
        for m, sub, payload, status in self._routes:
            if m == method and sub in url:
                return payload, status
        return {}, 404

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, params))
        p, s = self._match("GET", url)
        return _FakeResp(p, s)

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, data))
        p, s = self._match("POST", url)
        return _FakeResp(p, s)

    def request(self, method, url, data=None, headers=None, timeout=None):
        self.calls.append((method, url, data))
        p, s = self._match(method, url)
        return _FakeResp(p, s)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None))
        p, s = self._match("DELETE", url)
        return _FakeResp(p, s)


def test_venue_enum_has_cex():
    assert {v.value for v in Venue} >= {"binance", "bybit", "okx", "aster", "solana"}


def test_factory_builds_cex():
    for name, cls in [("binance", BinanceBroker), ("bybit", BybitBroker), ("okx", OkxBroker)]:
        b = make_venue_broker(name)
        assert isinstance(b, cls)
    try:
        make_venue_broker("nope")
        assert False, "must raise"
    except ValueError:
        pass


def test_okx_symbol_map():
    assert _to_inst("ETCUSDT") == "ETC-USDT-SWAP"
    assert _to_inst("TRXUSDT") == "TRX-USDT-SWAP"
    assert _to_inst("ETC-USDT-SWAP") == "ETC-USDT-SWAP"


# ---- Binance ----
@pytest.mark.asyncio
async def test_binance_reads():
    routes = [
        ("GET", "/fapi/v1/ticker/price", {"symbol": "ETCUSDT", "price": "20.5"}, 200),
        ("GET", "/fapi/v1/klines", [["t", "o", "h", "l", "20.5", "1"]], 200),
        ("GET", "/fapi/v1/premiumIndex", {"symbol": "ETCUSDT", "lastFundingRate": "0.0001"}, 200),
        ("GET", "/fapi/v2/balance", [{"asset": "USDT", "crossWalletBalance": "123.45"}], 200),
        ("GET", "/fapi/v2/positionRisk", [{"symbol": "ETCUSDT", "positionAmt": "-2.5", "entryPrice": "20.0", "leverage": "2"}], 200),
    ]
    b = BinanceBroker(api_key="k", api_secret="s", session=_FakeSession(routes))
    assert await b.get_price("ETCUSDT") == pytest.approx(20.5)
    assert await b.get_klines("ETCUSDT")
    assert await b.get_funding_rate("ETCUSDT") == pytest.approx(0.0001)
    assert await b.get_balance() == pytest.approx(123.45)
    pos = await b.get_position("ETCUSDT")
    assert pos is not None and pos.side == "SHORT" and pos.size == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_binance_writes_and_deadman():
    routes = [
        ("POST", "/fapi/v1/order", {"orderId": 111, "avgPrice": "20.4", "executedQty": "1.0"}, 200),
        ("DELETE", "/fapi/v1/order", {}, 200),
        ("POST", "/fapi/v1/leverage", {"leverage": 2}, 200),
        ("POST", "/fapi/v1/marginType", {"code": 200}, 200),
        ("POST", "/fapi/v1/countdownCancelAll", {}, 200),
    ]
    b = BinanceBroker(api_key="k", api_secret="s", session=_FakeSession(routes))
    b.set_deadman_symbols(["ETCUSDT"])
    m = await b.market_open("ETCUSDT", Side.BUY, 1.0)
    assert m.ok and m.oid == "111"
    lim = await b.limit_open("ETCUSDT", Side.SELL, 1.0, 21.0)
    assert lim.ok
    assert await b.cancel("ETCUSDT", "111") is True
    assert await b.set_leverage("ETCUSDT", 2) is True
    assert await b.enable_deadman(60) is True


@pytest.mark.asyncio
async def test_binance_missing_keys_fail_closed():
    b = BinanceBroker(api_key="", api_secret="", session=_FakeSession([]))
    # inject empty config keys
    r = await b.market_open("ETCUSDT", Side.BUY, 1.0)
    assert not r.ok
    assert await b.enable_deadman(60) is False
    assert await b.get_price("ETCUSDT") == 0.0  # public path 404 -> soft 0


@pytest.mark.asyncio
async def test_binance_deadman_needs_symbol():
    b = BinanceBroker(api_key="k", api_secret="s", session=_FakeSession([]))
    assert await b.enable_deadman(60) is False


# ---- Bybit ----
@pytest.mark.asyncio
async def test_bybit_reads():
    routes = [
        ("GET", "/v5/market/tickers", {"retCode": 0, "result": {"list": [{"symbol": "ETCUSDT", "lastPrice": "20.5"}]}}, 200),
        ("GET", "/v5/market/kline", {"retCode": 0, "result": {"list": [["t", "o", "h", "l", "c", "v", "x"]]}}, 200),
        ("GET", "/v5/market/funding/history", {"retCode": 0, "result": {"list": [{"fundingRate": "0.0002"}]}}, 200),
        ("GET", "/v5/account/wallet-balance", {"retCode": 0, "result": {"list": [{"totalWalletBalance": "500.0"}]}}, 200),
        ("GET", "/v5/position/list", {"retCode": 0, "result": {"list": [{"symbol": "ETCUSDT", "side": "Sell", "size": "1.5", "avgPrice": "20.0", "leverage": "2"}]}}, 200),
    ]
    b = BybitBroker(api_key="k", api_secret="s", session=_FakeSession(routes))
    assert await b.get_price("ETCUSDT") == pytest.approx(20.5)
    assert await b.get_klines("ETCUSDT")
    assert await b.get_funding_rate("ETCUSDT") == pytest.approx(0.0002)
    assert await b.get_balance() == pytest.approx(500.0)
    pos = await b.get_position("ETCUSDT")
    assert pos is not None and pos.side == "SHORT" and pos.size == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_bybit_writes():
    routes = [
        ("POST", "/v5/order/create", {"retCode": 0, "result": {"orderId": "abc123"}}, 200),
        ("POST", "/v5/order/cancel", {"retCode": 0, "result": {}}, 200),
        ("POST", "/v5/position/set-leverage", {"retCode": 0, "result": {}}, 200),
        ("POST", "/v5/order/cancel-all", {"retCode": 0, "result": {}}, 200),
    ]
    b = BybitBroker(api_key="k", api_secret="s", session=_FakeSession(routes))
    m = await b.market_open("ETCUSDT", Side.BUY, 1.0)
    assert m.ok and m.oid == "abc123"
    assert (await b.limit_open("ETCUSDT", Side.SELL, 1.0, 21.0)).ok
    assert await b.cancel("ETCUSDT", "abc123") is True
    assert await b.set_leverage("ETCUSDT", 2) is True
    assert await b.enable_deadman(60) is True


@pytest.mark.asyncio
async def test_bybit_missing_keys_fail_closed():
    b = BybitBroker(api_key="", api_secret="", session=_FakeSession([]))
    # force config empty
    import execution.brokers.cex_config as cc
    _k, _s = cc.BybitConfig.api_key, cc.BybitConfig.api_secret
    cc.BybitConfig.api_key = classmethod(lambda cls: "")
    cc.BybitConfig.api_secret = classmethod(lambda cls: "")
    try:
        r = await b.market_open("ETCUSDT", Side.BUY, 1.0)
        assert not r.ok
        assert await b.set_leverage("ETCUSDT", 2) is False
    finally:
        cc.BybitConfig.api_key = _k
        cc.BybitConfig.api_secret = _s


# ---- OKX ----
@pytest.mark.asyncio
async def test_okx_reads():
    routes = [
        ("GET", "/api/v5/market/ticker", {"code": "0", "data": [{"instId": "ETC-USDT-SWAP", "lastPx": "20.5"}]}, 200),
        ("GET", "/api/v5/market/candles", {"code": "0", "data": [["t", "o", "h", "l", "c", "v", "x", "y", "z"]]}, 200),
        ("GET", "/api/v5/public/funding-rate", {"code": "0", "data": [{"fundingRate": "0.0003"}]}, 200),
        ("GET", "/api/v5/account/balance", {"code": "0", "data": [{"details": [{"ccy": "USDT", "availBal": "300.0"}]}]}, 200),
        ("GET", "/api/v5/account/positions", {"code": "0", "data": [{"instId": "ETC-USDT-SWAP", "posSide": "short", "pos": "2", "avgPx": "20.0", "lever": "2"}]}, 200),
    ]
    b = OkxBroker(api_key="k", api_secret="s", passphrase="p", session=_FakeSession(routes))
    assert await b.get_price("ETCUSDT") == pytest.approx(20.5)
    assert await b.get_klines("ETCUSDT")
    assert await b.get_funding_rate("ETCUSDT") == pytest.approx(0.0003)
    assert await b.get_balance() == pytest.approx(300.0)
    pos = await b.get_position("ETCUSDT")
    assert pos is not None and pos.side == "SHORT" and pos.size == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_okx_writes():
    routes = [
        ("POST", "/api/v5/trade/order", {"code": "0", "data": [{"ordId": "okx1"}]}, 200),
        ("POST", "/api/v5/trade/cancel-order", {"code": "0", "data": [{}]}, 200),
        ("POST", "/api/v5/account/set-leverage", {"code": "0", "data": [{}]}, 200),
        ("POST", "/api/v5/trade/cancel-algos", {"code": "0", "data": [{}]}, 200),
    ]
    b = OkxBroker(api_key="k", api_secret="s", passphrase="p", session=_FakeSession(routes))
    m = await b.market_open("ETCUSDT", Side.BUY, 1.0)
    assert m.ok and m.oid == "okx1"
    assert (await b.limit_open("ETCUSDT", Side.SELL, 1.0, 21.0)).ok
    assert await b.cancel("ETCUSDT", "okx1") is True
    assert await b.set_leverage("ETCUSDT", 2) is True
    assert await b.enable_deadman(60) is True


@pytest.mark.asyncio
async def test_okx_missing_keys_fail_closed():
    b = OkxBroker(api_key="", api_secret="", passphrase="", session=_FakeSession([]))
    import execution.brokers.cex_config as cc
    _k, _s, _p = cc.OkxConfig.api_key, cc.OkxConfig.api_secret, cc.OkxConfig.passphrase
    cc.OkxConfig.api_key = classmethod(lambda cls: "")
    cc.OkxConfig.api_secret = classmethod(lambda cls: "")
    cc.OkxConfig.passphrase = classmethod(lambda cls: "")
    try:
        r = await b.market_open("ETCUSDT", Side.BUY, 1.0)
        assert not r.ok
        assert await b.enable_deadman(60) is False
    finally:
        cc.OkxConfig.api_key = _k
        cc.OkxConfig.api_secret = _s
        cc.OkxConfig.passphrase = _p


def test_cex_configs_validate_only_when_enabled(monkeypatch):
    monkeypatch.setenv("VENUES_ENABLED", "aster")
    BinanceConfig.validate_env()
    BybitConfig.validate_env()
    OkxConfig.validate_env()
    monkeypatch.setenv("VENUES_ENABLED", "binance")
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_API_SECRET", "")
    try:
        BinanceConfig.validate_env()
        assert False, "must raise"
    except ValueError:
        pass
