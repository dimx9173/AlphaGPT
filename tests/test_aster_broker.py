"""P4 Phase 3: Aster V3 signing + broker unit tests (no network)."""
import urllib.parse

import pytest

from execution.brokers.aster import AsterBroker, NEED_DEPOSIT
from execution.brokers.aster_sign import (
    DOMAIN_CHAIN_ID,
    AsterSigner,
    build_typed_data,
    micro_nonce,
)
from execution.brokers.base import Side


class TestSigner:
    def test_nonce_increases(self):
        n1, n2 = micro_nonce(), micro_nonce()
        assert int(n2) >= int(n1)
        assert len(n1) >= 16  # ms * 1e6 digits

    def test_typed_data_domain(self):
        td = build_typed_data("symbol=ASTERUSDT")
        assert td["domain"]["chainId"] == DOMAIN_CHAIN_ID == 1666
        assert td["domain"]["name"] == "AsterSignTransaction"
        assert td["primaryType"] == "Message"

    def test_sign_fields_shape(self):
        s = AsterSigner("0xuser", "0xsigner", "0x" + "11" * 32)
        out = s.sign_fields({"symbol": "ASTERUSDT"})
        assert out["signer"] == "0xsigner" and "nonce" in out
        sig = out["signature"]
        assert isinstance(sig, str) and len(sig) == 130  # 65 bytes hex
        qs = urllib.parse.urlencode({"symbol": "ASTERUSDT", "signer": "0xsigner",
                                     "nonce": out["nonce"]})
        assert len(qs) > 10

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            AsterSigner("", "0xsigner", "0x" + "11" * 32)


# ---- broker with fake session ----
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
    def __init__(self, handler):
        self._h = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return _FakeResp(self._h("GET", url, params, None))

    def request(self, method, url, data=None, headers=None, timeout=None):
        self.calls.append((method, url, data))
        return _FakeResp(self._h(method, url, None, data))

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None))
        return _FakeResp(self._h("DELETE", url, None, None))


def _signer():
    return AsterSigner("0xuser", "0xsigner", "0x" + "11" * 32)


@pytest.mark.asyncio
async def test_get_price_and_klines():
    def h(method, url, params, data):
        if url.endswith("/fapi/v3/ticker/price"):
            return {"symbol": "ASTERUSDT", "price": "0.74650"}
        if url.endswith("/fapi/v3/klines"):
            return [[1700000000000, "0.7", "0.8", "0.6", "0.75", "1000"]]
        return {}

    b = AsterBroker(signer=_signer(), session=_FakeSession(h))
    assert await b.get_price("ASTERUSDT") == pytest.approx(0.7465)
    kl = await b.get_klines("ASTERUSDT")
    assert kl and kl[0][4] == "0.75"


@pytest.mark.asyncio
async def test_get_position_and_balance():
    def h(method, url, params, data):
        if url.endswith("/fapi/v3/positionRisk"):
            return [{"symbol": "ASTERUSDT", "positionAmt": "-20",
                     "entryPrice": "0.75", "leverage": "3"}]
        if url.endswith("/fapi/v3/accountWithJoinMargin"):
            return {"totalWalletBalance": "1500.5"}
        return {}

    b = AsterBroker(signer=_signer(), session=_FakeSession(h))
    pos = await b.get_position("ASTERUSDT")
    assert pos is not None and pos.side == "SHORT" and pos.size == pytest.approx(20.0)
    assert await b.get_balance() == pytest.approx(1500.5)


@pytest.mark.asyncio
async def test_market_limit_cancel_leverage():
    def h(method, url, params, data):
        if url.endswith("/fapi/v3/order") and method == "POST":
            return {"orderId": 999, "avgPrice": "0.74", "executedQty": "20"}
        if url.endswith("/fapi/v3/order") and method == "DELETE":
            return {"code": 200}
        if url.endswith("/fapi/v3/leverage"):
            return {"leverage": 3}
        if url.endswith("/fapi/v3/marginType"):
            return {"code": 200}
        return {}

    b = AsterBroker(signer=_signer(), session=_FakeSession(h))
    m = await b.market_open("ASTERUSDT", Side.BUY, 20.0)
    assert m.ok and m.oid == "999" and m.fill_price == pytest.approx(0.74)
    lim = await b.limit_open("ASTERUSDT", Side.SELL, 5.0, 0.8)
    assert lim.ok
    assert await b.cancel("ASTERUSDT", "999") is True
    assert await b.set_leverage("ASTERUSDT", 3) is True


@pytest.mark.asyncio
async def test_need_deposit_maps_error():
    def h(method, url, params, data):
        return {"code": NEED_DEPOSIT, "msg": "deposit first"}

    b = AsterBroker(signer=_signer(), session=_FakeSession(h))
    res = await b.market_open("ASTERUSDT", Side.BUY, 1.0)
    assert not res.ok and "deposit" in res.reason.lower()


@pytest.mark.asyncio
async def test_missing_signer_maps_error():
    b = AsterBroker(signer=None, session=_FakeSession(lambda *a: {}))
    res = await b.market_open("ASTERUSDT", Side.BUY, 1.0)
    assert not res.ok
