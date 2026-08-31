from unittest.mock import AsyncMock
import pytest
from strategy_manager.portfolio import PortfolioManager


@pytest.mark.asyncio
async def test_idempotent_add_position(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokA", "SYM", 1.0, 100, 1.0, tx_sig="sig1")
    assert pm.has_sig("sig1") is True
    pm.add_position("TokA", "SYM", 2.0, 200, 2.0, tx_sig="sig1")
    assert pm.positions["TokA"].amount_held == 100
    assert pm.positions["TokA"].entry_price == 1.0


@pytest.mark.asyncio
async def test_reconcile_corrects_amount(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokA", "SYM", 1.0, 100, 1.0)
    pm.reconcile("TokA", 80)
    assert pm.positions["TokA"].amount_held == 80


@pytest.mark.asyncio
async def test_reconcile_closes_on_zero(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokA", "SYM", 1.0, 100, 1.0)
    pm.reconcile("TokA", 0)
    assert "TokA" not in pm.positions


@pytest.mark.asyncio
async def test_rpc_simulate_blocks_send(tmp_state_file):
    from execution.rpc_handler import QuickNodeClient

    class FakeClient:
        async def simulate_transaction(self, txn):
            class R:
                value = type("V", (), {"err": "err"})()
            return R()
        async def send_transaction(self, *a, **kw):
            raise AssertionError("should not send when simulate fails")
        async def get_latest_blockhash(self):
            return None
        async def confirm_transaction(self, *a, **kw):
            return None
        async def get_signature_statuses(self, sigs):
            return None
        async def close(self):
            return None

    rpc = QuickNodeClient(client=FakeClient())
    res = await rpc.send_and_confirm(object())
    assert res is None
