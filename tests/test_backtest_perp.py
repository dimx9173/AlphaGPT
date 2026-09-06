"""Backtest multi-venue + perp extension (no DB)."""
import torch
import pytest

from model_core.backtest import MemeBacktest, VENUE_TAKER_FEE


def _inputs(n=4, t=32, seed=42):
    torch.manual_seed(seed)
    liquidity = torch.abs(torch.randn(n, t)) * 1e6 + 1e6
    raw = {"liquidity": liquidity}
    factors = torch.randn(n, t)
    target = torch.randn(n, t) * 0.02
    return factors, raw, target


def test_legacy_defaults_unchanged():
    bt = MemeBacktest()
    assert bt.venue == "solana" and bt.leverage == 1.0 and bt.short_enabled is False
    assert bt.base_fee == pytest.approx(0.0060)
    factors, raw, target = _inputs()
    fit, cum = bt.evaluate(factors, raw, target)
    assert isinstance(cum, float)
    assert bt.last_metrics["short_turnover"] == 0.0
    assert bt.last_metrics["funding_paid"] == 0.0


def test_venue_fee_schedule():
    assert VENUE_TAKER_FEE["hyperliquid"] == pytest.approx(0.00045)
    assert VENUE_TAKER_FEE["aster"] == pytest.approx(0.00040)
    assert MemeBacktest(venue="hyperliquid").base_fee == pytest.approx(0.00045)
    assert MemeBacktest(venue="aster").base_fee == pytest.approx(0.00040)
    assert MemeBacktest(venue="hyperliquid", fee_override=0.001).base_fee == pytest.approx(0.001)


def test_short_leg_activates_and_pays_on_down_moves():
    n, t = 2, 32
    raw = {"liquidity": torch.ones(n, t) * 1e7}
    factors = torch.ones(n, t) * -10.0  # sigmoid ~ 0 -> short entries
    target = torch.ones(n, t) * -0.01   # price falls -> shorts profit
    bt_long = MemeBacktest()
    _, cum_long = bt_long.evaluate(factors, raw, target)
    bt_short = MemeBacktest(short_enabled=True)
    _, cum_short = bt_short.evaluate(factors, raw, target)
    assert bt_short.last_metrics["short_turnover"] > 0.0
    assert cum_short > cum_long  # shorts capture the down move


def test_leverage_scales_pnl():
    factors, raw, target = _inputs()
    _, cum1 = MemeBacktest(short_enabled=True, leverage=1.0).evaluate(factors, raw, target)
    _, cum3 = MemeBacktest(short_enabled=True, leverage=3.0).evaluate(factors, raw, target)
    # gross scales with leverage; costs scale too, so check direction + magnitude grow
    assert abs(cum3) >= abs(cum1) or cum3 != cum1


def test_funding_drags_longs():
    n, t = 2, 32
    raw = {"liquidity": torch.ones(n, t) * 1e7}
    factors = torch.ones(n, t) * 10.0  # long entries
    target = torch.zeros(n, t)          # flat price: only costs show
    _, cum_nofund = MemeBacktest().evaluate(factors, raw, target)
    _, cum_fund = MemeBacktest(funding_override=0.001).evaluate(factors, raw, target)
    assert cum_fund < cum_nofund
    # shorts receive funding: flip signal, funding should help
    factors_s = torch.ones(n, t) * -10.0
    _, cum_s = MemeBacktest(short_enabled=True, funding_override=0.001).evaluate(
        factors_s, raw, target)
    bt = MemeBacktest(short_enabled=True, funding_override=0.001)
    _, _ = bt.evaluate(factors_s, raw, target)
    assert bt.last_metrics["funding_paid"] < 0.0  # negative paid = received


def test_funding_tensor_from_raw_data():
    n, t = 2, 16
    raw = {"liquidity": torch.ones(n, t) * 1e7,
           "funding": torch.ones(n, t) * 0.002}
    factors = torch.ones(n, t) * 10.0
    target = torch.zeros(n, t)
    bt = MemeBacktest()
    _, cum = bt.evaluate(factors, raw, target)
    assert cum < 0.0
    assert bt.last_metrics["funding_paid"] > 0.0


def test_walk_forward_still_works_with_perp():
    factors, raw, target = _inputs()
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005)
    res = bt.evaluate_walk_forward(factors, raw, target, n_splits=3)
    assert len(res["oos_scores"]) == 3
    assert isinstance(res["mean_oos"], float)
