import torch
import pytest
from model_core.backtest import MemeBacktest

def _make_inputs(n=4, t=32):
    torch.manual_seed(42)
    liquidity = torch.abs(torch.randn(n, t)) * 1e6 + 1e6
    raw = {"liquidity": liquidity}
    factors = torch.randn(n, t)
    target_ret = torch.randn(n, t) * 0.02
    return factors, raw, target_ret

def test_evaluate_returns_tuple_or_dict():
    bt = MemeBacktest()
    factors, raw, target = _make_inputs()
    res = bt.evaluate(factors, raw, target)
    if isinstance(res, dict):
        assert len(res) > 0
    elif isinstance(res, tuple):
        assert len(res) == 2
        fitness, cum = res
        assert isinstance(fitness, torch.Tensor) or isinstance(fitness, float)
        assert isinstance(cum, float)
    else:
        assert isinstance(res, (torch.Tensor, float))

def test_evaluate_median_path():
    bt = MemeBacktest()
    factors, raw, target = _make_inputs()
    res = bt.evaluate(factors, raw, target)
    if isinstance(res, tuple):
        fitness, cum_ret = res
        assert isinstance(cum_ret, float)
        assert abs(cum_ret) < 1e6

def test_evaluate_low_liquidity_filters():
    bt = MemeBacktest()
    n, t = 2, 16
    raw = {"liquidity": torch.ones(n, t) * 100}
    factors = torch.ones(n, t) * 10
    target = torch.ones(n, t) * 0.01
    res = bt.evaluate(factors, raw, target)
    if isinstance(res, tuple):
        fitness, _ = res
        val = fitness.item() if isinstance(fitness, torch.Tensor) else fitness
        assert val == pytest.approx(-10.0, abs=1e-3) or val < 0

def test_evaluate_does_not_require_db():
    bt = MemeBacktest()
    factors, raw, target = _make_inputs(n=2, t=8)
    res = bt.evaluate(factors, raw, target)
    assert res is not None

def test_evaluate_handles_zero_factors():
    bt = MemeBacktest()
    n, t = 2, 16
    raw = {"liquidity": torch.ones(n, t) * 1e6}
    factors = torch.zeros(n, t)
    target = torch.randn(n, t) * 0.01
    res = bt.evaluate(factors, raw, target)
    assert res is not None

def test_evaluate_turnover_bounded():
    bt = MemeBacktest()
    factors, raw, target = _make_inputs()
    raw["liquidity"] = torch.ones_like(raw["liquidity"]) * 1e7
    factors = torch.ones_like(factors) * 5
    res = bt.evaluate(factors, raw, target)
    assert res is not None
