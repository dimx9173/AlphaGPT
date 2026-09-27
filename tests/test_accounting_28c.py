#!/usr/bin/env python3
"""
Unit tests for research/accounting_28c.py
"""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from research.accounting_28c import (
    scheduled_funding_rates,
    position_from_signal,
    accounting_bar_returns,
    account_portfolio,
    compound_equity,
    max_drawdown,
    daily_sharpe,
    metrics,
    ACCOUNTING_VERSION,
    FUND_RATE,
)

# Test timestamps: 30m bars starting at 2024-01-01 00:00 UTC
def make_timestamps(start_ms: int, n: int) -> np.ndarray:
    return np.array([start_ms + i * 1800000 for i in range(n)], dtype=np.int64)


class TestScheduledFundingRates:
    def test_funding_at_event_hours(self):
        # 00:00, 08:00, 16:00 UTC - need at least 33 bars for 16:00
        ts = make_timestamps(1704067200000, 40)  # 2024-01-01 00:00 UTC
        rates = scheduled_funding_rates(ts)
        # Index 0 is 00:00 -> funding
        assert rates[0] == FUND_RATE
        # Index 16 is 08:00 -> funding
        assert rates[16] == FUND_RATE
        # Index 32 is 16:00 -> funding
        assert rates[32] == FUND_RATE
    
    def test_no_funding_at_non_event_hours(self):
        ts = make_timestamps(1704067200000, 10)
        rates = scheduled_funding_rates(ts)
        # Index 1 is 00:30 -> no funding
        assert rates[1] == 0.0
        # Index 8 is 04:00 -> no funding
        assert rates[8] == 0.0
    
    def test_custom_fund_rate(self):
        ts = make_timestamps(1704067200000, 2)
        rates = scheduled_funding_rates(ts, fund_rate=0.001)
        assert rates[0] == 0.001


class TestPositionFromSignal:
    def test_long_only(self):
        signal = np.array([2.0, 2.0, 2.0, 2.0])
        ts = make_timestamps(1704067200000, 4)
        pos = position_from_signal(signal, ts, long_th=0.85, short_th=0.15)
        # First bar is flat (lag), then long
        assert pos[0] == 0.0
        assert pos[1] == 1.0
        assert pos[2] == 1.0
        assert pos[3] == 1.0
    
    def test_short_only(self):
        signal = np.array([-2.0, -2.0, -2.0, -2.0])
        ts = make_timestamps(1704067200000, 4)
        pos = position_from_signal(signal, ts, long_th=0.85, short_th=0.15)
        assert pos[0] == 0.0
        assert pos[1] == -1.0
        assert pos[2] == -1.0
        assert pos[3] == -1.0
    
    def test_cooldown(self):
        # Signal: long, wants short, wants short, wants short, wants long
        signal = np.array([2.0, -2.0, -2.0, -2.0, 2.0, 2.0])
        ts = make_timestamps(1704067200000, 6)
        pos = position_from_signal(signal, ts, long_th=0.85, short_th=0.15, cooldown_bars=2)
        # t=0: flat (lag)
        # t=1: signal[0]=2 -> long, lock=2
        # t=2: signal[1]=-2 wants short, lock=2->1, stays long
        # t=3: signal[2]=-2 wants short, lock=1->0, stays long
        # t=4: signal[3]=-2 wants short, lock=0, flips to short
        # t=5: signal[4]=2 wants long, lock=2, stays short
        assert pos[0] == 0.0
        assert pos[1] == 1.0
        assert pos[2] == 1.0
        assert pos[3] == 1.0
        assert pos[4] == -1.0
        assert pos[5] == -1.0


class TestAccountingBarReturns:
    def test_basic_long(self):
        pos = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        ret = np.array([0.0, 0.01, 0.01, 0.01, 0.01])
        fee = 0.0004
        fund = np.zeros(5)
        lev = 2.0
        net = accounting_bar_returns(pos, ret, fee, fund, lev)
        # t=0: flat, net=0
        # t=1: turnover=1, gross=1*0.01*2=0.02, fee=1*0.0004*2=0.0008, net=0.0192
        assert net[0] == 0.0
        assert abs(net[1] - 0.0192) < 1e-10
        assert abs(net[2] - 0.02) < 1e-10  # no turnover after first
    
    def test_turnover_fee(self):
        pos = np.array([0.0, 1.0, -1.0, 1.0])
        ret = np.array([0.0, 0.0, 0.0, 0.0])
        fee = 0.0004
        fund = np.zeros(4)
        lev = 2.0
        net = accounting_bar_returns(pos, ret, fee, fund, lev)
        # t=1: turnover=1, fee=0.0008, net=-0.0008
        # t=2: turnover=2, fee=0.0016, net=-0.0016
        # t=3: turnover=2, fee=0.0016, net=-0.0016
        assert abs(net[1] + 0.0008) < 1e-10
        assert abs(net[2] + 0.0016) < 1e-10
        assert abs(net[3] + 0.0016) < 1e-10
    
    def test_funding_long_pays(self):
        pos = np.array([0.0, 1.0, 1.0])
        ret = np.zeros(3)
        fee = 0.0
        fund = np.array([0.0, 0.0005, 0.0])
        lev = 2.0
        net = accounting_bar_returns(pos, ret, fee, fund, lev)
        # t=1: long pays funding = 1 * 0.0005 * 2 = 0.001
        assert abs(net[1] + 0.001) < 1e-10
    
    def test_funding_short_receives(self):
        pos = np.array([0.0, -1.0, -1.0])
        ret = np.zeros(3)
        fee = 0.0
        fund = np.array([0.0, 0.0005, 0.0])
        lev = 2.0
        net = accounting_bar_returns(pos, ret, fee, fund, lev)
        # t=1: short receives funding = -1 * 0.0005 * 2 = -0.001 (cost negative = receipt)
        assert abs(net[1] - 0.001) < 1e-10  # net is positive = receipt


class TestAccountPortfolio:
    def test_single_leg_matches_single(self):
        n = 5
        pos = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        w = np.ones((n, 1))
        notionals = w * pos[:, None]
        ret = np.array([[0.0, 0.01, 0.01, 0.01, 0.01]]).T
        fee = 0.0004
        fund = np.zeros(n)
        lev = 2.0
        net = account_portfolio(notionals, ret, fee, fund, lev)
        # Should match single-leg
        single_net = accounting_bar_returns(pos, ret.ravel(), fee, fund, lev)
        assert np.allclose(net, single_net)
    
    def test_weight_turnover(self):
        # Two legs, weight shifts from leg 0 to leg 1
        n = 3
        pos = np.ones((n, 2))
        w = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        notionals = w * pos
        ret = np.ones((n, 2)) * 0.01
        fee = 0.0004
        fund = np.zeros(n)
        lev = 2.0
        net = account_portfolio(notionals, ret, fee, fund, lev)
        # t=0: flat
        # t=1: notionals [[0.5, 0.5]], turnover = |0.5| + |0.5| = 1.0
        # gross = 2 * (0.5*0.01 + 0.5*0.01) = 0.02
        # fee = 1.0 * 0.0004 * 2 = 0.0008
        # net = 0.02 - 0.0008 = 0.0192
        assert abs(net[1] - 0.0192) < 1e-10


class TestCompoundEquity:
    def test_compounding(self):
        net = np.array([0.10, -0.20, 0.05])
        equity, solvent = compound_equity(net)
        # equity: 1.0, 1.10, 0.88, 0.924
        assert equity[0] == 1.0
        assert abs(equity[1] - 1.10) < 1e-10
        assert abs(equity[2] - 0.88) < 1e-10
        assert abs(equity[3] - 0.924) < 1e-10
        assert solvent is True
    
    def test_insolvency(self):
        net = np.array([0.0, -1.5])  # -150% return
        equity, solvent = compound_equity(net)
        assert solvent is False
        assert equity[2] == 0.0
        # equity length is n+1 = 3, index 3 is out of bounds


class TestMaxDrawdown:
    def test_basic(self):
        equity = np.array([1.0, 1.10, 0.88, 0.924])
        mdd = max_drawdown(equity)
        # Peak 1.10, trough 0.88, drawdown = (1.10 - 0.88) / 1.10 = 0.2
        assert abs(mdd - 0.2) < 1e-10
    
    def test_no_drawdown(self):
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        mdd = max_drawdown(equity)
        assert mdd == 0.0


class TestDailySharpe:
    def test_utc_day_grouping(self):
        # 48 bars = 1 day at 30m
        net = np.ones(48) * 0.001
        ts = make_timestamps(1704067200000, 48)  # 2024-01-01
        sharpe, count = daily_sharpe(net, ts)
        assert count == 1
        assert sharpe == 0.0  # zero variance -> 0 sharpe
    
    def test_multiple_days(self):
        net = np.array([0.01] * 48 + [-0.01] * 48)
        ts = make_timestamps(1704067200000, 96)
        sharpe, count = daily_sharpe(net, ts)
        assert count == 2


class TestMetrics:
    def test_full_metrics(self):
        net = np.array([0.10, -0.20, 0.05])
        ts = make_timestamps(1704067200000, 3)
        m = metrics(net, ts)
        assert m["accounting_version"] == ACCOUNTING_VERSION
        assert abs(m["final_x"] - 0.924) < 1e-10
        assert abs(m["total_return"] + 0.076) < 1e-10
        assert abs(m["mdd"] - 0.2) < 1e-10
        assert m["solvent"] is True
        assert m["daily_return_count"] == 1


class TestValidation:
    def test_length_mismatch_raises(self):
        pos = np.array([0.0, 1.0])
        ret = np.array([0.0])
        fund = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            accounting_bar_returns(pos, ret, 0.0004, fund)
    
    def test_non_finite_raises(self):
        pos = np.array([0.0, np.nan])
        ret = np.array([0.0, 0.0])
        fund = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            accounting_bar_returns(pos, ret, 0.0004, fund)
    
    def test_account_portfolio_shape_mismatch(self):
        notionals = np.zeros((3, 2))
        returns = np.zeros((3, 3))
        with pytest.raises(ValueError):
            account_portfolio(notionals, returns, 0.0004, np.zeros(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
