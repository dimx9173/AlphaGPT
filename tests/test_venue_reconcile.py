"""P4 Phase 4: venue-aware portfolio reconcile (no chain)."""
import json

from strategy_manager.portfolio import PortfolioManager


def test_add_venue_positions_isolated(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("HYPE", "HYPE", 25.0, 10.0, 0.0, tx_sig="sig1",
                    venue="hyperliquid")
    pm.add_position("HYPE", "HYPE", 25.0, 5.0, 0.0, tx_sig="sig2",
                    venue="aster")
    assert pm.get_open_count() == 2
    assert "hyperliquid::HYPE" in pm.positions
    assert "aster::HYPE" in pm.positions


def test_replay_same_sig_same_venue_no_double(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("HYPE", "HYPE", 25.0, 10.0, 0.0, tx_sig="sigX",
                    venue="hyperliquid")
    pm.add_position("HYPE", "HYPE", 99.0, 99.0, 0.0, tx_sig="sigX",
                    venue="hyperliquid")
    assert pm.get_open_count() == 1
    assert pm.positions["hyperliquid::HYPE"].amount_held == 10.0


def test_reconcile_per_venue(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("HYPE", "HYPE", 25.0, 10.0, 0.0, venue="hyperliquid")
    pm.add_position("HYPE", "HYPE", 25.0, 5.0, 0.0, venue="aster")
    assert pm.reconcile("HYPE", 4.0, venue="hyperliquid") is True
    assert pm.positions["hyperliquid::HYPE"].amount_held == 4.0
    assert pm.positions["aster::HYPE"].amount_held == 5.0
    assert pm.reconcile("HYPE", 0.0, venue="aster") is True
    assert "aster::HYPE" not in pm.positions


def test_legacy_solana_path_unchanged(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokenA", "TKA", 1.0, 100.0, 2.0, tx_sig="s1")
    assert "TokenA" in pm.positions
    assert pm.reconcile("TokenA", 50.0) is True
    assert pm.positions["TokenA"].amount_held == 50.0


def test_old_state_file_loads_with_defaults(tmp_state_file):
    with open(tmp_state_file, "w") as f:
        json.dump({"TokenA": {
            "token_address": "TokenA", "symbol": "TKA", "entry_price": 1.0,
            "entry_time": 1.0, "amount_held": 10.0, "initial_cost_sol": 2.0,
            "highest_price": 1.0, "is_moonbag": False}}, f)
    pm = PortfolioManager(state_file=tmp_state_file)
    assert pm.positions["TokenA"].venue == "solana"
    assert pm.positions["TokenA"].leverage == 1.0
