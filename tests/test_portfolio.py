import json
import threading
import time
from strategy_manager.portfolio import PortfolioManager

def test_atomic_write_and_load_roundtrip(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokenA", "TKA", 1.23, 100.0, 2.0)
    pm2 = PortfolioManager(state_file=tmp_state_file)
    assert "TokenA" in pm2.positions
    assert pm2.positions["TokenA"].symbol == "TKA"
    assert pm2.positions["TokenA"].entry_price == 1.23

def test_save_and_load_multiple_positions(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("Tok1", "TK1", 1.0, 10, 1.0)
    pm.add_position("Tok2", "TK2", 2.0, 20, 1.0)
    assert pm.get_open_count() == 2
    pm3 = PortfolioManager(state_file=tmp_state_file)
    assert pm3.get_open_count() == 2

def test_close_and_update(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("TokX", "TKX", 5.0, 50, 1.0)
    pm.update_price("TokX", 10.0)
    assert pm.positions["TokX"].highest_price == 10.0
    pm.update_holding("TokX", 0)
    assert "TokX" not in pm.positions
    pm.add_position("TokY", "TKY", 1.0, 10, 1.0)
    pm.close_position("TokY")
    assert "TokY" not in pm.positions

def test_concurrent_writes_produce_valid_json(tmp_state_file):
    errors = []
    def worker(idx):
        try:
            for i in range(5):
                token = f"T{idx}_{i}_{time.time_ns()}"
                pm2 = PortfolioManager(state_file=tmp_state_file)
                pm2.add_position(token, f"S{idx}", float(idx), float(i), 1.0)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"errors: {errors}"
    with open(tmp_state_file) as f:
        data = json.load(f)
    assert isinstance(data, dict)

def test_load_save_roundtrip_preserves_fields(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    pm.add_position("AAA", "AAA", 9.99, 123.45, 3.14)
    raw = json.load(open(tmp_state_file))
    assert raw["AAA"]["highest_price"] == 9.99
    pm2 = PortfolioManager(state_file=tmp_state_file)
    assert pm2.positions["AAA"].amount_held == 123.45
    assert pm2.positions["AAA"].initial_cost_sol == 3.14

def test_empty_state_loads_cleanly(tmp_state_file):
    pm = PortfolioManager(state_file=tmp_state_file)
    assert pm.get_open_count() == 0
    pm2 = PortfolioManager(state_file=tmp_state_file)
    assert pm2.get_open_count() == 0
