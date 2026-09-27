from research.universe_28c import ACCEPTANCE_GATE_28C

def test_28c_acceptance_gate_min_positive_coins_is_24():
    assert ACCEPTANCE_GATE_28C["min_positive_coins"] == 24
    assert ACCEPTANCE_GATE_28C["min_positive_walk_forward_folds"] == 3
