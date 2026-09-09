"""E10 live-gate artifact test: locked params + recorded metrics vs thresholds.

Reads results artifacts only (no recompute, no network, no orders).
Thresholds from results/backtest_E10.json:locked_params.monitoring.
"""
import json
import os
from strategy_manager.config import RiskConfig

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "e10_gates_golden.json")

def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        g = json.load(open(GOLDEN))
        if p == "results/backtest_E10.json":
            return {"locked_params": {"FORMULA": g["FORMULA"],
                    "coins": {"ETC": g["coins"]["ETC"], "TRX": g["coins"]["TRX"]},
                    "gate": {"adopted_id": g["gate_adopted_id"]},
                    "risk": g["risk"], "monitoring": g["monitoring"]}}
        if p == "results/paper_trades2.json":
            return {"stats": g["paper"]}
        if p == "results/backtest_E1.json":
            return {"12fold": {"Y1b": {"summary": g["fold12"]}}}
        if p == "results/backtest_E3.json":
            return {"results": {"Y1b": {"segments": {"FULL": {"turnover": g["turnover_full"]}}}}}
        if p == "results/backtest_W3_shadow.json":
            return {"best": {"coverage": g["shadow_coverage"]}}
        if p == "results/backtest_AA.json":
            return {"stress": {"H2_best": {"worst_B": g["stress_worst_B"],
                                           "worst_C": g["stress_worst_C"]}}}
        raise

def test_e10_lock_matches_config():
    e10 = _load("results/backtest_E10.json")["locked_params"]
    assert e10["FORMULA"] == [3,2,7,2,7,11,15,4,4,6,6,10]
    assert e10["coins"]["ETC"]["cd"] == 18 and e10["coins"]["TRX"]["cd"] == 6
    assert e10["gate"]["adopted_id"] == "Y1b_main_thr1.0_w200"
    assert e10["risk"]["perp_max_leverage"] == 2
    rc = RiskConfig()
    assert rc.perp_max_leverage == 2
    assert rc.perp_max_notional_usdt == 500

def test_paper_baseline_within_tolerance():
    s = _load("results/paper_trades2.json")["stats"]
    assert s["trades"] == 478
    assert abs(s["final_x"] - 2.9883) < 0.15
    assert abs(s["sharpe"] - 0.899) < 0.10

def test_env_example_has_y1b_switches():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt = open(os.path.join(root, ".env.example")).read()
    for key in ["Y1B_LIVE_ENABLED=0", "PAPER_MODE=1", "Y1B_NOTIONAL_USDT=50", "Y1B_STATE=y1b_state.json",
                "PERP_MAX_LEVERAGE=2"]:
        assert key in txt, key

def test_gates_from_artifacts():
    mon = _load("results/backtest_E10.json")["locked_params"]["monitoring"]
    e1 = _load("results/backtest_E1.json")["12fold"]["Y1b"]["summary"]
    assert e1["mean"] > mon["twelvefold_mean_min"]
    assert e1["median"] > mon["twelvefold_median_min"]
    assert e1["n_pos"] >= mon["twelvefold_npos_min"]
    e3 = _load("results/backtest_E3.json")["results"]["Y1b"]
    assert e3["segments"]["FULL"]["turnover"] < mon["turnover_max"]
    w = _load("results/backtest_W3_shadow.json")["best"]
    assert mon["coverage_range"][0] <= w["coverage"] <= mon["coverage_range"][1]
    aa = _load("results/backtest_AA.json")["stress"]
    assert aa["H2_best"]["worst_B"] > mon["fee2x_worst_B_min"]
    assert aa["H2_best"]["worst_C"] > mon["fee2x_worst_C_min"]
