"""H13 realized-vs-assumed slippage tests (1h native, Top5).

Schema checks accept either the FULL artifact (5 coins, 8760 bars) or a
smoke artifact; the live subprocess check runs smoke mode only to a temp
OUT so the committed FULL artifact is not clobbered. Pure-math unit tests
(adverse_bp / summarize / slip_drag_ann / parse_fills on synthetic jsonl)
never touch the network or brokers.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_H13_slipreal.json")
SRC = pathlib.Path("research/run_iter_h13_slipreal.py")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
DIST_KEYS = {"n", "mean", "median", "stdev", "min", "max", "q25", "q75",
             "q90", "share_adverse", "share_abs_gt5"}


def _load():
    assert OUT.exists(), "results/iter_H13_slipreal.json missing; run research/run_iter_h13_slipreal.py"
    return json.loads(OUT.read_text())


def _mod(name="h13mod"):
    spec = importlib.util.spec_from_file_location(name, str(SRC))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h13_units():
    m = _mod("h13_units")
    # adverse_bp: positive = paid worse than the plan quote
    assert m.adverse_bp(1.0, 100.0, 101.0) == 100.0  # buy, filled higher
    assert m.adverse_bp(1.0, 100.0, 99.0) == -100.0  # buy, filled lower
    assert m.adverse_bp(-1.0, 100.0, 99.0) == 100.0  # short, covered lower
    assert m.adverse_bp(-1.0, 100.0, 101.0) == -100.0  # short, adverse up
    assert m.adverse_bp(0.0, 100.0, 101.0) is None  # flat has no slip
    assert m.adverse_bp(1.0, 0.0, 101.0) is None  # bad plan price
    assert m.adverse_bp(1.0, None, 101.0) is None
    # summarize: distribution keys + shares bounded
    d = m.summarize([5.0, -5.0, 10.0, 0.0])
    assert set(d) == DIST_KEYS
    assert d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.5
    assert d["min"] == -5.0 and d["max"] == 10.0
    assert d["share_adverse"] == 0.5
    assert d["share_abs_gt5"] == 0.25
    e = m.summarize([])
    assert e["n"] == 0 and set(e) == DIST_KEYS
    # slip_drag_ann: turnover * slip * LEV * BPY
    assert m.slip_drag_ann(0.02, 0.0005) == 0.02 * 0.0005 * 2.0 * 8760.0
    assert m.slip_drag_ann(0.0, 0.0005) == 0.0
    # quantile helper
    assert m.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert m.quantile([], 0.9) == 0.0


def test_iter_h13_parse_fills_unit(tmp_path):
    m = _mod("h13_parse")
    rows = [
        {"ts": "t1", "plans": [
            {"coin": "APT", "symbol": "APTUSDT", "want": -1.0, "price": 1.0},
            {"coin": "ETC", "symbol": "ETCUSDT", "want": 1.0, "price": 10.0}],
         "results": [
            {"symbol": "APTUSDT", "ok": True, "fill": 0.99},
            {"symbol": "ETCUSDT", "ok": True, "fill": 10.05},
            {"symbol": "XXXUSDT", "skipped": True, "reason": "flat"}]},
        {"ts": "t2", "plans": [],
         "results": [{"symbol": "APTUSDT", "ok": True, "fill": 1.0}]},
    ]
    p = tmp_path / "fills.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fills, skipped = m.parse_fills(str(p))
    assert len(fills) == 2
    apt = next(f for f in fills if f["symbol"] == "APTUSDT")
    assert apt["slip_bp"] == 100.0  # opening short sells: fill below plan = adverse
    etc = next(f for f in fills if f["symbol"] == "ETCUSDT")
    assert etc["slip_bp"] == 50.0  # buy filled higher = adverse
    assert skipped["results_skipped"] >= 1
    assert skipped["rows_no_plans"] >= 1
    fills2, _ = m.parse_fills(str(tmp_path / "missing.jsonl"))
    assert fills2 == []


def test_iter_h13_schema():
    d = _load()
    assert d.get("status") == "final"
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert "no adoption" in d["decision_note"].lower()
    assert "live untouched" in d["decision_note"].lower()
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["assumed_slip_bp"] == 5.0
    assert abs(cfg["assumed_slip"] - 5e-4) < 1e-12
    assert cfg["min_fills"] == 10
    assert cfg["fills_src"] == "results/y1b_hourly.jsonl"
    assert cfg["venue"] == "aster" and cfg["lev"] == 2.0
    smoke = cfg["smoke"]
    coins = {"ETC", "TRX"} if smoke else set(COINS)
    assert set(cfg["coins"]) == coins
    assert set(cfg["basket"]) == coins
    assert cfg["grid_bars"] == (3000 if smoke else 8760)
    assert cfg["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    for c in coins:
        assert cfg["basket"][c]["q"] == 0.3
    b = d["basket"]
    assert SEG <= set(b) and b["n"] == cfg["grid_bars"]
    assert b["turnover"] >= 0.0
    assert set(d["per_coin"]) == coins
    for c, leg in d["per_coin"].items():
        assert SEG <= set(leg) and leg["n"] == cfg["grid_bars"]
    r = d["realized"]
    assert r["n_fills"] == len(r["fills"])
    assert DIST_KEYS <= set(r["dist_bp"])
    assert r["dist_bp"]["n"] == r["n_fills"]
    assert 0.0 <= r["dist_bp"]["share_adverse"] <= 1.0
    assert 0.0 <= r["dist_bp"]["share_abs_gt5"] <= 1.0
    for f in r["fills"]:
        assert {"ts", "symbol", "coin", "side_want", "plan", "fill", "slip_bp"} <= set(f)
        assert f["plan"] > 0 and f["fill"] > 0
        assert f["side_want"] in (1.0, -1.0)
    for c, dd in r["by_coin_bp"].items():
        assert DIST_KEYS <= set(dd)
    c = d["compare"]
    for f in ("turnover", "assumed_bp", "realized_mean_bp",
              "realized_q90_bp", "drag_assumed_yr", "drag_realized_mean_yr",
              "drag_realized_q90_yr", "fee_drag_yr",
              "drag_ratio_mean_over_assumed", "conservative_covers_q90",
              "n_fills", "sample_adequate"):
        assert f in c, f
    assert c["n_fills"] == r["n_fills"]
    assert c["sample_adequate"] == (c["n_fills"] >= cfg["min_fills"])
    exp_drag = c["turnover"] * 5e-4 * 2.0 * 8760.0
    assert abs(c["drag_assumed_yr"] - round(exp_drag, 4)) < 1e-9
    assert c["drag_realized_mean_yr"] == round(
        c["turnover"] * (c["realized_mean_bp"] * 1e-4) * 2.0 * 8760.0, 4)
    assert c["conservative_covers_q90"] == (
        c["drag_assumed_yr"] >= c["drag_realized_q90_yr"])
    assert "PENDING" in d["conclusion"]


def test_iter_h13_no_broker():
    src = SRC.read_text()
    assert "data/data_1y/1h" in src
    assert "results/y1b_hourly.jsonl" in src
    assert "8760" in src
    for bad in ("place_order", "submit_order", "api_key",
                "make_broker", "Y1B_LIVE", "market_open"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h13_script_runs_offline():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H13_slipreal.json")
        log = os.path.join(td, "iter_H13_slipreal.log")
        env = dict(os.environ, ITER_H13_SMOKE="1",
                   ITER_H13_OUT=out, ITER_H13_LOG=log)
        r = subprocess.run([sys.executable, str(SRC)], capture_output=True,
                           text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["status"] == "final"
        assert d["verdict"] == "PENDING"
        assert d["decision"] == "NO_ADOPTION"
        assert d["config"]["smoke"] is True
        assert set(d["config"]["coins"]) == {"ETC", "TRX"}
        assert d["realized"]["n_fills"] >= 0
