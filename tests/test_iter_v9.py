"""V9 VWAP filter (15m Top5) tests: recompute slice by script helpers + schema."""
import csv
import json
import pathlib
import subprocess
import sys

import torch

OUT = pathlib.Path("results/iter_V9_vwap.json")
SCRIPT = "research/run_iter_v9_vwap.py"


def _load():
    assert OUT.exists(), "results/iter_V9_vwap.json missing; run research/run_iter_v9_vwap.py"
    return json.loads(OUT.read_text())


def _import_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_v9_vwap", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_iter_v9_lock_and_schema():
    from strategy_manager.config import FORMULA, LEV
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["units"] == ["base", "vwap96"]
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale"] == 16
    assert d["config"]["vwap"]["window"] == 96
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == {"base", "vwap96"}
    for u in ("base", "vwap96"):
        for sec in ("FULL", "FULL_fee2x"):
            for f in ("sharpe", "mdd", "cum", "ann", "n", "turnover", "trades", "long_entries", "short_entries"):
                assert f in d["rows"][u][sec], (u, sec, f)
            assert d["rows"][u][sec]["n"] == 35040
        for c in ("ETC", "TRX", "ATOM", "APT", "KAS"):
            assert c in d["rows"][u]["per_coin"], (u, c)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_NO_VWAP"
    c = d["compare"]
    assert set(c) >= {"d_sharpe", "d_sharpe_fee2x", "turnover_ratio", "trades_base", "trades_vwap", "trades_cut"}
    assert c["trades_base"] == d["rows"]["base"]["FULL"]["trades"]
    assert c["trades_vwap"] == d["rows"]["vwap96"]["FULL"]["trades"]


def test_iter_v9_vwap_math_unit():
    m = _import_script()
    bars = [(10.0, 12.0, 8.0, 11.0, 100.0), (10.0, 11.0, 9.0, 10.0, 200.0), (10.0, 13.0, 9.0, 12.0, 300.0)]
    vw = m.rolling_vwap(bars, window=2)
    # bar0: typ=(12+8+11)/3=10.3333, vol100 -> 10.3333
    assert abs(vw[0] - (12 + 8 + 11) / 3.0) < 1e-9
    # bar1: (10.3333*100 + 10.0*200)/300
    assert abs(vw[1] - ((12 + 8 + 11) / 3.0 * 100 + (11 + 9 + 10) / 3.0 * 200) / 300) < 1e-9
    # bar2: (10.0*200 + 11.3333*300)/500
    assert abs(vw[2] - ((11 + 9 + 10) / 3.0 * 200 + (13 + 9 + 12) / 3.0 * 300) / 500) < 1e-9
    lok, sok = m.vwap_gate_mask([11.0, 10.0, 12.0], [10.0, 10.0, 13.0])
    assert lok == [1.0, 0.0, 0.0]  # above only
    assert sok == [0.0, 0.0, 1.0]  # below only; tie blocks both
    # gate actually suppresses the blocked leg in lp/sp construction
    n = 8
    sig = torch.full((1, n), 5.0)  # strong long everywhere
    raw = {"liquidity": torch.full((1, n), 1e7)}
    import types
    spec = {"lth": 0.88, "sth": 0.12, "cd": 0, "sl": None, "ts": 0, "vt": None, "vw": 12, "q": 0.0}
    rt = torch.zeros(1, n)
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True, long_th=0.88, short_th=0.12, bars_per_year=35040.0)
    sg = torch.sigmoid(sig)
    lp = ((sg > bt.long_th).float())
    assert float(lp.sum()) == n  # all-long without gate
    pos_gate, _ = m.leg_legs({"liquidity": torch.full((1, n), 1e7)}, rt, sig, spec,
                             gate=([0.0] * n, [1.0] * n))
    assert all(abs(v) < 0.5 for v in pos_gate)  # longs blocked -> flat


def test_iter_v9_recompute_slice():
    m = _import_script()
    bars, _ = m.common15m(m.COINS)
    n = len(bars["ETC"])
    assert n == 35040
    # recompute first 1500 bars of TRX base legs with script helpers, compare trades direction
    c = "TRX"
    bc = bars[c][:1500]
    raw, rt, sg = m.build_sig(bc)
    pos, turn = m.leg_legs(raw, rt, sg, m.SPECS[c], None)
    assert len(pos) == 1500 and len(turn) == 1500
    assert set(v for v in pos if abs(v) > 0.5) <= {1.0, -1.0}
    vw = m.rolling_vwap(bc, m.VWAP_WIN)
    assert len(vw) == 1500
    lok, sok = m.vwap_gate_mask([b[3] for b in bc], vw)
    assert len(lok) == 1500 and set(lok) <= {0.0, 1.0} and set(sok) <= {0.0, 1.0}
    pos_g, _ = m.leg_legs(raw, rt, sg, m.SPECS[c], (lok, sok))
    # gated entries subset (gate can only remove pre-q signals)
    el, es = m.count_entries(pos)
    elg, esg = m.count_entries(pos_g)
    assert elg <= el and esg <= es
    d = _load()
    assert d["rows"]["vwap96"]["FULL"]["trades"] <= d["rows"]["base"]["FULL"]["trades"]


def test_iter_v9_script_runs_offline():
    r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["rows"]) == {"base", "vwap96"}
    assert d["partial"] is False
