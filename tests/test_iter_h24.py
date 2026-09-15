"""H24 1h quantile sweep (Top5) tests. Recompute-cheap, never clobber FULL artifact."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H24_q.json")
SRC = pathlib.Path("research/run_iter_h24_q.py")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
ATTR = {"long_entries", "short_entries", "long_pnl", "short_pnl",
        "long_share", "short_share"}
QS = [0.1, 0.2, 0.3, 0.4, 0.5]
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}


def _load():
    assert OUT.exists(), "results/iter_H24_q.json missing; run research/run_iter_h24_q.py"
    return json.loads(OUT.read_text())


def _run_smoke(tmp_path):
    out = tmp_path / "iter_H24_q.json"
    lg = tmp_path / "iter_H24_q.log"
    env = dict(os.environ, ITER_H24_SMOKE="1", ITER_H24_OUT=str(out),
               ITER_H24_LOG=str(lg))
    r = subprocess.run([sys.executable, str(SRC)], capture_output=True,
                       text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text())


def test_iter_h24_lock():
    from strategy_manager.config import LEV, FEE, FUND, FEE2X, FORMULA
    from strategy_manager.config import LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FEE == 0.0004 and FEE2X == 0.0008 and FUND == 0.0005
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_h24_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["qs"] == QS
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["lev"] == 2.0
    assert cfg["fund"] == 0.0005
    assert cfg["fee"] == 0.0004
    assert cfg["fee2x"] == 0.0008
    assert cfg["smoke"] is False
    assert set(cfg["weights"]) == COINS
    assert cfg["grid_bars"] == 8760
    assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
    rows = d["rows"]
    assert [r["q"] for r in rows] == QS
    for r in rows:
        assert SEG <= set(r["FULL"]), r["q"]
        assert SEG <= set(r["FULL_fee2x"]), r["q"]
        assert ATTR <= set(r["attribution_full"]), r["q"]
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert r["FULL_fee2x"]["n"] == cfg["grid_bars"]
        a = r["attribution_full"]
        assert a["long_entries"] + a["short_entries"] > 0
        assert abs(a["long_pnl"] + a["short_pnl"] - r["FULL"]["cum"]) < 0.05
        assert abs(r["gap_fee2x_sharpe"] - round(r["FULL_fee2x"]["sharpe"] - r["FULL"]["sharpe"], 3)) < 1e-9
        assert r["FULL_fee2x"]["sharpe"] <= r["FULL"]["sharpe"] + 1e-9
    assert len(d["fee2x_curve"]) == len(QS)
    assert [c["q"] for c in d["fee2x_curve"]] == QS
    assert [c["FULL_fee2x_sharpe"] for c in d["fee2x_curve"]] == [r["FULL_fee2x"]["sharpe"] for r in rows]
    assert d["knee"]["q"] in QS
    assert 0 <= d["knee"]["idx"] < len(QS)
    assert d["q03_check"]["isolated_peak"] in (True, False)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_q0.3"
    assert "PENDING" in d["conclusion"]


def test_iter_h24_no_broker():
    src = SRC.read_text()
    assert "data/data_1y/1h" in src
    assert "8760" in src
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h24_smoke_runs_offline(tmp_path):
    d = _run_smoke(tmp_path)
    assert d["config"]["smoke"] is True
    assert d["config"]["qs"] == [0.1, 0.3, 0.5]
    assert d["config"]["grid_bars"] == 3000
    assert [r["q"] for r in d["rows"]] == [0.1, 0.3, 0.5]
    for r in d["rows"]:
        assert SEG <= set(r["FULL"])
        assert SEG <= set(r["FULL_fee2x"])
        assert ATTR <= set(r["attribution_full"])
    assert d["knee"]["q"] in [0.1, 0.3, 0.5]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_q0.3"
