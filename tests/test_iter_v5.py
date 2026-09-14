"""V5 15m Bollinger entry-filter (Top5) tests: base vs BB(96,2) FULL sharpe/trades."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path(os.getenv("ITER_V5_OUT", "results/iter_V5_bb.json"))
SRC = pathlib.Path("research/run_iter_v5_bb.py")
SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}


def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_v5_bb.py" % path
    return json.loads(path.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location("run_iter_v5_bb", str(SRC))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_iter_v5_locks():
    from strategy_manager.config import LEV, FUND, FEE, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_v5_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["bb"] == {"window": 96, "k": 2.0, "on": "close", "warmup": "pass-through", "gate": "raw intent pre-cooldown"}
    assert cfg["arms"] == ["base", "bb"]
    assert cfg["grid"] == "15m"
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["venue"] == "aster" and cfg["lev"] == 2.0 and cfg["lev_locked"] == 2.0
    assert cfg["smoke"] in (True, False)
    if cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX"]
        assert cfg["grid_bars"] == 3000
    else:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert cfg["grid_bars"] == 35040
    n = cfg["grid_bars"]
    assert set(d["units"]) == set(cfg["coins"])
    for c in cfg["coins"]:
        u = d["units"][c]
        assert SEG <= set(u["base"]["FULL"]), c
        assert SEG <= set(u["bb"]["FULL"]), c
        assert u["base"]["FULL"]["n"] == n
        assert u["bb"]["FULL"]["n"] == n
        assert u["base"]["trades"] >= 0 and u["bb"]["trades"] >= 0
        assert u["d_sharpe"] == round(u["bb"]["FULL"]["sharpe"] - u["base"]["FULL"]["sharpe"], 3)
        assert u["d_trades"] == u["bb"]["trades"] - u["base"]["trades"]
        assert 0 <= u["bb_pass"] <= n
        assert u["bb_pass_rate"] == round(u["bb_pass"] / n, 5)
    for arm in ("base", "bb"):
        assert SEG <= set(d[arm]), arm
        assert d[arm]["n"] == n
        assert d[arm]["trades"] >= 0
    cmp = d["compare"]
    assert cmp["d_sharpe"] == round(d["bb"]["sharpe"] - d["base"]["sharpe"], 3)
    assert cmp["d_trades"] == d["bb"]["trades"] - d["base"]["trades"]
    assert cmp["d_turnover"] == round(d["bb"]["turnover"] - d["base"]["turnover"], 6)
    assert d["verdict"] == "PENDING"
    assert d["verdict_detail"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_CHANGE"
    assert d["status"] == "done"
    assert "no live change" in d["conclusion"]


def test_iter_v5_bb_mask_semantics():
    m = _mod()
    assert m.BB_WIN == 96 and m.BB_K == 2.0
    flat = [100.0] * 200
    out = m.bb_mask(flat, 96, 2.0)
    assert out[:95] == [1.0] * 95
    assert out[95:] == [0.0] * 105
    sp = [100.0] * 200
    sp[-1] = 200.0
    out2 = m.bb_mask(sp, 96, 2.0)
    assert out2[-1] == 1.0
    assert all(v in (0.0, 1.0) for v in out2)
    # strict inequality: close exactly on a zero-width band stays inside
    assert m.bb_mask([5.0] * 100, 96, 2.0)[99] == 0.0
    # brute-force cross-check on noisy data
    import random
    random.seed(7)
    closes = [100.0 + random.gauss(0, 1) for _ in range(500)]
    got = m.bb_mask(closes, 96, 2.0)
    import math
    for tt in range(95, 500):
        w = closes[tt - 95:tt + 1]
        mm = sum(w) / 96
        vv = max(sum(x * x for x in w) / 96 - mm * mm, 0.0)
        sd = math.sqrt(vv)
        exp = 1.0 if (closes[tt] > mm + 2.0 * sd or closes[tt] < mm - 2.0 * sd) else 0.0
        assert got[tt] == exp, tt


def test_iter_v5_bb_never_adds_trades_and_native_grid():
    d = _load()
    for c, u in d["units"].items():
        assert u["bb"]["trades"] <= u["base"]["trades"], (c, u)
        assert u["bb"]["FULL"]["turnover"] <= u["base"]["FULL"]["turnover"] + 1e-9, (c, u)
    assert d["compare"]["d_trades"] <= 0
    src = SRC.read_text()
    assert "data/data_1y/15m" in src
    assert "35040" in src
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_v5_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "v5.json")
        log = os.path.join(td, "v5.log")
        env = dict(os.environ, ITER_V5_SMOKE="1", ITER_V5_OUT=out, ITER_V5_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_v5_bb.py"], capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(open(out).read())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert set(dd["units"]) == {"ETC", "TRX"}
        assert dd["verdict"] == "PENDING"
        assert dd["status"] == "done"
