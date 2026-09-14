"""W8 exit-timing (15m native) tests: early-exit K in {0,1,2} vs hold-to-flip. Recompute by script."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_W8_exit.json")
SCRIPT = "research/run_iter_w8_exit.py"


def _load():
    assert OUT.exists(), "results/iter_W8_exit.json missing; run research/run_iter_w8_exit.py"
    return json.loads(OUT.read_text())


def test_iter_w8_exit_lock():
    from strategy_manager.config import LEV
    assert LEV == 2.0


def test_iter_w8_exit_schema():
    d = _load()
    assert d["config"]["ks"] == [0, 1, 2]
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["oracle"] is True
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == {"0", "1", "2"}
    for k, v in d["rows"].items():
        for f in ("sharpe", "mdd", "final_x", "cum", "ann", "n", "turnover", "trades"):
            assert f in v["FULL"], (k, f)
        assert v["FULL"]["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_HOLD_TO_FLIP"
    c = d["compare"]
    assert set(c["sharpe_by_k"]) == {"0", "1", "2"}
    assert set(c["trades_by_k"]) == {"0", "1", "2"}
    assert set(c["turnover_by_k"]) == {"0", "1", "2"}


def test_iter_w8_exit_baseline_is_hold():
    d = _load()
    # K=0 must equal the plain hold-to-flip run: no bars zeroed, trades defined
    assert d["rows"]["0"]["FULL"]["trades"] > 0
    assert d["compare"]["d_sharpe_vs_hold"]["1"] == round(d["rows"]["1"]["FULL"]["sharpe"] - d["rows"]["0"]["FULL"]["sharpe"], 4)
    assert d["compare"]["d_trades_vs_hold"]["1"] == d["rows"]["1"]["FULL"]["trades"] - d["rows"]["0"]["FULL"]["trades"]
    # early-exit monotonically cuts exposure -> turnover must not rise vs hold
    assert d["rows"]["1"]["FULL"]["turnover"] <= d["rows"]["0"]["FULL"]["turnover"]
    assert d["rows"]["2"]["FULL"]["turnover"] <= d["rows"]["0"]["FULL"]["turnover"]


def test_iter_w8_exit_early_exit_cut_is_strict():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_w8_exit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_iter_w8_exit"] = mod
    spec.loader.exec_module(mod)
    pos = [1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0]
    assert mod.apply_early_exit(pos, 0) == pos
    cut1 = mod.apply_early_exit(pos, 1)
    assert cut1[2] == 0.0 and cut1[3] == -1.0  # bar before flip zeroed
    assert cut1[7] == 1.0  # bar far from flip intact
    cut2 = mod.apply_early_exit(pos, 2)
    assert cut2[1] == 0.0 and cut2[2] == 0.0
    # no-position signal: no flips -> identity
    flat = [0.0] * 8
    assert mod.apply_early_exit(flat, 2) == flat
    # exit-to-flat also counts as an event
    assert mod.apply_early_exit([1.0, 1.0, 0.0], 1) == [1.0, 0.0, 0.0]


def test_iter_w8_exit_script_runs_offline():
    r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["rows"]) == {"0", "1", "2"}
