"""H2 1h weights tests (Top5 equal vs invvol/invvol_cap/meanvar/riskparity)."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H2_weights.json")
ARMS = ["equal", "invvol", "invvol_cap", "meanvar", "riskparity"]


def _load():
    assert OUT.exists(), "results/iter_H2_weights.json missing; run research/run_iter_h2_weights.py"
    return json.loads(OUT.read_text())


def test_iter_h2_weights_arms_present():
    d = _load()
    assert set(d["arms"]) == set(ARMS), d["arms"].keys()
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"].startswith("KEEP_equal") or d["decision"].startswith("REVIEW_")


def test_iter_h2_weights_schema():
    d = _load()
    assert d["config"]["grid"] == "1h"
    assert d["config"]["grid_bars"] == 8760
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["vol_window"] == 60
    assert d["config"]["clip"] == [0.10, 0.35]
    assert d["config"]["vol_target"] == 0.35
    assert d["config"]["lev_clamp"] == [0.25, 2.0]
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert set(d["config"]["locked_4h"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["config"]["specs_1h"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    for k in ("cd", "ts", "vw"):
        for c in ("ETC", "TRX", "ATOM", "APT", "KAS"):
            assert d["config"]["specs_1h"][c][k] == d["config"]["locked_4h"][c][k] * 4, (c, k)
    for arm, v in d["arms"].items():
        for sec in ("FULL", "H2", "fee2x_FULL", "fold12"):
            assert sec in v, (arm, sec)
        for f in ("sharpe", "mdd", "final_x", "turnover"):
            assert f in v["FULL"], (arm, f)
        assert v["FULL"]["n"] == d["config"]["grid_bars"]
        assert abs(sum(v["w_final"].values()) - 1.0) < 1e-6, (arm, v["w_final"])
        assert abs(sum(v["w_mean"].values()) - 1.0) < 1e-3, (arm, v["w_mean"])


def test_iter_h2_weights_equal_static():
    d = _load()
    e = d["arms"]["equal"]
    assert all(abs(w - 0.2) < 1e-9 for w in e["w_final"].values())
    assert e["lev_final"] == 1.0
    assert e["lev_mean"] == 1.0
    assert all(abs(w - 0.2) < 1e-9 for w in e["w_mean"].values())


def test_iter_h2_weights_decision_rule():
    d = _load()
    base = d["arms"]["equal"]["FULL"]
    for arm, v in d["compare"].items():
        f = d["arms"][arm]["FULL"]
        expect = bool(f["sharpe"] > base["sharpe"] and f["mdd"] < base["mdd"]
                      and f["turnover"] < base["turnover"])
        assert v["beats_equal_on_all3"] is expect, arm
    winners = [a for a, v in d["compare"].items() if v["beats_equal_on_all3"]]
    if winners:
        assert d["decision"] == "REVIEW_" + winners[0]
    else:
        assert d["decision"] == "KEEP_equal"


def test_iter_h2_weights_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_h2_weights.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["arms"]) == set(ARMS)
    assert d["verdict"] == "PENDING_P03_FAIL"
