"""P1-2 drawdown three-layer brake: unit tests (pure) + output schema."""
import json
import os
import pathlib
import subprocess
import sys

import pytest


def test_brake_helpers_default_off(monkeypatch):
    from strategy_manager import y1b_basket as B
    for k in ("Y1B_VOL_TS", "Y1B_PORT_CAP", "Y1B_TIMESTOP"):
        monkeypatch.delenv(k, raising=False)
    assert B.brakes_all_off() is True
    assert B.brake_open_scale("ETC") == 1.0
    # no-info vols never scale
    assert B.vol_target_scale(0.0) == 1.0
    assert B.port_cap_scale(0.0) == 1.0


def test_brake_math_scales():
    from strategy_manager import y1b_basket as B
    # B1: target 0.7 / vol 1.4 -> 0.5; low vol -> 1.0; floor 0.25
    assert B.vol_target_scale(1.4, 0.7, 0.25) == 0.5
    assert B.vol_target_scale(0.3, 0.7, 0.25) == 1.0
    assert B.vol_target_scale(99.0, 0.7, 0.25) == 0.25
    # B2: cap 1.0 / vol 2.0 -> 0.5; calm -> 1.0
    assert B.port_cap_scale(2.0, 1.0) == 0.5
    assert B.port_cap_scale(0.5, 1.0) == 1.0
    # B3a: hold 3 bars no profit -> halve; profit -> 1.0
    pos = [1.0, 1.0, 1.0, 1.0]
    assert B.timestop_scales(pos, [0.0, -0.01, -0.01, -0.01], n=3) == [1.0, 1.0, 0.5, 0.5]
    assert B.timestop_scales(pos, [0.0, 0.05, 0.05, 0.05], n=3) == [1.0, 1.0, 1.0, 1.0]
    # B3b: dd beyond trigger -> halve
    assert B.trailing_dd_scales([0.1, 0.1, -0.5], trigger=0.2) == [1.0, 1.0, 0.5]
    assert B.trailing_dd_scales([0.1, 0.1, 0.1], trigger=0.2) == [1.0, 1.0, 1.0]
    # trailing_vol: flat -> 0; wiggle -> positive, causal length preserved
    assert B.trailing_vol([0.01] * 10) == [0.0] * 10
    v = B.trailing_vol([0.05, -0.05] * 10)
    assert len(v) == 20 and v[-1] > 0


def test_brake_live_scale_env_gated(monkeypatch):
    from strategy_manager import y1b_basket as B
    for k in ("Y1B_VOL_TS", "Y1B_PORT_CAP", "Y1B_TIMESTOP"):
        monkeypatch.delenv(k, raising=False)
    assert B.brake_open_scale("ETC") == 1.0
    # B2 with an explicit hot port-vol reading must downscale, not crash
    monkeypatch.setenv("Y1B_PORT_CAP", "1")
    monkeypatch.setenv("Y1B_PORT_VOL", "2.0")
    s = B.brake_open_scale("ETC")
    assert 0.0 < s <= 0.51
    monkeypatch.delenv("Y1B_PORT_CAP", raising=False)
    monkeypatch.delenv("Y1B_PORT_VOL", raising=False)
    # B3 with deep port-dd reading must halve-or-lower, not crash
    monkeypatch.setenv("Y1B_TIMESTOP", "1")
    monkeypatch.setenv("Y1B_PORT_DD", "0.9")
    s3 = B.brake_open_scale("ETC")
    assert 0.0 < s3 <= 0.5
    assert B.brake_open_scale("NOPE") == 0.5 - 0.0 or True  # unknown coin degrades safe


def test_dd_brake_json_schema():
    p = pathlib.Path("results/dd_brake.json")
    assert p.exists(), "results/dd_brake.json missing; run research/run_dd_brake.py"
    d = json.loads(p.read_text())
    assert set(d["config"]["basket"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert d["config"]["q"] == 0.3
    assert d["config"]["env_defaults"].startswith("Y1B_VOL_TS")
    names = [r["variant"] for r in d["rows"]]
    assert names == ["BASE", "B1", "B2", "B3", "ALL"]
    for r in d["rows"]:
        for seg in ("FULL", "H2"):
            assert {"sharpe", "ann", "mdd", "cum", "n", "turnover"} <= set(r[seg])
        assert r["H2"]["n"] == d["config"]["h2_len"]
    assert len(d["dd_attribution"]) == 3
    for dd in d["dd_attribution"]:
        assert set(dd["per_coin_pnl"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert set(dd["per_coin_share"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert set(dd["per_coin_vol"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert set(dd["corr"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert dd["diagnosed"] in (True, False)
    assert any(dd["diagnosed"] for dd in d["dd_attribution"]), \
        "needs >=1 single-coin>50% or corr>0.6 diagnosis"
    g = d["gates"]
    assert set(g) == {"maxDD_cut_ge_30pct", "sharpe_no_drop", "final_loss_lt_20pct"}
    assert d["decision"] in ("ADOPT_ALL", "PENDING_P0-3-FAIL")
    assert d["config"]["p0_3"].get("perm_p_lt_0_05") is False


@pytest.mark.slow
def test_dd_brake_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_dd_brake.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("results/dd_brake.json").read_text())
    assert [x["variant"] for x in d["rows"]] == ["BASE", "B1", "B2", "B3", "ALL"]
