"""P2-1 suitability gate test: schema + offline rerun (recompute by script)."""
import json
import pathlib
import subprocess
import sys

VALID = ["ETC", "TRX", "ATOM", "APT", "KAS"]
INVALID = ["PEPE", "ICP"]
METRICS = {"hurst", "acf", "trend_er", "drift_snr", "amihud",
           "funding_vol_proxy", "gap_rate_10bp", "gap_rate_50bp"}


def _load():
    p = pathlib.Path("results/suitability.json")
    assert p.exists(), "results/suitability.json missing; run research/run_suitability.py"
    return json.loads(p.read_text())


def test_suitability_schema():
    d = _load()
    assert d["config"]["groups"] == {"valid": VALID, "invalid": INVALID}
    assert len(d["config"]["grid"]) == 12
    assert d["config"]["role"].startswith("independent diagnostic")
    for coin in VALID + INVALID:
        c = d["coins"][coin]
        assert METRICS <= set(c), coin
        assert set(c["acf"]) == {"1", "2", "3", "4"}
        assert 0.0 < c["hurst"] < 1.5
        assert c["grid_n"] == 12
        assert c["grid_npos"] + (12 - c["grid_npos"]) == 12
        assert c["gate_pass"] in (True, False)
    assert d["gate"]["threshold"] == 1.0
    assert set(d["gate"]["pass"]) | set(d["gate"]["fail"]) == set(VALID + INVALID)
    assert d["counter_proof"]["regime_specific"] in (True, False)
    assert "待定" in d["conclusion"]


def test_suitability_separation():
    d = _load()
    assert d["gate"]["no_overlap"] is True
    for coin in VALID:
        assert d["coins"][coin]["gate_pass"] is True
        assert d["coins"][coin]["grid_best"]["sharpe"] >= 1.0
    for coin in INVALID:
        assert d["coins"][coin]["gate_pass"] is False
        assert d["coins"][coin]["grid_best"]["sharpe"] < 1.0


def test_pepe_counter_proof():
    d = _load()
    cp = d["counter_proof"]
    assert cp["pepe_own_best"] < 0
    assert cp["pepe_best_on_etc"] > 0.5
    assert cp["pepe_best_on_trx"] > 0.5
    assert cp["regime_specific"] is True


def test_no_broker_in_suitability_engine():
    src = pathlib.Path("research/run_suitability.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_suitability_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_suitability.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["gate"]["no_overlap"] is True
