"""Z9 H1/H2 split-half tests: schema + recompute by script."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_Z9_h1h2.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_Z9_h1h2.json missing; run research/run_iter_z9_h1h2.py"
    return json.loads(OUT.read_text())


def test_iter_z9_lock():
    from strategy_manager.config import FORMULA, LEV, FUND, FEE
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert LEV == 2.0
    assert FUND == 0.0005
    assert FEE == 0.0004


def test_iter_z9_schema():
    d = _load()
    assert set(d["per_coin"]) == set(COINS)
    for c in COINS:
        for seg in ("FULL", "H1", "H2"):
            for f in ("sharpe", "ann", "mdd", "cum", "n", "turnover"):
                assert f in d["per_coin"][c][seg], (c, seg, f)
    n = d["config"]["grid_bars"]
    assert n == 35040
    h1, h2 = d["config"]["h1"], d["config"]["h2"]
    assert h1 == [0, 17520] and h2 == [17520, 35040]
    for c in COINS:
        assert d["per_coin"][c]["H1"]["n"] == 17520
        assert d["per_coin"][c]["H2"]["n"] == 17520
        assert d["per_coin"][c]["FULL"]["n"] == 35040
    for seg in ("FULL", "H1", "H2"):
        assert set(d["ranks"][seg]) == set(COINS)
    assert isinstance(d["rank_corr_H1_vs_H2"], float)
    assert -1.0 <= d["rank_corr_H1_vs_H2"] <= 1.0
    for seg in ("FULL", "H1", "H2"):
        assert d["basket"][seg]["n"] == (35040 if seg == "FULL" else 17520)
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale_4h_to_15m"] == 16
    assert set(d["config"]["weights"]) == set(COINS)


def test_iter_z9_rank_corr_recompute():
    import math
    d = _load()
    h1 = {c: d["per_coin"][c]["H1"]["sharpe"] for c in COINS}
    h2 = {c: d["per_coin"][c]["H2"]["sharpe"] for c in COINS}
    ra = {k: i + 1 for i, (k, _) in enumerate(sorted(h1.items(), key=lambda kv: (-kv[1], kv[0])))}
    rb = {k: i + 1 for i, (k, _) in enumerate(sorted(h2.items(), key=lambda kv: (-kv[1], kv[0])))}
    xa = [ra[c] for c in COINS]
    xb = [rb[c] for c in COINS]
    ma = sum(xa) / len(xa)
    mb = sum(xb) / len(xb)
    cov = sum((a - ma) * (b - mb) for a, b in zip(xa, xb))
    va = sum((a - ma) ** 2 for a in xa)
    vb = sum((b - mb) ** 2 for b in xb)
    rho = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    assert d["ranks"]["H1"] == ra
    assert d["ranks"]["H2"] == rb
    assert abs(d["rank_corr_H1_vs_H2"] - round(rho, 4)) < 1e-9


def test_iter_z9_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_z9_h1h2.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["per_coin"]) == set(COINS)
    assert d["verdict"] == "PENDING_P03_FAIL"
