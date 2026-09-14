"""Z5 15m quantile sweep (Top5) tests: q {0.1..0.5} FULL sharpe/turnover/attribution + fee2x knee."""
import json
import math
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_Z5_OUT", "results/iter_Z5_q.json"))
SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ATTR = {"long_entries", "short_entries", "long_pnl", "short_pnl", "long_share", "short_share"}

def _load():
    assert OUT.exists(), "results/iter_Z5_q.json missing; run research/run_iter_z5_q.py"
    return json.loads(OUT.read_text())

def test_iter_z5_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)

def test_iter_z5_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["grid"] == "15m"
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["venue"] == "aster" and cfg["lev"] == 2.0
    assert cfg["smoke"] in (True, False)
    if cfg["smoke"]:
        assert cfg["qs"] == [0.1, 0.3]
        assert cfg["coins"] == ["ETC", "TRX"]
        assert cfg["grid_bars"] == 3000
    else:
        assert cfg["qs"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert cfg["grid_bars"] == 35040
    n = cfg["grid_bars"]
    assert len(d["rows"]) == len(cfg["qs"])
    assert [r["q"] for r in d["rows"]] == cfg["qs"]
    for r in d["rows"]:
        assert SEG <= set(r["FULL"]), r["q"]
        assert SEG <= set(r["H2"]), r["q"]
        assert SEG <= set(r["FULL_fee2x"]), r["q"]
        assert SEG <= set(r["H2_fee2x"]), r["q"]
        assert ATTR <= set(r["attribution_full"]), r["q"]
        assert ATTR <= set(r["attribution_h2"]), r["q"]
        assert r["FULL"]["n"] == n
        assert r["FULL_fee2x"]["n"] == n
        assert r["H2"]["n"] == n - cfg["h2_start"]
        assert r["FULL"]["turnover"] >= 0
        assert r["attribution_full"]["long_entries"] >= 0
        assert r["attribution_full"]["short_entries"] >= 0
        # fee2x never beats base fee on the same grid
        assert r["FULL_fee2x"]["sharpe"] <= r["FULL"]["sharpe"] + 1e-9, r["q"]
    assert len(d["fee2x_curve"]) == len(cfg["qs"])
    assert len(d["base_curve"]) == len(cfg["qs"])
    for e, r in zip(d["fee2x_curve"], d["rows"]):
        assert e["q"] == r["q"]
        assert e["FULL_fee2x_sharpe"] == r["FULL_fee2x"]["sharpe"]
    for e, r in zip(d["base_curve"], d["rows"]):
        assert e["q"] == r["q"]
        assert e["FULL_sharpe"] == r["FULL"]["sharpe"]
    # knee recompute: max perpendicular distance on (q, fee2x curve)
    xs = cfg["qs"]; ys = [r["FULL_fee2x"]["sharpe"] for r in d["rows"]]
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.sqrt(dx * dx + dy * dy)
    if norm >= 1e-12 and len(xs) >= 3:
        ds = [abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm for x, y in zip(xs, ys)]
        exp_i = max(range(len(xs)), key=lambda i: ds[i])
        assert d["knee"]["idx"] == exp_i
        assert d["knee"]["q"] == xs[exp_i]
    assert d["knee"]["curve"] == ys
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_q0.3"
    assert "PENDING" in d["conclusion"]
    assert "no live change" in d["conclusion"]

def test_iter_z5_attribution_consistent():
    d = _load()
    for r in d["rows"]:
        a = r["attribution_full"]
        tot = a["long_pnl"] + a["short_pnl"]
        # side split must tile the FULL cum (rounding tolerance)
        assert abs(tot - r["FULL"]["cum"]) < 0.05, (r["q"], tot, r["FULL"]["cum"])
        if abs(r["FULL"]["cum"]) < 1e-9:
            assert a["long_share"] == 0.0 and a["short_share"] == 0.0
        else:
            s = round(a["long_share"] + a["short_share"], 4)
            exp = 1.0 if r["FULL"]["cum"] >= 0 else -1.0
            assert abs(s - exp) < 1e-3, (r["q"], s, exp)
        # fee2x costs more on identical positions: cum never exceeds base cum
        assert r["FULL_fee2x"]["cum"] <= r["FULL"]["cum"] + 1e-9, (r["q"], r["FULL_fee2x"]["cum"], r["FULL"]["cum"])

def test_iter_z5_no_broker():
    src = pathlib.Path("research/run_iter_z5_q.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad

def test_iter_z5_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_Z5_q.json"
    lg = tmp_path / "iter_Z5_q.log"
    env = dict(os.environ, ITER_Z5_SMOKE="1", ITER_Z5_OUT=str(out), ITER_Z5_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_z5_q.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert dd["config"]["qs"] == [0.1, 0.3]
    assert dd["config"]["grid_bars"] == 3000
    assert len(dd["rows"]) == 2
    assert dd["verdict"] == "PENDING"
    assert dd["decision"] == "KEEP_q0.3"
