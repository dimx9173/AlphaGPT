"""H54 1h Bollinger-filter tests (Top5): outside-band entry gate.

Schema accepts full or smoke artifact. Recompute-by-script only via
smoke subprocess; never re-runs the full sweep here. Verdict PENDING
(P0-3 FAIL), no adoption, live untouched.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H54_bb.json")
SRC = pathlib.Path("research/run_iter_h54_bb.py")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3},
    "TRX": {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3},
    "ATOM": {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3},
    "APT": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3},
    "KAS": {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24, "q": 0.3},
}


def _load():
    assert OUT.exists(), "results/iter_H54_bb.json missing; run research/run_iter_h54_bb.py"
    return json.loads(OUT.read_text())


def test_iter_h54_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["bb"]["window"] == 96
    assert cfg["bb"]["k"] == 2.0
    assert cfg["bb"]["warmup_bars"] == 96
    assert "outside" in cfg["bb"]["gate"]
    coins = cfg.get("coins", COINS)
    assert set(coins) <= set(COINS)
    assert len(coins) >= 2
    assert cfg["grid_bars"] >= (2500 if cfg["smoke"] else 8000)
    assert cfg["h2_start"] == cfg["grid_bars"] // 2
    assert set(cfg["basket"]) == set(COINS) if not cfg["smoke"] else set(cfg["basket"]) == set(coins)
    for c in coins:
        assert cfg["basket"][c]["q"] == 0.3, c
        for k in ("lth", "sth", "cd", "sl", "ts"):
            assert cfg["basket"][c][k] == BASE_SPECS[c][k], (c, k)
    assert set(d["coins"]) == set(coins)
    for c in coins:
        cell = d["coins"][c]
        assert SEG <= set(cell["FULL"]), c
        assert SEG <= set(cell["BASE_FULL"]), c
        assert SEG <= set(cell["LONG"]), c
        assert SEG <= set(cell["SHORT"]), c
        assert cell["FULL"]["n"] == cfg["grid_bars"], c
        assert isinstance(cell["FULL"]["sharpe"], float), c
        assert isinstance(cell["FULL"]["trades"], int), c
        assert cell["FULL"]["trades"] == cell["LONG"]["trades"] + cell["SHORT"]["trades"], c
        assert cell["FULL"]["trades"] >= 0, c
        assert 0.0 <= cell["bb"]["frac_outside"] <= 1.0, c
        assert cell["bb"]["n_outside"] == round(cell["bb"]["frac_outside"] * cfg["grid_bars"]), c
        assert isinstance(cell["H2_FULL_sharpe"], float), c
    assert SEG <= set(d["basket"]["FULL"])
    assert SEG <= set(d["baseline"]["basket_FULL"])
    assert set(d["basket"]["coin_pnl_share"]) == set(coins)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert "PENDING" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]
    assert d["status"] == "COMPLETE"


def test_iter_h54_bb_math():
    """BB(96,2) helper: population-SD bands, causal, warmup neutral."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("h54bb", str(SRC))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h54bb"] = mod
    spec.loader.exec_module(mod)
    flat = [10.0] * 120
    mid, up, lo, wd = mod.bollinger(flat)
    assert mid[96] == 10.0
    assert up[96] == 10.0 and lo[96] == 10.0 and wd[96] == 0.0
    assert mid[0] == 0.0 and up[0] == 0.0  # warmup
    ramp = [float(i) for i in range(1, 121)]
    mid2, up2, lo2, wd2 = mod.bollinger(ramp)
    m = sum(ramp[1:97]) / 96
    v = sum((x - m) ** 2 for x in ramp[1:97]) / 96
    assert abs(mid2[96] - m) < 1e-9
    assert abs(up2[96] - (m + 2.0 * math.sqrt(v))) < 1e-9
    assert abs(lo2[96] - (m - 2.0 * math.sqrt(v))) < 1e-9
    g = mod.gate_outside(ramp, up2, lo2)
    assert all(v == 0.0 for v in g[:96])  # warmup closed
    # flat series: price on band (sd=0) -> inside -> closed
    gf = mod.gate_outside(flat, up, lo)
    assert all(v == 0.0 for v in gf)
    # breakout series: last close far above -> open
    outs = [10.0] * 96 + [50.0]
    mo, uo, lo_, _ = mod.bollinger(outs)
    go = mod.gate_outside(outs, uo, lo_)
    assert go[96] == 1.0


def test_iter_h54_no_broker():
    src = SRC.read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h54_engine_mirror():
    src = SRC.read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec[",
                  "time_stop=int(spec[", "vol_window=int(spec[",
                  "bollinger", "gate_outside", "BB_WINDOW = 96", "BB_K = 2.0"):
        assert token in src, token
    assert "short_enabled=True" in src
    assert "[3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]" in src


def test_iter_h54_smoke_runs_offline():
    env = dict(os.environ, ITER_H54_SMOKE="1",
               ITER_H54_OUT="results/iter_H54_bb_smoke.json",
               ITER_H54_LOG="logs/iter_H54_bb_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_h54_bb.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("results/iter_H54_bb_smoke.json").read_text())
    assert d["config"]["smoke"] is True
    assert d["config"]["bb"]["window"] == 96
    assert set(d["coins"]) == {"ETC", "TRX"}
    for c in ("ETC", "TRX"):
        assert SEG <= set(d["coins"][c]["FULL"])
        assert isinstance(d["coins"][c]["FULL"]["trades"], int)
    assert d["verdict"] == "PENDING"
    assert d["status"] == "COMPLETE"
