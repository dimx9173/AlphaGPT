"""H71 1h Keltner-filter tests (Top5): outside-channel entry gate Keltner(24,2xATR).

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

OUT = pathlib.Path("results/iter_H71_kelt.json")
SRC = pathlib.Path("research/run_iter_h71_kelt.py")
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
    assert OUT.exists(), "results/iter_H71_kelt.json missing; run research/run_iter_h71_kelt.py"
    return json.loads(OUT.read_text())


def test_iter_h71_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["kelt"]["window"] == 24
    assert cfg["kelt"]["mult"] == 2.0
    assert cfg["kelt"]["warmup_bars"] == 48
    assert "outside" in cfg["kelt"]["gate"]
    coins = cfg.get("coins", COINS)
    assert set(coins) <= set(COINS)
    assert len(coins) >= 2
    assert cfg["grid_bars"] >= (2500 if cfg["smoke"] else 8000)
    assert cfg["h2_start"] == cfg["grid_bars"] // 2
    if not cfg["smoke"]:
        assert set(cfg["basket"]) == set(COINS)
    else:
        assert set(cfg["basket"]) == set(coins)
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
        assert 0.0 <= cell["kelt"]["frac_outside"] <= 1.0, c
        assert cell["kelt"]["n_outside"] == round(cell["kelt"]["frac_outside"] * cfg["grid_bars"]), c
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


def test_iter_h71_kelt_math():
    """Keltner(24,2) helper: EMA mid + Wilder ATR bands, causal, warmup neutral."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("h71kelt", str(SRC))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h71kelt"] = mod
    spec.loader.exec_module(mod)
    assert mod.KELT_WINDOW == 24
    assert mod.KELT_MULT == 2.0
    assert mod.WARMUP == 48
    n = 120
    flat = [10.0] * n
    hi = [10.5] * n
    lo = [9.5] * n
    mid, up, lw, wd = mod.keltner(hi, lo, flat)
    assert all(v == 0.0 for v in mid[:48])  # warmup
    assert abs(mid[60] - 10.0) < 1e-9  # flat EMA converges to price
    assert up[60] > mid[60] > lw[60]  # ATR > 0 widens channel
    assert abs(up[60] - mid[60] - (mid[60] - lw[60])) < 1e-9  # symmetric
    assert abs(wd[60] - 2.0 * (up[60] - mid[60])) < 1e-9
    # flat close inside channel (ATR wide) -> gate closed
    g = mod.gate_outside(flat, up, lw)
    assert all(v == 0.0 for v in g[:48])
    assert all(v == 0.0 for v in g)
    # breakout: last close far above channel -> open
    outs = [10.0] * 100
    hi2 = [10.5] * 100
    lo2 = [9.5] * 100
    outs2 = list(outs) + [50.0]
    hi3 = list(hi2) + [50.5]
    lo3 = list(lo2) + [49.5]
    m2, u2, l2, _ = mod.keltner(hi3, lo3, outs2)
    g2 = mod.gate_outside(outs2, u2, l2)
    assert g2[-1] == 1.0
    # ramp determinism: same input -> same bands
    m3, u3, l3, _ = mod.keltner(hi3, lo3, outs2)
    assert m2 == m3 and u2 == u3 and l2 == l3


def test_iter_h71_no_broker():
    src = SRC.read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h71_engine_mirror():
    src = SRC.read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec[",
                  "time_stop=int(spec[", "vol_window=int(spec[",
                  "keltner", "gate_outside", "KELT_WINDOW = 24", "KELT_MULT = 2.0"):
        assert token in src, token
    assert "short_enabled=True" in src
    assert "[3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]" in src


def test_iter_h71_smoke_runs_offline():
    env = dict(os.environ, ITER_H71_SMOKE="1",
               ITER_H71_OUT="results/iter_H71_kelt_smoke.json",
               ITER_H71_LOG="logs/iter_H71_kelt_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_h71_kelt.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("results/iter_H71_kelt_smoke.json").read_text())
    assert d["config"]["smoke"] is True
    assert d["config"]["kelt"]["window"] == 24
    assert set(d["coins"]) == {"ETC", "TRX"}
    for c in ("ETC", "TRX"):
        assert SEG <= set(d["coins"][c]["FULL"])
        assert isinstance(d["coins"][c]["FULL"]["trades"], int)
    assert d["verdict"] == "PENDING"
    assert d["status"] == "COMPLETE"
