"""H92 1h CCI v2 filter tests (Top5): entry only CCI(48)>+100 long / CCI(48)<-100 short.

Schema accepts full or smoke artifact. Recompute-by-script only via
smoke subprocess; never re-runs the full sweep here. Verdict PENDING
(P0-3 FAIL), no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H92_cci.json")
SRC = pathlib.Path("research/run_iter_h92_cci.py")
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
    assert OUT.exists(), "results/iter_H92_cci.json missing; run research/run_iter_h92_cci.py"
    return json.loads(OUT.read_text())


def test_iter_h92_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["cci"]["window"] == 48
    assert cfg["cci"]["long_threshold"] == 100.0
    assert cfg["cci"]["short_threshold"] == -100.0
    assert cfg["cci"]["long_gate"] == "CCI>+100"
    assert cfg["cci"]["short_gate"] == "CCI<-100"
    assert cfg["cci"]["warmup_bars"] == 48
    assert cfg["lineage"] == "v2 of H72 CCI(24): window 48 (48h lookback), same +-100 entry-only thresholds"
    coins = cfg.get("coins", COINS)
    assert set(coins) <= set(COINS)
    assert len(coins) >= 2
    assert cfg["grid_bars"] >= (2500 if cfg["smoke"] else 8000)
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
        assert isinstance(cell["LONG"]["trades"], int), c
        assert isinstance(cell["SHORT"]["trades"], int), c
        assert cell["FULL"]["trades"] >= 0, c
        cc = cell["cci"]
        assert 0.0 <= cc["frac_long_open"] <= 1.0, c
        assert 0.0 <= cc["frac_short_open"] <= 1.0, c
        assert cc["n_long_open"] == round(cc["frac_long_open"] * cfg["grid_bars"]), c
        assert cc["n_short_open"] == round(cc["frac_short_open"] * cfg["grid_bars"]), c
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


def test_iter_h92_additivity():
    d = _load()
    coins = d["config"].get("coins", COINS)
    for c in coins:
        f = d["coins"][c]["FULL"]["cum"]
        lv = d["coins"][c]["LONG"]["cum"]
        sv = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (lv + sv)) < 1e-3, (c, f, lv, sv)
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3


def test_iter_h92_cci_math():
    """classic_cci(48): warmup neutral, flat=0, ramp-up overbought, ramp-down oversold."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("h92cci", str(SRC))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h92cci"] = mod
    spec.loader.exec_module(mod)
    flat = [(1.0, 10.0, 10.0, 10.0, 1.0)] * 200
    c = mod.classic_cci(flat)
    assert all(v == 0.0 for v in c[:48])
    assert all(v == 0.0 for v in c[48:])
    ramp = [(1.0, float(i), float(i), float(i), 1.0) for i in range(1, 201)]
    c2 = mod.classic_cci(ramp)
    assert all(v == 0.0 for v in c2[:48])
    assert c2[48] > 100.0 and c2[-1] > 100.0  # steady climb pins CCI overbought
    dn = [(1.0, float(201 - i), float(201 - i), float(201 - i), 1.0) for i in range(1, 201)]
    c3 = mod.classic_cci(dn)
    assert all(v == 0.0 for v in c3[:48])
    assert c3[48] < -100.0 and c3[-1] < -100.0  # steady fall pins CCI oversold


def test_iter_h92_engine_mirror():
    src = SRC.read_text()
    for token in ("quantile", "cooldown", "_apply_cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec[",
                  "time_stop=int(spec[", "vol_window=int(spec[",
                  "classic_cci", "CCI_WINDOW = 48", "CCI_LONG", "CCI_SHORT"):
        assert token in src, token
    assert "short_enabled=True" in src
    assert "[3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]" in src


def test_iter_h92_no_broker():
    src = SRC.read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h92_smoke_runs_offline():
    env = dict(os.environ, ITER_H92_SMOKE="1",
               ITER_H92_OUT="results/iter_H92_cci_smoke.json",
               ITER_H92_LOG="logs/iter_H92_cci_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_h92_cci.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("results/iter_H92_cci_smoke.json").read_text())
    assert d["config"]["smoke"] is True
    assert d["config"]["cci"]["window"] == 48
    assert set(d["coins"]) == {"ETC", "TRX"}
    for c in ("ETC", "TRX"):
        assert SEG <= set(d["coins"][c]["FULL"])
        assert isinstance(d["coins"][c]["FULL"]["trades"], int)
    assert d["verdict"] == "PENDING"
    assert d["status"] == "COMPLETE"
