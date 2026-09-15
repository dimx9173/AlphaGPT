"""H11 per-coin sl x ts x cd-small joint grid tests (1h native, Top5).

Full H11 is 18 combos/coin (sl 3 x ts 2 x cd 2) x 5 coins = 90 per-coin
rows + 18 uniform rows (smoke: coins {ETC,TRX}, sl {None,0.05} x ts {24}
x cd {6} -> 2 per-coin/coin + 2 uniform, to a temp OUT so the committed
FULL artifact is not clobbered). This test never re-runs the full sweep:
schema checks accept either artifact, and the live subprocess check runs
smoke mode only. Conclusion must stay PENDING (\u5f85\u5b9a): diagnostic
only, no adoption, live untouched.
"""
import itertools
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_H11_combo.json")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
BASE = {
    "ETC": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3},
    "TRX": {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3},
    "ATOM": {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3},
    "APT": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3},
    "KAS": {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24, "q": 0.3},
}
FULL_SL = [None, 0.03, 0.05]
FULL_TS = [12, 24]
FULL_CD = [3, 6]


def _load():
    assert OUT.exists(), "results/iter_H11_combo.json missing; run research/run_iter_h11_combo.py"
    return json.loads(OUT.read_text())


def _expected(d):
    smoke = d["config"]["smoke"]
    coins = {"ETC", "TRX"} if smoke else set(COINS)
    sl = [None, 0.05] if smoke else list(FULL_SL)
    ts = [24] if smoke else list(FULL_TS)
    cd = [6] if smoke else list(FULL_CD)
    return coins, {(s, t, c) for s in sl for t in ts for c in cd}


def _check_row(r, base_full):
    assert SEG <= set(r["FULL"]), r
    assert r["FULL"]["n"] > 8000
    assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - base_full["sharpe"], 3)) < 1e-9
    assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - base_full["mdd"], 4)) < 1e-9


def _check_best(best, rows):
    assert best is not None
    top = max(r["FULL"]["sharpe"] for r in rows)
    tied = [r for r in rows if r["FULL"]["sharpe"] == top]
    exp = min(tied, key=lambda r: r["FULL"]["mdd"])
    for k in ("sl", "ts", "cd"):
        assert best[k] == exp[k], (k, best[k], exp[k])


def test_iter_h11_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["grid_bars"] > 8000
    assert set(cfg["weights"]) == COINS
    for c, spec in BASE.items():
        assert cfg["basket"][c] == spec, c
    coins, combos = _expected(d)
    assert cfg["per_coin_cells"] == len(combos)
    assert cfg["uniform_cells"] == len(combos)
    got_pc = {(r["coin"], r["sl"], r["ts"], r["cd"]) for r in d["per_coin_rows"]}
    assert got_pc == {(c, s, t, k) for c in coins for (s, t, k) in combos}, (
        len(got_pc), len(coins) * len(combos))
    assert {(r["sl"], r["ts"], r["cd"]) for r in d["uniform_rows"]} == combos
    assert SEG <= set(d["base_FULL"])
    assert d["base_FULL"]["n"] == cfg["grid_bars"]
    for r in d["per_coin_rows"]:
        assert r["coin"] in coins
        _check_row(r, d["base_FULL"])
    for r in d["uniform_rows"]:
        _check_row(r, d["base_FULL"])
    for c in coins:
        rows_c = [r for r in d["per_coin_rows"] if r["coin"] == c]
        assert len(rows_c) == len(combos)
        _check_best(d["per_coin_best"][c], rows_c)
    _check_best(d["uniform_best"], d["uniform_rows"])
    assert d["verdict"] == "PENDING"
    assert "\u5f85\u5b9a" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h11_no_broker():
    src = pathlib.Path("research/run_iter_h11_combo.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h11_engine_mirror():
    src = pathlib.Path("research/run_iter_h11_combo.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec[\"cd\"]) * SCALE",
                  "time_stop=int(spec[\"ts\"]) * SCALE",
                  "vol_window=int(spec[\"vw\"]) * SCALE",
                  "data/data_1y/1h"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_h11_smoke_runs_offline():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H11_combo.json")
        log = os.path.join(td, "iter_h11_combo.log")
        env = dict(os.environ, ITER_H11_SMOKE="1", ITER_H11_OUT=out, ITER_H11_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_h11_combo.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(pathlib.Path(out).read_text())
        assert dd["config"]["smoke"] is True
        assert len(dd["per_coin_rows"]) == 4
        assert {(x["coin"], x["sl"], x["ts"], x["cd"]) for x in dd["per_coin_rows"]} == {
            ("ETC", None, 24, 6), ("ETC", 0.05, 24, 6),
            ("TRX", None, 24, 6), ("TRX", 0.05, 24, 6)}
        assert len(dd["uniform_rows"]) == 2
        assert dd["verdict"] == "PENDING"
