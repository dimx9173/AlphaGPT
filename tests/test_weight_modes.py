"""P1-1 weight-mode tests: basket env switch + results schema. Recompute by script."""
import json, os, pathlib, subprocess, sys

def _load():
    p = pathlib.Path("results/weight_modes.json")
    assert p.exists(), "results/weight_modes.json missing; run research/run_weight_modes.py"
    return json.loads(p.read_text())

def test_weight_default_equal(monkeypatch):
    import strategy_manager.y1b_basket as B
    import importlib
    monkeypatch.delenv("Y1B_WEIGHT_MODE", raising=False)
    importlib.reload(B)
    assert B.weight_mode() == "equal"
    bars = {"ETC": [(1, 2, 0.5, 10.0)] * 80, "TRX": [(1, 2, 0.5, 10.0)] * 80}
    w, meta = B.weights_for_bars(bars)
    assert meta["mode"] == "equal" and abs(sum(w.values()) - 1.0) < 1e-9
    assert set(w) == {"ETC", "TRX"}

def test_invvol_clip_and_target(monkeypatch):
    import strategy_manager.y1b_basket as B
    import importlib
    monkeypatch.setenv("Y1B_WEIGHT_MODE", "invvol")
    importlib.reload(B)
    assert B.weight_mode() == "invvol"
    bars = {"HI": [], "LO": [], "M1": [], "M2": [], "M3": []}
    import math, random
    rnd = random.Random(7)
    px = {"HI": 100.0, "LO": 100.0, "M1": 100.0, "M2": 100.0, "M3": 100.0}
    sig = {"HI": 0.10, "LO": 0.01, "M1": 0.03, "M2": 0.03, "M3": 0.03}
    for i in range(80):
        for c in px:
            px[c] *= 1 + rnd.gauss(0, sig[c])
            bars[c].append((px[c], px[c] * 1.01, px[c] * 0.99, px[c], 10.0))
    w, meta = B.weights_for_bars(bars, mode="invvol")
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0.10 - 1e-9 <= v <= 0.35 + 1e-9 for v in w.values()), w
    assert w["LO"] > w["HI"], w
    monkeypatch.setenv("Y1B_VOL_TARGET", "0.35")
    importlib.reload(B)
    w2, m2 = B.weights_for_bars(bars, mode="invvol_cap", target=0.35)
    assert m2["mode"] == "invvol_cap"
    assert 0.25 - 1e-9 <= m2["lev_scale"] <= 2.0 + 1e-9
    monkeypatch.delenv("Y1B_WEIGHT_MODE", raising=False)
    monkeypatch.delenv("Y1B_VOL_TARGET", raising=False)
    importlib.reload(B)
    assert B.weight_mode() == "equal"

def test_weight_modes_schema():
    d = _load()
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "KEEP_equal"
    assert set(d["modes"]) == {"equal", "invvol", "invvol_cap"}
    for m, v in d["modes"].items():
        for k in ("FULL", "H2", "dd", "fee_grid", "fold12", "w_final", "lev_final"):
            assert k in v, (m, k)
        assert abs(sum(v["w_final"].values()) - 1.0) < 1e-3, (m, v["w_final"])
        assert all(0.10 - 1e-3 <= x <= 0.35 + 1e-3 for x in v["w_final"].values() if m != "equal"), (m, v["w_final"])
        assert len(v["fee_grid"]) == 4
        assert len(v["fold12"]["sharpes"]) == 12
        assert v["dd"]["dd_end"] >= v["dd"]["dd_start"]
        assert abs(sum(v["dd"]["by_coin"].values()) - (-v["dd"]["dd_depth"])) < 0.05, (m, v["dd"])
    fees = {(g["fee"], g["fund"]) for g in d["modes"]["equal"]["fee_grid"]}
    assert fees == {(0.0004, 0.0005), (0.0008, 0.0005), (0.0004, 0.001), (0.0008, 0.001)}

def test_weight_modes_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_weight_modes.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["modes"]) == {"equal", "invvol", "invvol_cap"}
