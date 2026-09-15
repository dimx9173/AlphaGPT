# H16 1h tail-risk tests: lock + schema + helpers + smoke.
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
OUT = pathlib.Path(os.getenv("ITER_H16_OUT", "results/iter_H16_tail.json"))
COINS5 = ["ETC", "TRX", "ATOM", "APT", "KAS"]
TAIL_FIELDS = {"n", "mean", "sd", "sharpe", "cum", "skew", "kurt_excess", "var5_ret", "var5_loss", "es5_ret", "es5_loss", "es5_tail_n", "var1_ret", "var1_loss", "es1_ret", "es1_loss", "es1_tail_n", "worst_day"}
WD_FIELDS = {"day_idx", "start_bar", "end_bar", "day_net", "n_days", "leftover_bars", "start_ts"}

def _load():
    assert OUT.exists(), "results/iter_H16_tail.json missing; run research/run_iter_h16_tail.py"
    return json.loads(OUT.read_text())

def _mod():
    spec = importlib.util.spec_from_file_location("h16mod", "research/run_iter_h16_tail.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h16mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m

def test_iter_h16_lock():
    from strategy_manager.config import LEV, FEE, FUND, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)

def test_iter_h16_helpers():
    m = _mod()
    assert m.skew_kurt([0.0, 0.0, 0.0, 0.0]) == (0.0, 0.0)
    assert m.skew_kurt([1.0]) == (0.0, 0.0)
    sk, ku = m.skew_kurt([float(x) for x in range(1, 11)])
    assert abs(sk) < 1e-12
    assert abs(ku + 1.2242) < 0.01
    v = m.var_es([1.0, 2.0, 3.0, 4.0, 5.0], 0.2)
    assert v["var_ret"] == 1.0 and v["var_loss"] == -1.0
    assert v["es_ret"] == 1.0 and v["tail_n"] == 1 and v["cut_rank"] == 1
    v2 = m.var_es([-3.0, -1.0, 0.0, 2.0], 0.5)
    assert v2["var_ret"] == -1.0 and v2["es_ret"] == -2.0 and v2["tail_n"] == 2
    w = m.worst_day([1.0, 1.0, -5.0, -5.0, 3.0, 3.0], day_bars=2)
    assert (w["day_idx"], w["start_bar"], w["end_bar"]) == (1, 2, 4)
    assert w["day_net"] == -10.0 and w["n_days"] == 3 and w["leftover_bars"] == 0
    w2 = m.worst_day([0.5] * 5, day_bars=2)
    assert w2["n_days"] == 2 and w2["leftover_bars"] == 1
    t = m.leg_tail([0.01, -0.02, 0.03, -0.005, 0.012] * 20)
    assert t["n"] == 100
    assert t["var5_ret"] <= 0 and t["es5_ret"] <= t["var5_ret"] + 1e-12
    assert t["var1_ret"] <= t["var5_ret"] + 1e-12 and t["es1_ret"] <= t["var1_ret"] + 1e-12
    assert t["worst_day"]["n_days"] == 100 // 24

def test_iter_h16_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h" and cfg["bpy"] == 8760.0 and cfg["scale_4h_to_1h"] == 4
    assert cfg["day_bars"] == 24 and cfg["smoke"] is False
    assert cfg["coins"] == COINS5 and set(cfg["weights"]) == set(COINS5)
    assert all(abs(w - 0.2) < 1e-12 for w in cfg["weights"].values())
    for c in COINS5:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None
            else:
                assert sc[k] == int(base[k]) * 4
    n = cfg["grid_bars"]
    assert n > 2000
    assert set(d["per_coin"]) == set(COINS5)
    for c in COINS5:
        pc = d["per_coin"][c]
        assert TAIL_FIELDS <= set(pc), (c, sorted(set(pc)))
        assert pc["n"] == n
        assert math.isfinite(pc["skew"]) and math.isfinite(pc["kurt_excess"])
        assert pc["var5_loss"] == round(-pc["var5_ret"], 6)
        assert pc["es5_loss"] == round(-pc["es5_ret"], 6)
        assert pc["var1_loss"] == round(-pc["var1_ret"], 6)
        assert pc["es1_loss"] == round(-pc["es1_ret"], 6)
        assert pc["es5_ret"] <= pc["var5_ret"] + 1e-9
        assert pc["es1_ret"] <= pc["var1_ret"] + 1e-9
        assert pc["var1_ret"] <= pc["var5_ret"] + 1e-9
        assert pc["es5_tail_n"] >= math.ceil(0.05 * n) and pc["es1_tail_n"] >= math.ceil(0.01 * n)
        wd = pc["worst_day"]
        assert WD_FIELDS <= set(wd), (c, sorted(set(wd)))
        assert wd["end_bar"] - wd["start_bar"] == 24
        assert wd["start_bar"] == wd["day_idx"] * 24
        assert wd["n_days"] == n // 24 and wd["leftover_bars"] == n % 24
        assert wd["start_bar"] >= 0 and wd["end_bar"] <= n
    b = d["basket"]
    assert TAIL_FIELDS <= set(b)
    assert b["n"] == n
    assert b["es5_ret"] <= b["var5_ret"] + 1e-9 and b["es1_ret"] <= b["var1_ret"] + 1e-9
    wa = b["worst_day_attrib"]
    assert wa["end_bar"] - wa["start_bar"] == 24
    assert wa["start_bar"] == wa["day_idx"] * 24
    assert set(wa["contrib"]) == set(COINS5) and set(wa["share"]) == set(COINS5)
    assert abs(sum(wa["contrib"].values()) - wa["day_net"]) < 2e-6
    assert abs(wa["day_net"] - b["worst_day"]["day_net"]) < 2e-6
    assert wa["day_idx"] == b["worst_day"]["day_idx"]
    assert abs(sum(wa["share"].values()) - 1.0) < 0.01 or wa["day_net"] == 0.0
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]

def test_iter_h16_no_broker():
    src = pathlib.Path("research/run_iter_h16_tail.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad

def test_iter_h16_smoke_runs_offline(tmp_path):
    outp = tmp_path / "iter_H16_tail.json"
    lg = tmp_path / "iter_H16_tail.log"
    env = dict(os.environ, ITER_H16_SMOKE="1", ITER_H16_OUT=str(outp), ITER_H16_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h16_tail.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(outp.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 480
    assert dd["basket"]["n"] == 480
    assert dd["basket"]["worst_day"]["n_days"] == 480 // 24
    assert dd["verdict"] == "PENDING_P03_FAIL"
