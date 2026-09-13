"""P1-3 q-sweep test: light schema check, recompute done by the script."""
import json
import pathlib
import subprocess
import sys

QS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
REQ = {"q", "FULL", "H2", "FULL_fee2x", "H2_fee2x", "attribution_full",
       "attribution_h2", "bull", "bear"}
SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ATTR = {"long_entries", "short_entries", "long_pnl", "short_pnl",
        "long_share", "short_share"}
BB = {"sharpe", "n"}


def _load():
    p = pathlib.Path("results/qsweep.json")
    assert p.exists(), "results/qsweep.json missing; run research/run_qsweep.py"
    return json.loads(p.read_text())


def test_qsweep_schema():
    d = _load()
    assert d["config"]["qs"] == QS
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["rows"]
    assert [r["q"] for r in rows] == QS
    for r in rows:
        assert REQ <= set(r), r["q"]
        for k in ("FULL", "H2", "FULL_fee2x", "H2_fee2x"):
            assert SEG <= set(r[k]), (r["q"], k)
        for k in ("attribution_full", "attribution_h2"):
            assert ATTR <= set(r[k]), (r["q"], k)
        for k in ("bull", "bear"):
            assert BB <= set(r[k]), (r["q"], k)
        a = r["attribution_full"]
        assert a["long_entries"] + a["short_entries"] > 0
        assert r["bull"]["n"] + r["bear"]["n"] == r["FULL"]["n"]
        assert r["H2"]["n"] == d["config"]["h2_len"]
    assert len(d["fee2x_curve"]) == 6
    assert d["knee"]["q"] in QS
    assert d["q03_check"]["isolated_peak"] in (True, False)
    assert d["decision"] in ("KEEP_q0.3", "ADOPT_KNEE_q%.1f" % d["knee"]["q"])


def test_qsweep_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_qsweep.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert [x["q"] for x in d["rows"]] == QS
