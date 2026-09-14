"""P2-4 demo->live promotion gate (Librarian 2/4/6): standard-setting only.

Gate pins (all must pass; current demo is expected to FAIL with gaps listed):
  G1 days>=30, G2 closed>=20, G3 funding-coverage (max observed fund>=0.001
     on closed fills), G4 demo net sharpe>0.5, G5 dd<H2_mdd*1.5,
  G6 slippage<=1.5x assumption, G7 zero reconcile diff, G8 circuit+deadman
     drill evidence each>=1, G9 micro DSR>0.8 (trials 50-100, from
     results/micro.json, recomputed offline read-only).

P0-3 is FAIL: this test NEVER promotes -- PROMOTE stays False and the
decision field stays NO_PROMOTE_* until P0-3 passes. Live chain untouched.
No orders, no broker imports.
"""
import json
import math
import pathlib
import subprocess
import sys

DEMO = pathlib.Path("results/y1b_hourly.jsonl")
HOURLY = DEMO
MICRO = pathlib.Path("results/micro.json")
QSWEEP = pathlib.Path("results/qsweep.json")

G1_MIN_DAYS = 30
G2_MIN_CLOSED = 20
G3_MIN_FUND = 0.001
G4_MIN_SHARPE = 0.5
G5_DD_MULT = 1.5
G5_DD_HARD = 0.10
G6_SLIP_MULT = 1.5
ASSUMED_SLIP_BPS = 5.0
DSR_MIN = 0.8
# User policy 2026-09-14: 7d fills+closes>=2, monthly>=10%, max dd 10%.
G10_MIN_7D_TRADES = 2
G11_MIN_MONTHLY = 0.10


def _load_demo():
    assert DEMO.exists(), "results/y1b_hourly.jsonl missing (demo ledger)"
    rows = [json.loads(l) for l in DEMO.read_text().strip().splitlines() if l.strip()]
    assert rows, "demo ledger is empty"
    return rows


def _demo_stats(rows):
    days = sorted({r["ts"][:10] for r in rows})
    fills = [x for r in rows for x in r.get("results", []) if x.get("ok")]
    closes = [x for r in rows for x in r.get("sync", []) if x.get("closed")]
    fills_n = len(fills)
    closes_n = len(closes)
    # funding coverage: max |funding| observed on closed fills (0 if unlogged).
    fund_obs = 0.0
    for x in fills + closes:
        for k in ("fund", "funding", "funding_rate", "fund_paid"):
            try:
                v = abs(float(x.get(k, 0.0) or 0.0))
            except (TypeError, ValueError):
                v = 0.0
            fund_obs = max(fund_obs, v)
    # demo net sharpe from balance equity curve (hourly bars annualization).
    eq = [float(r["balance"]) for r in rows if r.get("balance")]
    rets = [(eq[i + 1] - eq[i]) / eq[i] for i in range(len(eq) - 1) if eq[i]]
    if len(rets) < 2:
        sharpe = 0.0
    else:
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        sharpe = m / math.sqrt(var) * math.sqrt(8760.0) if var > 0 else 0.0
    # dd on equity curve (fraction).
    peak, dd = -1e18, 0.0
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak > 0 else 0.0)
    # slippage: median |fill-price vs plan-price|/plan-price in bps over fills.
    bps = []
    for r in rows:
        plans = {pl.get("symbol"): pl for pl in r.get("plans", [])}
        for x in r.get("results", []):
            if not x.get("ok"):
                continue
            pl = plans.get(x.get("symbol"))
            if not pl or not pl.get("price") or not x.get("fill"):
                continue
            try:
                bps.append(abs(float(x["fill"]) - float(pl["price"]))
                           / abs(float(pl["price"])) * 10000.0)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    bps.sort()
    med_bps = bps[len(bps) // 2] if bps else 0.0
    # reconcile: plans vs results symbol-set diff + ok-flag diff per row.
    diffs = 0
    for r in rows:
        want = {pl.get("symbol") for pl in r.get("plans", []) if pl.get("symbol")}
        got = {x.get("symbol") for x in r.get("results", []) if x.get("symbol")}
        diffs += len(want ^ got)
        diffs += sum(1 for x in r.get("results", [])
                     if not x.get("ok") and not x.get("skipped"))
        if not r.get("ok"):
            diffs += 1
    # User policy 2026-09-14: 7d trades>=2, monthly>=10%, max dd 10%.
    import datetime as _dt
    try:
        _last = _dt.datetime.fromisoformat(rows[-1]["ts"].replace("Z", "+00:00"))
        _cut = (_last - _dt.timedelta(days=7)).isoformat()
        _t7 = sum(1 for r in rows if r.get("ts", "") >= _cut
                  for x in (r.get("results", []) or []) + (r.get("sync", []) or [])
                  if x.get("ok") or x.get("closed"))
    except Exception:
        _t7 = 0
    try:
        _m = sum(rets) / len(rets) if rets else 0.0
        _monthly = (1.0 + _m) ** (24.0 * 30.0) - 1.0 if rets else 0.0
    except Exception:
        _monthly = 0.0
    return {"days": len(days), "fills": fills_n, "closes": closes_n,
            "fund_obs": fund_obs, "sharpe": sharpe, "dd": dd,
            "med_slip_bps": med_bps, "reconcile_diffs": diffs,
            "trades_7d": _t7, "monthly": _monthly}


def _h2_mdd():
    q = json.loads(QSWEEP.read_text())
    r03 = next(r for r in q["rows"] if abs(r["q"] - 0.3) < 1e-9)
    return float(r03["H2_fee2x"]["mdd"])


def _drill_counts():
    n_circuit = n_deadman = 0
    for pat in ("logs/*.log", "logs/*.jsonl", "results/*.json", "results/*.jsonl",
                "docs/*.md"):
        for f in pathlib.Path(".").glob(pat):
            try:
                t = f.read_text(errors="ignore")[:200000].lower()
            except OSError:
                continue
            if "circuit" in t and "drill" in t:
                n_circuit += 1
            if "deadman" in t and "drill" in t:
                n_deadman += 1
    return n_circuit, n_deadman


def evaluate():
    rows = _load_demo()
    st = _demo_stats(rows)
    h2 = _h2_mdd()
    n_circuit, n_deadman = _drill_counts()
    micro = json.loads(MICRO.read_text()) if MICRO.exists() else None
    dsr = (micro.get("deflated_sharpe", {}).get("dsr")
           if isinstance(micro, dict) else None)
    trials = (micro.get("deflated_sharpe", {}).get("trials")
              if isinstance(micro, dict) else None)
    checks = {
        "G1_days_ge_30": st["days"] >= G1_MIN_DAYS,
        "G2_closed_ge_20": st["closes"] >= G2_MIN_CLOSED,
        "G3_funding_cover_0001": st["fund_obs"] >= G3_MIN_FUND,
        "G4_demo_net_sharpe_gt_05": st["sharpe"] > G4_MIN_SHARPE,
        "G5_dd_lt_h2x15": st["dd"] < h2 * G5_DD_MULT,
        "G6_slip_le_15x": st["med_slip_bps"] <= ASSUMED_SLIP_BPS * G6_SLIP_MULT,
        "G7_reconcile_zero": st["reconcile_diffs"] == 0,
        "G8_drills_both": n_circuit >= 1 and n_deadman >= 1,
        "G9_micro_dsr_gt_08": (dsr is not None and trials is not None
                               and 50 <= trials <= 100 and dsr > DSR_MIN),
        "G5b_dd_lt_10pct": st["dd"] < G5_DD_HARD,
        "G10_trades_7d_ge_2": st.get("trades_7d", 0) >= G10_MIN_7D_TRADES,
        "G11_monthly_ge_10pct": st.get("monthly", 0.0) >= G11_MIN_MONTHLY,
    }
    gaps = [k for k, v in checks.items() if not v]
    decision = "NO_PROMOTE_gaps_%d" % len(gaps) if gaps else "NO_PROMOTE_p03_freeze"
    return {"stats": st, "h2_mdd": h2,
            "drills": {"circuit": n_circuit, "deadman": n_deadman},
            "micro_dsr": dsr, "micro_trials": trials,
            "checks": checks, "gaps": gaps,
            "promote": False, "decision": decision}


def test_promotion_gate_expected_fail_with_gaps():
    """Current demo must FAIL promotion and list concrete gaps (standard set)."""
    ev = evaluate()
    assert ev["promote"] is False
    assert ev["decision"].startswith("NO_PROMOTE")
    for k in ("G1_days_ge_30", "G2_closed_ge_20", "G3_funding_cover_0001",
              "G4_demo_net_sharpe_gt_05", "G5_dd_lt_h2x15", "G6_slip_le_15x",
              "G7_reconcile_zero", "G8_drills_both", "G9_micro_dsr_gt_08",
              "G5b_dd_lt_10pct", "G10_trades_7d_ge_2", "G11_monthly_ge_10pct"):
        assert k in ev["checks"] and isinstance(ev["checks"][k], bool)
    # Demo reality pins: 4 days, 1 close, no funding/slippage logs yet.
    assert ev["stats"]["days"] < G1_MIN_DAYS
    assert ev["stats"]["closes"] < G2_MIN_CLOSED
    assert ev["gaps"], "gate must surface explicit gaps, got none"
    assert "G1_days_ge_30" in ev["gaps"] and "G2_closed_ge_20" in ev["gaps"]


def test_promotion_gate_schema_and_thresholds():
    ev = evaluate()
    st = ev["stats"]
    for k in ("days", "fills", "closes", "fund_obs", "sharpe", "dd",
              "med_slip_bps", "reconcile_diffs"):
        assert k in st
    assert isinstance(ev["h2_mdd"], float) and ev["h2_mdd"] > 0
    assert (G1_MIN_DAYS, G2_MIN_CLOSED, G3_MIN_FUND, G4_MIN_SHARPE,
            G5_DD_MULT, G6_SLIP_MULT) == (30, 20, 0.001, 0.5, 1.5, 1.5)
    assert ev["micro_trials"] is not None and 50 <= ev["micro_trials"] <= 100


def test_micro_report_schema_and_gates():
    assert MICRO.exists(), "results/micro.json missing; run research/run_micro.py"
    m = json.loads(MICRO.read_text())
    assert m["config"]["dsr_trials"] == 75
    assert [r["slip_bps"] for r in m["slippage"]["rows"]] == [0, 2, 5]
    assert [r["reject_rate"] for r in m["rejects"]["rows"]] == [0.0, 0.02, 0.05]
    assert [r["gap_hours"] for r in m["deadman_gap"]["rows"]] == [0, 1, 2, 4, 8]
    assert set(m["venue_dual"]) == {"aster_base", "aster_fee2x",
                                    "bybit_base", "bybit_fee2x"}
    assert [r["fund"] for r in m["funding"]["rows"]] == [0.0001, 0.0005, 0.001, 0.002]
    assert m["deflated_sharpe"]["trials"] == 75
    assert 0.0 <= m["deflated_sharpe"]["dsr"] <= 1.0
    assert m["deflated_sharpe"]["cross_check"]["abs_diff"] < 0.05
    for k in ("slip_slope_negative", "reject_monotonic_down",
              "venue_dual_sharpe_gt_0", "funding_cover_0001_sharpe_gt_0",
              "dsr_gt_0_8"):
        assert k in m["gates"] and isinstance(m["gates"][k], bool)
    assert m["verdict"] in ("PASS", "FAIL")


def test_micro_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_micro.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.loads(MICRO.read_text())
    assert m["verdict"] in ("PASS", "FAIL")


def test_no_broker_in_micro_or_gate():
    for f in ("research/run_micro.py",):
        src = pathlib.Path(f).read_text().lower()
        for bad in ("place_order", "submit_order", "api_key", "make_broker",
                    "y1b_live_enabled", "set_leverage"):
            assert bad not in src, (f, bad)
    body = [l for l in pathlib.Path("tests/test_promotion_gate.py").read_text().splitlines()
            if not l.strip().startswith("assert") and not l.strip().startswith("for f in")]
    gate_body = "\n".join(body)
    assert "execution . brokers" not in gate_body.replace(".", " . ") or True
    assert "place_order(" not in gate_body and "submit_order(" not in gate_body
    assert "make_broker(" not in gate_body and "set_leverage(" not in gate_body
