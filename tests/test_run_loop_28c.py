"""The loop must not be able to proceed on a verdict it did not earn.

These are the properties that make the loop worth having. Each one is a
failure that was either observed during construction or is the exact shape of
a bug this project has already produced four times: the pipeline runs to
completion, every check passes, and the number it reports is the wrong one.

    - a moved null control (tests/test_frozen_null_control_28c.py)
    - a silent grammar routing trap (tests/test_api_traps_28c.py)
    - Sharpe measured against zero instead of a holdable position
      (tests/test_buy_and_hold_benchmark_28c.py)
    - a gate verdict produced by a tool that could not start
      (run_gate_report.py, fixed in this cycle)
    - a verdict scraped from truncated stdout instead of parsed
      (run_loop_28c.py, fixed in this cycle)

The loop is the place where all of these would matter most, because it is the
thing that decides whether something gets traded.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "research" / "run_loop_28c.py"
PY = sys.executable


def source():
    return LOOP.read_text()


def test_the_loop_parses():
    ast.parse(source())


def test_live_is_refused_with_a_reason_not_silently_ignored():
    """A missing flag is not a refusal. Asking must produce an answer."""
    r = subprocess.run([PY, str(LOOP), "--live"], capture_output=True,
                       text=True, cwd=ROOT, timeout=300)
    assert r.returncode != 0
    low = r.stdout.lower()
    assert "refus" in low
    # and it must name the three things that are forbidden
    for word in ("account", "key", "order"):
        assert word in low


def test_dry_run_executes_nothing():
    """A plan that runs the plan is not a plan."""
    r = subprocess.run([PY, str(LOOP), "--dry-run", "--with-null",
                        "--cycles", "3"],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0
    assert "dry run" in r.stdout.lower()
    before = set((ROOT / "results").glob("loop_*"))
    r2 = subprocess.run([PY, str(LOOP), "--dry-run"],
                        capture_output=True, text=True, cwd=ROOT, timeout=300)
    after = set((ROOT / "results").glob("loop_*"))
    assert before == after, "a dry run created artifacts"


def test_the_acceptance_criterion_is_buy_and_hold_not_zero():
    """The whole point. If this ever reads zero, the loop is a drift machine."""
    s = source()
    assert "excess_sharpe_vs_buy_and_hold" in s
    # the comparison that admits a strategy
    assert '"beats_buy_and_hold"' in s
    assert "rec[\"excess\"] > 0.0" in s


def test_test_verdict_is_parsed_from_junit_not_scraped_from_stdout():
    """Text-scraping a truncated stream is how a clean run reported FAIL."""
    s = source()
    assert "junitxml" in s
    assert "ET.parse" in s
    # The old expression survives in a comment that explains why it was
    # removed, so matching the bare text would fail on the explanation. The
    # check that matters is that no *code* line uses it.
    code_lines = [ln for ln in s.splitlines()
                  if '"passed" in out' in ln and not ln.strip().startswith("#")]
    assert not code_lines, (
        "the loop went back to inferring the test verdict from stdout text")
    # zero collected is a failure, not a pass
    assert 'stats["tests"] > 0' in s


def test_stages_use_the_running_interpreter_not_bare_python3():
    s = source()
    for bad in ('sh("python3 ', 'sh(f"python3 ', '"python3 -m pytest'):
        assert bad not in s, f"still shells out to a bare interpreter: {bad}"
    assert "{PY}" in s


def test_the_loop_refuses_to_start_without_its_dependencies():
    assert "def require(" in source()
    assert "cannot import" in source()


def test_the_in_scope_test_list_only_names_files_that_exist():
    """A stale path in the scope list would silently shrink the gate."""
    import re
    names = re.findall(r'"(tests/test_[a-z0-9_]+\.py)"', source())
    assert len(names) >= 15
    missing = [n for n in names if not (ROOT / n).exists()]
    assert not missing, f"in-scope tests that do not exist: {missing}"


def test_the_loop_runs_the_real_tools_rather_than_reimplementing_them():
    """Each stage shells out, so the loop cannot drift from what it drives."""
    s = source()
    assert "research/ga_28c_30m_3y.py" in s
    assert "research/run_paper_28c_pit_30m.py" in s


def test_the_record_states_the_paper_only_policy():
    s = source()
    assert "paper-only" in s
    assert "broker-free" in s


def test_a_real_report_has_the_shape_the_skill_documents():
    """Guard the doc against the code drifting out from under it."""
    # The loop's report and the GA's artifacts share a prefix, so filter on
    # the shape instead of the filename. Sorting by name picked a GA artifact
    # the first time, which is its own small version of this bug.
    reports = []
    for f in (ROOT / "results").glob("loop_*.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "run_id" in d and "stages" in d:
            reports.append(d)
    if not reports:
        pytest.skip("no loop report yet")
    rep = reports[-1]
    for key in ("run_id", "stages", "accepted", "reason", "policy"):
        assert key in rep, f"report is missing {key!r}"
    assert "paper-only" in rep["policy"]
    # every stage records its command and its exit code
    for st in rep["stages"]:
        assert "rc" in st and "stage" in st
    # if it was not accepted, the reason must say why in words
    if not rep["accepted"]:
        assert rep["reason"], "a halted loop must say why"
