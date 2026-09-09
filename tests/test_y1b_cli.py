"""Y1b once-CLI tests: refusal without confirm flag, dry-run by default."""
import json
import subprocess
import sys


def _run(*args):
    r = subprocess.run([sys.executable, "research/run_y1b_once.py", *args],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads(r.stdout)


def test_live_without_confirm_refused():
    d = _run("--live")
    assert d.get("refused") is True


def test_default_is_dry_run():
    d = _run()
    assert all(x.get("dry_run") for x in d["results"])
    assert {p["coin"] for p in d["plans"]} == {"ETC", "TRX"}
