#!/usr/bin/env python3
"""run_y1b_verify.py — Y1b wiring verification (no orders, no keys).

Runs four dry checks:
  1. shadow dry-run: mock broker, plans gated, market_open zero-call
  2. STOP-block: STOP file + Y1B_LIVE_ENABLED=1 + dry_run=False -> blocked
  3. paper baseline: research/run_paper2.py must print trades=478 and final_x within 2.98+-0.15
  4. fee2x: FEE=0.0008 must print trades=478 and final_x>1, baseline restored
Usage: python3 research/run_y1b_verify.py
"""
import os
import sys
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.conftest  # noqa: F401,E402  (solders stub)
import asyncio  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

def check_shadow():
    for k in ["Y1B_LIVE_ENABLED", "PAPER_MODE"]:
        os.environ.pop(k, None)
    os.environ["Y1B_STATE"] = "/tmp/y1b_verify.json"
    from strategy_manager.y1b_executor import run_once
    async def main():
        b = AsyncMock()
        b.get_price = AsyncMock(return_value=20.0)
        b.get_position = AsyncMock(return_value=None)
        b.enable_deadman = AsyncMock(return_value=True)
        b.market_open = AsyncMock(side_effect=AssertionError("no live orders"))
        plans, res, sync = await run_once(broker=b, notional=50.0)
        assert {p.coin for p in plans} == {"ETC", "TRX"}
        assert all(r.get("dry_run") for r in res)
        b.market_open.assert_not_awaited()
    asyncio.run(main())
    print("[1/4] shadow dry-run OK")

def check_stop():
    from pathlib import Path
    p = Path("/tmp/STOP_Y1B_VERIFY")
    p.write_text("STOP")
    os.environ["STOP_SIGNAL_PATH"] = str(p)
    os.environ["Y1B_LIVE_ENABLED"] = "1"
    os.environ.pop("PAPER_MODE", None)
    os.environ["Y1B_STATE"] = "/tmp/y1b_s_verify.json"
    from strategy_manager.y1b_executor import run_once
    async def main():
        b = AsyncMock()
        b.get_price = AsyncMock(return_value=20.0)
        b.get_position = AsyncMock(return_value=None)
        b.enable_deadman = AsyncMock(return_value=True)
        b.market_open = AsyncMock(side_effect=AssertionError("blocked"))
        plans, res, sync = await run_once(broker=b, notional=50.0, dry_run=False)
        assert res and res[0].get("blocked") is True
        b.market_open.assert_not_awaited()
    asyncio.run(main())
    print("[2/4] STOP-block OK")

def check_paper():
    r = subprocess.run([sys.executable, "research/run_paper2.py"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    import re
    assert "trades=478" in out, out[-500:]
    m = re.search(r"final_x=([0-9.]+)", out)
    assert m and abs(float(m.group(1)) - 2.98) < 0.15, out[-500:]
    print(f"[3/4] paper baseline OK (478/{m.group(1)} within 2.98+-0.15)")

def check_fee2x():
    import re
    import shutil
    src = "research/run_paper2.py"
    tmp = "research/_fee2x_verify_tmp.py"
    shutil.copy(src, tmp)
    try:
        txt = open(tmp).read().replace("FEE = 0.0004", "FEE = 0.0008", 1)
        assert "FEE = 0.0008" in txt
        open(tmp, "w").write(txt)
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
        out = r.stdout + r.stderr
        assert "trades=478" in out, out[-500:]
        m2 = re.search(r"final_x=([0-9.]+)", out)
        assert m2 and float(m2.group(1)) > 1.0, out[-500:]
        print(f"[4/4] fee2x OK (478/{m2.group(1)} final_x>1)")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        subprocess.run([sys.executable, "research/run_paper2.py"], capture_output=True)

if __name__ == "__main__":
    check_shadow()
    check_stop()
    check_paper()
    check_fee2x()
    print("Y1B_VERIFY PASS: no orders, no keys")
