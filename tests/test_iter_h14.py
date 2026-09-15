"""H14 realized-funding (1h native, Top5) tests.

Reads results/iter_H14_fundreal.json (produced by
research/run_iter_h14_fundreal.py). Verdict stays PENDING (P0-3 FAIL);
no adoption. Venue reads are public-only Bybit demo (no keys/orders).
"""
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H14_OUT", "results/iter_H14_fundreal.json"))
COINS5 = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SMOKE_COINS = ["ETC", "TRX"]


def _load():
    assert OUT.exists(), "results/iter_H14_fundreal.json missing; run research/run_iter_h14_fundreal.py"
    return json.loads(OUT.read_text())


def test_iter_h14_locks_and_config():
    from strategy_manager.config import LEV, FUND, FEE, FEE2X, FORMULA
    from strategy_manager.config import LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE2X - 2 * FEE) < 1e-12
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"], LOCKED_ETC["q"]) == (0.88, 0.12, 18, None, 24, 0.3)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"], LOCKED_TRX["q"]) == (0.85, 0.12, 6, 0.05, 24, 0.3)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"], LOCKED_ATOM["q"]) == (0.85, 0.15, 6, 0.05, 24, 0.3)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"], LOCKED_APT["q"]) == (0.88, 0.12, 18, None, 24, 0.3)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"], LOCKED_KAS["q"]) == (0.88, 0.12, 6, None, 24, 0.3)
    d = _load()
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["lev"] == 2.0
    assert abs(cfg["fund_assumed"] - 0.0005) < 1e-12
    assert cfg["venue"] == "aster"
    assert "bybit-demo" in cfg["live_venue_reads"]
    assert cfg["bybit_demo_base"] == "https://api-demo.bybit.com"
    for c in cfg["coins"]:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None
            else:
                assert sc[k] == int(base[k]) * 4
    if not cfg["smoke"]:
        assert cfg["coins"] == COINS5
        assert set(cfg["weights"]) == set(COINS5)
        assert all(abs(w - 0.2) < 1e-6 for w in cfg["weights"].values())
        assert cfg["grid_bars"] > 8000
    assert cfg["smoke"] is False


def test_iter_h14_schema_and_shares():
    d = _load()
    assert d.get("status") == "final"
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert "PENDING" in d["decision_note"]
    assert isinstance(d["conclusion"], str) and len(d["conclusion"]) > 0
    coins = d["config"]["coins"]
    assert set(d["per_coin"]) == set(coins)
    assert set(d["venue"]) == set(coins)
    for c in coins:
        pc = d["per_coin"][c]
        assert pc["symbol"] == c + "USDT"
        assert pc["n"] == d["config"]["grid_bars"]
        for blk in ("assumed", "realized"):
            b = pc[blk]
            for f in ("sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
                      "fund_cum", "gross_cum", "net_cum", "share_net", "share_gross"):
                assert f in b, (c, blk, f)
            assert b["n"] == d["config"]["grid_bars"]
            assert abs(b["final_x"] - round(1.0 + b["cum"], 4)) < 1e-9
            assert abs(round(b["net_cum"], 4) - b["cum"]) < 1e-9
            for s in ("share_net", "share_gross"):
                assert b[s] is None or isinstance(b[s], (int, float))
            if b["net_cum"] != 0:
                assert b["share_net"] is not None
                assert abs(b["share_net"] - round(b["fund_cum"] / b["net_cum"], 4)) < 1e-9
            else:
                assert b["share_net"] is None
            if b["gross_cum"] != 0:
                assert b["share_gross"] is not None
                assert abs(b["share_gross"] - round(b["fund_cum"] / b["gross_cum"], 4)) < 1e-9
            else:
                assert b["share_gross"] is None
        assert abs(pc["rate_delta_vs_assumed"] - round(pc["realized_rate_used"] - 0.0005, 8)) < 1e-9
        assert abs(pc["delta"]["cum"] - round(pc["realized"]["net_cum"] - pc["assumed"]["net_cum"], 6)) < 1e-9
        assert abs(pc["delta"]["sharpe"] - round(pc["realized"]["sharpe"] - pc["assumed"]["sharpe"], 3)) < 1e-9
        v = pc["venue"]
        assert isinstance(v["ok"], bool)
        assert isinstance(v["n_hist"], int) and v["n_hist"] >= 0
        if v["mean_rate"] is not None:
            assert pc["realized_rate_source"] == "demo_mean"
            assert abs(pc["realized_rate_used"] - v["mean_rate"]) < 1e-12
        else:
            assert pc["realized_rate_source"] == "fallback_assumed"
            assert abs(pc["realized_rate_used"] - 0.0005) < 1e-12
    b = d["basket"]
    for blk in ("assumed", "realized"):
        assert b[blk]["n"] == d["config"]["grid_bars"]
        assert abs(b[blk]["final_x"] - round(1.0 + b[blk]["cum"], 4)) < 1e-9
        if b[blk]["net_cum"] != 0:
            assert abs(b[blk]["share_net"] - round(b[blk]["fund_cum"] / b[blk]["net_cum"], 4)) < 1e-9
    assert abs(b["delta"]["cum"] - round(b["realized"]["net_cum"] - b["assumed"]["net_cum"], 6)) < 1e-9


def test_iter_h14_readonly_and_basket_math():
    src = pathlib.Path("research/run_iter_h14_fundreal.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "data/data_1y/1h/" in src
    assert "https://api-demo.bybit.com" in src
    low = src.lower()
    for bad in ("place_order", "submit_order", "market_open", "limit_open", "cancel-all", "cancel_all",
                "api_key", "api_secret", "recv-window", "recv_window", "x-bapi", "position/list",
                "order/create", "set-leverage", "set_leverage", "y1b_live", "enable_live"):
        assert bad not in low, bad
    assert "urllib.request" in src or "urllib" in src
    d = _load()
    coins = d["config"]["coins"]
    bw = 1.0 / len(coins)
    exp_assumed_fund = round(sum(d["per_coin"][c]["assumed"]["fund_cum"] * bw for c in coins), 6)
    exp_real_fund = round(sum(d["per_coin"][c]["realized"]["fund_cum"] * bw for c in coins), 6)
    assert abs(d["basket"]["assumed"]["fund_cum"] - exp_assumed_fund) < 2e-6
    assert abs(d["basket"]["realized"]["fund_cum"] - exp_real_fund) < 2e-6
    exp_assumed_net = round(sum(d["per_coin"][c]["assumed"]["net_cum"] * bw for c in coins), 6)
    exp_real_net = round(sum(d["per_coin"][c]["realized"]["net_cum"] * bw for c in coins), 6)
    assert abs(d["basket"]["assumed"]["net_cum"] - exp_assumed_net) < 2e-6
    assert abs(d["basket"]["realized"]["net_cum"] - exp_real_net) < 2e-6


def test_iter_h14_script_runs_smoke():
    env = dict(os.environ, ITER_H14_SMOKE="1", ITER_H14_NO_VENUE="1",
               ITER_H14_OUT="/tmp/iter_H14_smoke_t2.json", ITER_H14_LOG="/tmp/iter_H14_smoke_t2.log")
    r = subprocess.run([sys.executable, "research/run_iter_h14_fundreal.py"],
                       capture_output=True, text=True, cwd=".", env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(pathlib.Path("/tmp/iter_H14_smoke_t2.json").read_text())
    assert dd["status"] == "final"
    assert dd["verdict"] == "PENDING"
    assert dd["decision"] == "NO_ADOPTION"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["no_venue"] is True
    assert set(dd["config"]["coins"]) == set(SMOKE_COINS)
    assert dd["config"]["grid_bars"] == 3000
    for c in SMOKE_COINS:
        assert dd["per_coin"][c]["venue"]["ok"] is False
        assert dd["per_coin"][c]["realized_rate_source"] == "fallback_assumed"
        assert abs(dd["per_coin"][c]["realized_rate_used"] - 0.0005) < 1e-12
        assert dd["per_coin"][c]["delta"]["sharpe"] == 0.0
        assert dd["per_coin"][c]["delta"]["cum"] == 0.0
    assert dd["basket"]["delta"]["sharpe"] == 0.0
    assert dd["basket"]["delta"]["cum"] == 0.0
