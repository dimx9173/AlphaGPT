"""X6 decision-frequency test: schema-level only, recompute by smoke script."""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ARMS = {"hourly", "aligned_4h"}

def _load():
    p = pathlib.Path("results/iter_X6_freq.json")
    assert p.exists(), "results/iter_X6_freq.json missing; run research/run_iter_x6_freq.py"
    return json.loads(p.read_text())

def test_iter_x6_schema():
    d = _load()
    cfg = d['config']
    assert cfg['grid'] == '1h'
    assert cfg['grid_bars'] > 2000
    assert cfg['h2_len'] == 493*4
    assert cfg['smoke'] in (True, False)
    if not cfg['smoke']:
        assert cfg["coins"] == ["ETC","TRX","ATOM","APT","KAS"]
    assert set(d['arms']) == ARMS
    n = cfg['grid_bars']
    for a in ARMS:
        arm = d['arms'][a]
        assert SEG <= set(arm['FULL'])
        assert SEG <= set(arm['H2'])
        assert arm['FULL']['n'] == n
        assert arm['H2']['n'] == cfg['h2_len']
        assert arm['FULL']['turnover'] >= 0
        assert arm['flips'] >= 0 and arm['entries'] >= 0
        for coin, pc in arm['per_coin'].items():
            assert SEG <= set(pc['FULL'])
            assert SEG <= set(pc['H2'])
            assert pc['FULL']['n'] == n
    assert set(d['fee2x']) == ARMS
    for a in ARMS:
        assert d["fee2x"][a]["FULL_sharpe"] <= d["arms"][a]["FULL"]["sharpe"] + 1e-9
    cmp = d['compare']
    fh, fa = d["arms"]["hourly"]["FULL"], d["arms"]["aligned_4h"]["FULL"]
    assert abs(cmp["d_sharpe_FULL"] - round(fh["sharpe"]-fa["sharpe"],3)) < 1e-9
    assert abs(cmp["d_turnover_FULL"] - round(fh["turnover"]-fa["turnover"],6)) < 1e-9
    assert cmp['turnover_ratio_hourly_over_4h'] >= 1.0
    assert cmp["d_flips"] == d["arms"]["hourly"]["flips"]-d["arms"]["aligned_4h"]["flips"]
    assert cmp["d_entries"] == d["arms"]["hourly"]["entries"]-d["arms"]["aligned_4h"]["entries"]
    rec = d["recommendation"]
    assert rec["Y1B_DECISION_ALIGN"] in ("0","1")
    exp = "0" if cmp["d_sharpe_FULL"] >= 0.2 else "1"
    assert rec["Y1B_DECISION_ALIGN"] == exp
    assert d["decision"] == "PENDING"
    assert "PENDING" in d["conclusion"] or "no live change" in d["conclusion"]

def test_iter_x6_4h_gate_semantics():
    src = pathlib.Path('research/run_iter_x6_freq.py').read_text()
    assert 'def apply_gate' in src
    assert 't % 4 == 0' in src
    assert 'aligned_4h' in src and 'hourly' in src
    ns = {}
    exec('def apply_gate(d, freq):\n    n=len(d)\n    exec_src=True\n', ns)
    import importlib.util
    spec = importlib.util.spec_from_file_location('x6mod', 'research/run_iter_x6_freq.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['x6mod'] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    d = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    h = m.apply_gate(d, 'hourly')
    a = m.apply_gate(d, 'aligned_4h')
    assert h == [0.0] + d[:-1]
    assert a[0] == 0.0 and a[1] == d[0] and a[4] == d[0] and a[5] == d[4]
    assert m.apply_gate([1.0, -1.0, 1.0, -1.0, 1.0], 'aligned_4h')[2] == 1.0

def test_iter_x6_no_broker():
    src = pathlib.Path('research/run_iter_x6_freq.py').read_text().lower()
    for bad in ('place_order','submit_order','api_key','make_broker','y1b_live','market_open'):
        assert bad not in src, bad

def test_iter_x6_smoke_runs_offline(tmp_path):
    out = tmp_path / 'iter_X6_freq.json'
    lg = tmp_path / 'iter_x6.log'
    env = dict(os.environ, ITER_X6_SMOKE='1', ITER_X6_OUT=str(out), ITER_X6_LOG=str(lg))
    r = subprocess.run([sys.executable, 'research/run_iter_x6_freq.py'], capture_output=True, text=True, cwd='.', env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd['config']['smoke'] is True
    assert set(dd['arms']) == ARMS
    assert dd['config']['grid_bars'] == 3000
