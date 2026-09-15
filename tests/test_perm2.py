"""P0-3b perm2 tests: helpers + fast recompute (small N) of run_perm2.py."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import torch

OUT = pathlib.Path(os.getenv("PERM2_OUT", "results/permutation2.json"))


def _mod():
    spec = importlib.util.spec_from_file_location("perm2mod", "research/run_perm2.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["perm2mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_block_perm_is_valid_permutation():
    m = _mod()
    g = torch.Generator().manual_seed(0)
    idx = m.block_perm(100, 24, g)
    assert sorted(idx.tolist()) == list(range(100))


def test_block_perm_preserves_within_block_order():
    m = _mod()
    g = torch.Generator().manual_seed(1)
    idx = m.block_perm(96, 24, g).tolist()
    # each consecutive run inside the output that came from one input block
    # must be increasing; check globally: positions of values within same
    # input block appear in increasing relative order
    pos = {v: i for i, v in enumerate(idx)}
    for s in range(0, 96, 24):
        seq = [pos[v] for v in range(s, s + 24)]
        assert seq == sorted(seq)


def test_block1_equals_full_shuffle_distribution():
    m = _mod()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    assert m.block_perm(50, 1, g1).tolist() == torch.randperm(50, generator=g2).tolist()


def test_reverse_idx():
    m = _mod()
    assert m.reverse_idx(5).tolist() == [4, 3, 2, 1, 0]


def test_summ_p_value_formula():
    m = _mod()
    s = m._summ([1.0, 2.0, 3.0], 2.5)
    assert s["p_value"] == round((1.0 + 1) / (1.0 + 3), 4)
    assert s["observed"] == 2.5
    assert s["gate_p_lt_0_05"] is False


def test_perm2_script_runs_small_n():
    env = dict(os.environ, PERM2_N="8", PERM2_OUT="/tmp/perm2_test.json")
    r = subprocess.run([sys.executable, "research/run_perm2.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("/tmp/perm2_test.json").read_text())
    assert set(d["blocks_both_legs"]) == {"1", "24", "96"}
    assert d["config"]["n_perm"] == 8
    assert all(v["n_perm"] == 8 for v in d["blocks_both_legs"].values())
    for b, dg in d["single_leg_diagnostic"].items():
        assert set(dg) == {"ETC", "TRX"}
        assert all(v["diagnostic_only"] is True for v in dg.values())
    assert "combo_sharpe" in d["reversal"]
    assert isinstance(d["verdict"]["combo_block24_pass"], bool)
