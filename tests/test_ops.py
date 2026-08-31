import torch
from model_core.ops import OPS_CONFIG, _ts_delay, _op_gate, _op_jump, _op_decay

def _rand(n=2, t=16):
    return torch.randn(n, t)

def test_all_ops_produce_expected_shape():
    n, t = 3, 20
    x = _rand(n, t)
    y = _rand(n, t)
    z = _rand(n, t)
    for name, func, arity in OPS_CONFIG:
        if arity == 1:
            res = func(x)
        elif arity == 2:
            res = func(x, y)
        else:
            res = func(x, y, z)
        assert res.shape == (n, t), f"{name} shape mismatch {res.shape}"

def test_div_by_zero_finite():
    x = torch.ones(2, 8)
    y = torch.zeros(2, 8)
    _, func, _ = OPS_CONFIG[3]
    res = func(x, y)
    assert torch.isfinite(res).all()
    assert res.shape == (2, 8)

def test_gate_logic():
    cond = torch.tensor([[1.0, -1.0, 0.0, 2.0]])
    x = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    y = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    res = _op_gate(cond, x, y)
    assert res[0,0].item() == 10.0
    assert res[0,1].item() == 1.0
    assert res[0,2].item() == 1.0
    assert res[0,3].item() == 10.0

def test_jump_shape_and_relu():
    x = torch.randn(2, 16)
    res = _op_jump(x)
    assert res.shape == x.shape
    assert (res >= 0).all()

def test_decay_shape():
    x = _rand(2, 10)
    res = _op_decay(x)
    assert res.shape == x.shape
    assert torch.isfinite(res).all()

def test_delay1_zero_pad():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    res = _ts_delay(x, 1)
    assert res[0,0].item() == 0.0
    assert res[0,1].item() == 1.0
    assert res[0,3].item() == 3.0

def test_max3_is_max_of_delayed():
    x = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
    d1 = _ts_delay(x, 1)
    d2 = _ts_delay(x, 2)
    expected = torch.max(x, torch.max(d1, d2))
    _, func, _ = OPS_CONFIG[11]
    res = func(x)
    assert torch.allclose(res, expected)

def test_ts_delay_zero_returns_same():
    x = _rand(2, 8)
    assert torch.allclose(_ts_delay(x, 0), x)

def test_neg_abs_sign():
    x = torch.tensor([[1.0, -2.0, 0.0]])
    neg = OPS_CONFIG[4][1](x)
    assert neg[0,0].item() == -1.0
    assert neg[0,1].item() == 2.0
    ab = OPS_CONFIG[5][1](x)
    assert ab[0,1].item() == 2.0
    sg = OPS_CONFIG[6][1](x)
    assert sg[0,0].item() == 1.0
    assert sg[0,1].item() == -1.0
    assert sg[0,2].item() == 0.0
