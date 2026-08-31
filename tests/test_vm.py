import torch
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB

def test_vm_valid_formula_returns_tensor(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    formula = [0, 1, off + 0]
    res = vm.execute(formula, sample_feat_tensor)
    assert res is not None
    assert isinstance(res, torch.Tensor)
    assert res.shape == (sample_feat_tensor.shape[0], sample_feat_tensor.shape[2])
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()

def test_vm_valid_unary_returns_tensor(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    res = vm.execute([2, off + 4], sample_feat_tensor)
    assert res is not None
    assert res.shape == (4, 32)

def test_vm_invalid_token_returns_none(sample_feat_tensor):
    vm = StackVM()
    assert vm.execute([999], sample_feat_tensor) is None
    assert vm.execute([1000, 1001], sample_feat_tensor) is None

def test_vm_nan_handled(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    feat = sample_feat_tensor.clone()
    feat[0, 0, 0] = float('nan')
    feat[1, 1, 5] = float('nan')
    res = vm.execute([0, off + 4], feat)
    assert res is not None
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()

def test_vm_inf_handled(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    feat = sample_feat_tensor.clone()
    feat[0, 0, 0] = float('inf')
    feat[0, 1, 1] = float('-inf')
    res = vm.execute([0, 1, off + 0], feat)
    assert res is not None
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()

def test_vm_arity_mismatch_returns_none(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    assert vm.execute([off + 0], sample_feat_tensor) is None
    assert vm.execute([0, off + 0], sample_feat_tensor) is None
    assert vm.execute([off + 7], sample_feat_tensor) is None
    assert vm.execute([0, off + 7], sample_feat_tensor) is None

def test_vm_empty_or_extra_stack_returns_none(sample_feat_tensor):
    vm = StackVM()
    assert vm.execute([], sample_feat_tensor) is None
    assert vm.execute([0, 1], sample_feat_tensor) is None

def test_vm_feature_out_of_bounds_returns_none():
    vm = StackVM()
    small = torch.randn(2, 2, 8)
    assert vm.execute([5], small) is None
    assert vm.execute([2], small) is None

def test_vm_gate_valid(sample_feat_tensor):
    vm = StackVM()
    off = vm.feat_offset
    res = vm.execute([0, 1, 2, off + 7], sample_feat_tensor)
    assert res is not None
    assert res.shape == (4, 32)
