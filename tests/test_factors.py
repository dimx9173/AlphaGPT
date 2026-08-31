import torch
from model_core.factors import FeatureEngineer, AdvancedFactorEngineer, MemeIndicators

def test_feature_engineer_compute_features_shape(sample_raw_data):
    feats = FeatureEngineer.compute_features(sample_raw_data)
    n, t = sample_raw_data["close"].shape
    assert feats.shape == (n, 6, t)
    assert torch.isfinite(feats).all()

def test_robust_norm_clamps():
    eng = AdvancedFactorEngineer()
    t = torch.tensor([[1.0, 2.0, 100.0, 3.0, -100.0]])
    res = eng.robust_norm(t)
    assert res.shape == t.shape
    assert (res <= 5.0).all() and (res >= -5.0).all()
    assert torch.isfinite(res).all()

def test_robust_norm_handles_constant():
    eng = AdvancedFactorEngineer()
    t = torch.ones(2, 10)
    res = eng.robust_norm(t)
    assert torch.isfinite(res).all()
    assert (res == 0).all()

def test_six_factors_are_finite(sample_raw_data):
    feats = FeatureEngineer.compute_features(sample_raw_data)
    for i in range(6):
        assert torch.isfinite(feats[:, i, :]).all(), f"factor {i} not finite"

def test_advanced_engineer_twelve_features(sample_raw_data):
    eng = AdvancedFactorEngineer()
    feats = eng.compute_advanced_features(sample_raw_data)
    n, t = sample_raw_data["close"].shape
    assert feats.shape == (n, 12, t)
    assert torch.isfinite(feats).all()

def test_feature_engineer_input_dim():
    assert FeatureEngineer.INPUT_DIM == 6

def test_meme_indicators_liquidity_health_bounds(sample_raw_data):
    liq = sample_raw_data["liquidity"]
    fdv = sample_raw_data["fdv"]
    res = MemeIndicators.liquidity_health(liq, fdv)
    assert ((res >= 0) & (res <= 1)).all()
    assert res.shape == liq.shape

def test_compute_features_no_nan(sample_raw_data):
    sample_raw_data["volume"][0, 0] = 0
    feats = FeatureEngineer.compute_features(sample_raw_data)
    assert not torch.isnan(feats).any()
