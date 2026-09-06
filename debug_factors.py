
import torch, pandas as pd
from run_backtest import load_csv, compute_factors, compute_target_ret, build_raw_data
from model_core.backtest import MemeBacktest

df = load_csv("hl_HYPE_1h.csv")
torch.manual_seed(42)
n_tokens, n_steps = 4, 32
factors = compute_factors(df, n_tokens, n_steps)
print("factors:", factors.flatten().tolist()[:10])
sig = torch.sigmoid(factors)
print("signal max:", sig.max().item())
print("long entries (>0.85):", (sig>0.85).sum().item())

# The issue: factor * 30 may produce signal > 0.85 but we need to check the actual values
# Let me amplify: use 50 instead of 30
factors2 = factors * 16  # scale up
sig2 = torch.sigmoid(factors2)
print("scaled signal max:", sig2.max().item())
print("long entries (scaled):", (sig2>0.85).sum().item())
print("short entries (scaled):", (sig2<0.15).sum().item())
