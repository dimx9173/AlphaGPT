"""Frozen offline 28-coin paper manifest.

This manifest is for research/paper only. It does not authorize broker orders.
The five legacy tuned symbols are intentionally not presented as fitted
parameters for the other 23 symbols; all 28 use the explicit common transfer
thresholds until a train-only selector is added.
"""

COINS_28C = [
    "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
    "ETC", "ETH", "HBAR", "ICP", "KAS", "LINK", "LTC", "NEAR", "PEPE", "POL",
    "RENDER", "AAVE", "SOL", "SUI", "TRX", "UNI", "XLM", "XRP",
]

# Explicit transfer defaults, not claims of per-coin optimization.
COMMON_THRESHOLDS_30M = {
    "long": 0.85,
    "short": 0.15,
    "cooldown_bars": 6,
    "stop_loss": 0.05,
}

# Kept for provenance only; the offline runner never calls a broker.
OFFLINE_VENUE_SYMBOLS = {c: f"{c}USDT" for c in COINS_28C}
OFFLINE_VENUE_SYMBOLS.update({"PEPE": "1000PEPEUSDT"})

MANIFEST_28C = {
    c: {
        "symbol": c,
        "venue_symbol": OFFLINE_VENUE_SYMBOLS[c],
        "thresholds": dict(COMMON_THRESHOLDS_30M),
        "threshold_source": "common_transfer_default_unfitted",
    }
    for c in COINS_28C
}

# Acceptance contract for future 28c OOS evaluation. The previous 24/28
# proposal was relaxed by operator instruction; this is a gate contract only,
# not an automatic demo-arm permission.
ACCEPTANCE_GATE_28C = {
    "min_positive_coins": 24,
    "min_oos_portfolio_sharpe": 1.0,
    "max_oos_mdd": 0.25,
    "min_positive_walk_forward_folds": 3,
}
