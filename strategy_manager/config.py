import os
from dataclasses import dataclass, field


class StrategyConfig:
    MAX_OPEN_POSITIONS = 3
    ENTRY_AMOUNT_SOL = 2.0
    STOP_LOSS_PCT = -0.05
    TAKE_PROFIT_Target1 = 0.10
    TP_Target1_Ratio = 0.5
    TRAILING_ACTIVATION = 0.05
    TRAILING_DROP = 0.03
    BUY_THRESHOLD = 0.85
    SELL_THRESHOLD = 0.45


def _parse_blacklist(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


def _parse_venues(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return {"solana"}
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


@dataclass(frozen=True)
class RiskConfig:
    daily_loss_pct: float = float(os.getenv("RISK_DAILY_LOSS_PCT", "0.10"))
    max_drawdown_pct: float = float(os.getenv("RISK_MAX_DRAWDOWN_PCT", "0.15"))
    max_single_exposure_sol: float = float(os.getenv("RISK_MAX_SINGLE_EXPOSURE_SOL", "1.0"))
    blacklist: set[str] = field(default_factory=lambda: _parse_blacklist(os.getenv("RISK_BLACKLIST", "")))
    velocity_max: int = int(os.getenv("RISK_VELOCITY_MAX", "3"))
    velocity_window_sec: int = int(os.getenv("RISK_VELOCITY_WINDOW_SEC", "60"))
    circuit_cooldown_sec: float = float(os.getenv("RISK_CIRCUIT_COOLDOWN_SEC", "300"))
    # P4 perp trio (conservative defaults; spot path unaffected)
    perp_max_leverage: int = int(os.getenv("PERP_MAX_LEVERAGE", "3"))
    perp_max_notional_usdt: float = float(os.getenv("PERP_MAX_NOTIONAL_USDT", "500"))
    max_funding_rate: float = float(os.getenv("PERP_MAX_FUNDING_RATE", "0.001"))
    venues_enabled: set[str] = field(default_factory=lambda: _parse_venues(os.getenv("VENUES_ENABLED", "solana")))

    def validate(self) -> None:
        if not 0 < self.daily_loss_pct < 1:
            raise ValueError(f"RISK_DAILY_LOSS_PCT must be (0,1), got {self.daily_loss_pct}")
        if not 0 < self.max_drawdown_pct < 1:
            raise ValueError(f"RISK_MAX_DRAWDOWN_PCT must be (0,1), got {self.max_drawdown_pct}")
        if self.max_single_exposure_sol <= 0:
            raise ValueError(f"RISK_MAX_SINGLE_EXPOSURE_SOL must be >0, got {self.max_single_exposure_sol}")
        if self.velocity_max < 0 or self.velocity_window_sec <= 0:
            raise ValueError(f"RISK_VELOCITY_MAX/WINDOW invalid: {self.velocity_max}/{self.velocity_window_sec}")
        if self.perp_max_leverage < 1 or self.perp_max_leverage > 125:
            raise ValueError(f"PERP_MAX_LEVERAGE must be [1,125], got {self.perp_max_leverage}")
        if self.perp_max_notional_usdt <= 0:
            raise ValueError(f"PERP_MAX_NOTIONAL_USDT must be >0, got {self.perp_max_notional_usdt}")
        if not 0 <= self.max_funding_rate < 1:
            raise ValueError(f"PERP_MAX_FUNDING_RATE must be [0,1), got {self.max_funding_rate}")
