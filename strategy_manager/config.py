import os
from dataclasses import dataclass, field

# === E10 LOCK 2026-09-07 — 10-route consolidation ===
# FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG
# venue aster perp 2x fund0.0005 fee0.0004/0.0008 n=6580 4h
# Y1b global: ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24, q0.3 long-only, 50/50
# overlay Z1 vt0.012/w12 tail-conditional only; challengers AA 0.10/15/9 + 0.12/15/9 shadow
# AB/AC/AD證: 不切short、不轉權重(50/50)、不換公式
# GATE: window200 thresh1.0 main (overlay trailing200 sharpe>1) coverage 0.652 adopted Y1b_main_thr1.0_w200
# Sources: docs/STRATEGY_E10.md + results/backtest_E10.json

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
FORMULA_DECODE = ["FOMO", "PRESSURE", "SUB", "PRESSURE", "SUB", "ABS", "DECAY", "DEV", "DEV", "ADD", "ADD", "NEG"]

# Locked per-coin (Y1b global)
LOCKED_ETC = dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24, vt=None, vw=12, q=0.3)
LOCKED_TRX = dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24, vt=None, vw=12, q=0.3)
LOCKED_WEIGHTS = [0.5, 0.5]
LOCKED_OVERLAY_Z1 = dict(ETC=dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24, vt=0.012, vw=12),
                         TRX=dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24, vt=0.012, vw=12))
LOCKED_GATE = dict(window=200, thresh=1.0, type="main", adopted_id="Y1b_main_thr1.0_w200", coverage=0.652)
LEV = 2.0
FUND = 0.0005
FEE = 0.0004
FEE2X = 0.0008

# Challenger branches (shadow-only, not active)
CHALLENGER_AA_H1 = dict(sth=0.10, etc_cd=15, trx_cd=9, vt=0.012, vw=12)  # unbiased H1->H2 PASS
CHALLENGER_AA_H2 = dict(sth=0.12, etc_cd=15, trx_cd=9, vt=0.012, vw=12)  # biased contender
# E13 hardened guard (shadow-only): AA-H1 cd+2 worstB 0.354->3.065 worstC 3.465->4.573
# FULL 2.010 H2 6.612 turnover 0.136 med12 2.376; source results/results_E13_fee.json
CHALLENGER_AA_H1_CD2 = dict(sth=0.10, etc_cd=17, trx_cd=11, vt=0.012, vw=12, ts=24, q=0.3)

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
    # E10 locked aliases (mirror LOCKED_* for code consumers)
    FORMULA = FORMULA
    LOCKED_ETC = LOCKED_ETC
    LOCKED_TRX = LOCKED_TRX
    LOCKED_WEIGHTS = LOCKED_WEIGHTS
    LOCKED_GATE = LOCKED_GATE


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
    # P4 perp trio — E10 LOCK: perp_max_leverage=2 (was 3), notional 500, funding 0.001
    perp_max_leverage: int = int(os.getenv("PERP_MAX_LEVERAGE", "2"))
    perp_max_notional_usdt: float = float(os.getenv("PERP_MAX_NOTIONAL_USDT", "500"))
    max_funding_rate: float = float(os.getenv("PERP_MAX_FUNDING_RATE", "0.001"))
    venues_enabled: set[str] = field(default_factory=lambda: _parse_venues(os.getenv("VENUES_ENABLED", "aster")))

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
