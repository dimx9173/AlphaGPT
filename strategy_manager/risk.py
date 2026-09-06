import time
from collections import defaultdict, deque
from .config import StrategyConfig, RiskConfig
from execution.jupiter import JupiterAggregator  # 复用 Jupiter 做模拟
from loguru import logger


class RiskEngine:
    def __init__(self, config=None, risk_config=None, jupiter=None):
        self.config = config or StrategyConfig()
        self.risk_config = risk_config or RiskConfig()
        self.jup = jupiter or JupiterAggregator()
        self._circuit_open_until: float = 0.0
        self._velocity: dict[str, deque[float]] = defaultdict(deque)
        # 可由外部注入的帳面指標，用於測試與 runner 回寫
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _open_circuit(self, reason: str) -> None:
        self._circuit_open_until = time.monotonic() + self.risk_config.circuit_cooldown_sec
        logger.warning(f"[x] Risk: CIRCUIT OPEN ({reason}) for {self.risk_config.circuit_cooldown_sec}s")

    def check_circuit(self, daily_pnl: float | None = None, drawdown: float | None = None) -> tuple[bool, str]:
        """返回 (is_blocked, reason)。daily_pnl 為負數表示虧損，drawdown 為 0~1。"""
        if self._is_circuit_open():
            return True, "circuit_cooldown"
        pnl = daily_pnl if daily_pnl is not None else self._daily_pnl
        if pnl < -self.risk_config.daily_loss_pct:
            self._open_circuit(f"daily_loss {pnl:.2%} < -{self.risk_config.daily_loss_pct:.0%}")
            return True, "daily_loss"
        if drawdown is not None and drawdown > self.risk_config.max_drawdown_pct:
            self._open_circuit(f"drawdown {drawdown:.2%} > {self.risk_config.max_drawdown_pct:.0%}")
            return True, "max_drawdown"
        return False, ""

    def is_blacklisted(self, token_address: str) -> bool:
        return token_address in self.risk_config.blacklist

    def check_velocity(self, token_address: str, now: float | None = None) -> bool:
        """True 表示可通過，False 表示超速被限。"""
        now = now if now is not None else time.monotonic()
        dq = self._velocity[token_address]
        window = self.risk_config.velocity_window_sec
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= self.risk_config.velocity_max:
            logger.warning(f"[x] Risk: Velocity limit hit for {token_address} ({len(dq)}/{self.risk_config.velocity_max} in {window}s)")
            return False
        dq.append(now)
        return True

    def set_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl = pnl

    def set_peak_equity(self, peak: float) -> None:
        self._peak_equity = peak

    async def check_safety(
        self,
        token_address: str,
        liquidity_usd: float,
        exposure_sol: float | None = None,
        daily_pnl: float | None = None,
        drawdown: float | None = None,
        leverage: int | None = None,
        notional_usdt: float | None = None,
        funding_rate: float | None = None,
    ) -> bool:
        # 1) 熔斷（可注入指標）
        blocked, reason = self.check_circuit(daily_pnl=daily_pnl, drawdown=drawdown)
        if blocked:
            logger.warning(f"[x] Risk: Blocked by circuit ({reason})")
            return False

        # 2) 黑名單
        if self.is_blacklisted(token_address):
            logger.warning(f"[x] Risk: Blacklisted {token_address}")
            return False

        # 3) 速度門檻
        if not self.check_velocity(token_address):
            return False

        # 4) 單幣敞口上限（若調用方提供當前敞口）
        if exposure_sol is not None and exposure_sol > self.risk_config.max_single_exposure_sol:
            logger.warning(f"[x] Risk: Single exposure {exposure_sol:.4f} SOL > {self.risk_config.max_single_exposure_sol} SOL")
            return False

        # 4b) perp 門檻（僅當調用方傳入 perp 參數時觸發，現貨行為不變）
        if leverage is not None or notional_usdt is not None or funding_rate is not None:
            ok, _ = self.check_perp(
                token_address,
                leverage if leverage is not None else 1,
                notional_usdt if notional_usdt is not None else 0.0,
                funding_rate,
            )
            if not ok:
                return False

        # 5) 原有：流動性門檻
        if liquidity_usd < 5000:
            logger.warning(f"[x] Risk: Liquidity too low (${liquidity_usd})")
            return False

        # 6) 原有：honeypot 探針
        try:
            quote = await self.jup.get_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount_integer=1000000,
                slippage_bps=1000,
            )
            if not quote:
                logger.warning(f"[x] Risk: Cannot verify sell path (Honeypot?)")
                return False
        except Exception:
            return False

        return True

    def check_perp(
        self,
        symbol: str,
        leverage: int,
        notional_usdt: float,
        funding_rate: float | None = None,
    ) -> tuple[bool, str]:
        """Perp gate: (allowed, reason). Empty reason means pass."""
        if leverage > self.risk_config.perp_max_leverage:
            logger.warning(
                f"[x] Risk: leverage {leverage}x > max {self.risk_config.perp_max_leverage}x ({symbol})"
            )
            return False, "leverage"
        if notional_usdt > self.risk_config.perp_max_notional_usdt:
            logger.warning(
                f"[x] Risk: notional ${notional_usdt:.0f} > max ${self.risk_config.perp_max_notional_usdt:.0f} ({symbol})"
            )
            return False, "notional"
        if funding_rate is not None and abs(funding_rate) > self.risk_config.max_funding_rate:
            logger.warning(
                f"[x] Risk: funding {funding_rate:.4%} > max {self.risk_config.max_funding_rate:.4%} ({symbol})"
            )
            return False, "funding"
        if self.is_blacklisted(symbol):
            logger.warning(f"[x] Risk: Blacklisted {symbol}")
            return False, "blacklist"
        return True, ""

    def calculate_position_size(self, wallet_balance_sol: float) -> float:
        size = self.config.ENTRY_AMOUNT_SOL
        # 受單幣敞口上限約束
        size = min(size, self.risk_config.max_single_exposure_sol)
        if wallet_balance_sol < size + 0.1:
            return 0.0
        return size

    async def close(self):
        await self.jup.close()
