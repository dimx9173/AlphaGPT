"""CEX venue configs (P4 Phase 5): Binance / Bybit / OKX USDT-margined perp.

Testnet/demo by default. No orders without explicit keys + live flags.
Only stdlib + os here (no solders import) so paper/shadow paths stay light.
"""
from __future__ import annotations

import os


def _get(key: str) -> str:
    return os.getenv(key, "").strip()


def _is_true(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


class BinanceConfig:
    """Binance USDT-M futures. Testnet default."""

    MAIN = "https://fapi.binance.com"
    TEST = "https://testnet.binancefuture.com"

    @classmethod
    def use_testnet(cls) -> bool:
        return _is_true(os.getenv("BINANCE_TESTNET", "true"))

    @classmethod
    def base_url(cls) -> str:
        return cls.TEST if cls.use_testnet() else cls.MAIN

    @classmethod
    def api_key(cls) -> str:
        return _get("BINANCE_API_KEY")

    @classmethod
    def api_secret(cls) -> str:
        return _get("BINANCE_API_SECRET")

    @classmethod
    def enabled(cls) -> bool:
        venues = os.getenv("VENUES_ENABLED", "solana")
        return "binance" in {v.strip().lower() for v in venues.split(",")}

    @classmethod
    def validate_env(cls):
        if not cls.enabled():
            return
        missing = [k for k, v in [
            ("BINANCE_API_KEY", cls.api_key()),
            ("BINANCE_API_SECRET", cls.api_secret()),
        ] if not v]
        if missing:
            raise ValueError(
                f"Missing env vars: {', '.join(missing)}. "
                "Create futures API keys first; see .env.example"
            )


class BybitConfig:
    """Bybit V5 (linear perp). Demo ONLY — mainnet/testnet disabled.

    Locked to https://api-demo.bybit.com. BYBIT_TESTNET is accepted but
    ignored (kept for .env compat); there is no other path.
    """

    TEST = "https://api-demo.bybit.com"

    @classmethod
    def use_testnet(cls) -> bool:
        return True

    @classmethod
    def base_url(cls) -> str:
        return cls.TEST

    @classmethod
    def api_key(cls) -> str:
        return _get("BYBIT_API_KEY")

    @classmethod
    def api_secret(cls) -> str:
        return _get("BYBIT_API_SECRET")

    @classmethod
    def enabled(cls) -> bool:
        venues = os.getenv("VENUES_ENABLED", "solana")
        return "bybit" in {v.strip().lower() for v in venues.split(",")}

    @classmethod
    def validate_env(cls):
        if not cls.enabled():
            return
        missing = [k for k, v in [
            ("BYBIT_API_KEY", cls.api_key()),
            ("BYBIT_API_SECRET", cls.api_secret()),
        ] if not v]
        if missing:
            raise ValueError(
                f"Missing env vars: {', '.join(missing)}. "
                "Create V5 API keys first; see .env.example"
            )


class OkxConfig:
    """OKX SWAP (USDT-margined). Demo (x-simulated-trading) default."""

    MAIN = "https://www.okx.com"

    @classmethod
    def use_demo(cls) -> bool:
        return _is_true(os.getenv("OKX_DEMO", "true"))

    @classmethod
    def base_url(cls) -> str:
        return _get("OKX_BASE_URL") or cls.MAIN

    @classmethod
    def api_key(cls) -> str:
        return _get("OKX_API_KEY")

    @classmethod
    def api_secret(cls) -> str:
        return _get("OKX_API_SECRET")

    @classmethod
    def passphrase(cls) -> str:
        return _get("OKX_PASSPHRASE")

    @classmethod
    def enabled(cls) -> bool:
        venues = os.getenv("VENUES_ENABLED", "solana")
        return "okx" in {v.strip().lower() for v in venues.split(",")}

    @classmethod
    def validate_env(cls):
        if not cls.enabled():
            return
        missing = [k for k, v in [
            ("OKX_API_KEY", cls.api_key()),
            ("OKX_API_SECRET", cls.api_secret()),
            ("OKX_PASSPHRASE", cls.passphrase()),
        ] if not v]
        if missing:
            raise ValueError(
                f"Missing env vars: {', '.join(missing)}. "
                "Create OKX API keys first; see .env.example"
            )
