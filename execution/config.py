import json
import os
import re
from dotenv import load_dotenv
from solders.keypair import Keypair

load_dotenv()

class ExecutionConfig:
    RPC_URL = os.getenv("QUICKNODE_RPC_URL", "")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")
    _PAYER_KEYPAIR = None

    PLACEHOLDERS = {"填入RPC地址", "password", ""}
    REQUIRED_KEYS = ["QUICKNODE_RPC_URL", "SOLANA_PRIVATE_KEY"]

    DEFAULT_SLIPPAGE_BPS = 200 # bps

    PRIORITY_LEVEL = "High"

    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    @classmethod
    def _is_placeholder(cls, value):
        return value.strip() in cls.PLACEHOLDERS if isinstance(value, str) else True

    @classmethod
    def validate_env(cls):
        missing = []
        for key in cls.REQUIRED_KEYS:
            val = os.getenv(key)
            if val is None or cls._is_placeholder(val):
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing or placeholder env vars: {', '.join(missing)}. "
                "Please set them in .env (cp .env.example .env)"
            )

    @classmethod
    def has_private_key(cls):
        return bool(cls._PRIV_KEY_STR)

    @classmethod
    def get_payer_keypair(cls):
        if cls._PAYER_KEYPAIR is not None:
            return cls._PAYER_KEYPAIR
        pk_str = os.getenv("SOLANA_PRIVATE_KEY", cls._PRIV_KEY_STR)
        if pk_str is None:
            pk_str = ""
        pk_str = pk_str.strip()
        if not pk_str or cls._is_placeholder(pk_str):
            raise ValueError("Missing SOLANA_PRIVATE_KEY in .env (hint: cp .env.example .env)")
        if pk_str.startswith("["):
            try:
                arr = json.loads(pk_str)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid SOLANA_PRIVATE_KEY format; expected base58 string or JSON array of integers") from e
            try:
                cls._PAYER_KEYPAIR = Keypair.from_bytes(arr)
            except (ValueError, TypeError) as e:
                raise ValueError("Invalid SOLANA_PRIVATE_KEY format; expected base58 string or JSON array of integers") from e
            return cls._PAYER_KEYPAIR
        if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,88}", pk_str):
            raise ValueError("Invalid SOLANA_PRIVATE_KEY format; expected base58 string or JSON array of integers")
        try:
            cls._PAYER_KEYPAIR = Keypair.from_base58_string(pk_str)
        except (ValueError, TypeError) as e:
            try:
                arr = json.loads(pk_str)
                cls._PAYER_KEYPAIR = Keypair.from_bytes(arr)
            except (json.JSONDecodeError, ValueError, TypeError):
                raise ValueError("Invalid SOLANA_PRIVATE_KEY format; expected base58 string or JSON array of integers") from e
        return cls._PAYER_KEYPAIR

    @classmethod
    def get_wallet_address(cls):
        return str(cls.get_payer_keypair().pubkey())


class HyperliquidConfig:
    """Hyperliquid venue config (P4 Phase 2). Testnet by default."""

    MAINNET_URL = "https://api.hyperliquid.xyz"
    TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

    @classmethod
    def use_testnet(cls) -> bool:
        return __import__("os").getenv("HYPERLIQUID_TESTNET", "true").strip().lower() in {
            "1", "true", "yes", "y",
        }

    @classmethod
    def base_url(cls) -> str:
        return cls.TESTNET_URL if cls.use_testnet() else cls.MAINNET_URL

    @classmethod
    def account_address(cls) -> str:
        return __import__("os").getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "").strip()

    @classmethod
    def secret_key(cls) -> str:
        return __import__("os").getenv("HYPERLIQUID_SECRET_KEY", "").strip()

    @classmethod
    def enabled(cls) -> bool:
        venues = __import__("os").getenv("VENUES_ENABLED", "solana")
        return "hyperliquid" in {v.strip().lower() for v in venues.split(",")}

    @classmethod
    def validate_env(cls):
        if not cls.enabled():
            return
        missing = []
        if not cls.account_address():
            missing.append("HYPERLIQUID_ACCOUNT_ADDRESS")
        if not cls.secret_key():
            missing.append("HYPERLIQUID_SECRET_KEY")
        if missing:
            raise ValueError(
                f"Missing env vars: {', '.join(missing)}. "
                "Set them in .env (testnet keys first; see .env.example)"
            )


class AsterConfig:
    """Aster venue config (P4 Phase 3). V3 EIP-712, testnet by default."""

    FUTURES_MAIN = "https://fapi.asterdex.com"
    FUTURES_TEST = "https://fapi.asterdex-testnet.com"
    SPOT_MAIN = "https://sapi.asterdex.com"
    SPOT_TEST = "https://sapi.asterdex-testnet.com"

    @classmethod
    def use_testnet(cls) -> bool:
        import os

        return os.getenv("ASTER_TESTNET", "true").strip().lower() in {
            "1", "true", "yes", "y",
        }

    @classmethod
    def futures_url(cls) -> str:
        return cls.FUTURES_TEST if cls.use_testnet() else cls.FUTURES_MAIN

    @classmethod
    def spot_url(cls) -> str:
        return cls.SPOT_TEST if cls.use_testnet() else cls.SPOT_MAIN

    @classmethod
    def _get(cls, key: str) -> str:
        import os

        return os.getenv(key, "").strip()

    @classmethod
    def user_address(cls) -> str:
        return cls._get("ASTER_USER_ADDRESS")

    @classmethod
    def signer_address(cls) -> str:
        return cls._get("ASTER_SIGNER_ADDRESS")

    @classmethod
    def signer_key(cls) -> str:
        return cls._get("ASTER_SIGNER_PRIVATE_KEY")

    @classmethod
    def enabled(cls) -> bool:
        import os

        venues = os.getenv("VENUES_ENABLED", "solana")
        return "aster" in {v.strip().lower() for v in venues.split(",")}

    @classmethod
    def validate_env(cls):
        if not cls.enabled():
            return
        missing = [k for k, v in [
            ("ASTER_USER_ADDRESS", cls.user_address()),
            ("ASTER_SIGNER_ADDRESS", cls.signer_address()),
            ("ASTER_SIGNER_PRIVATE_KEY", cls.signer_key()),
        ] if not v]
        if missing:
            raise ValueError(
                f"Missing env vars: {', '.join(missing)}. "
                "Create an API-wallet at /en/api-wallet; see .env.example"
            )

    @classmethod
    def make_signer(cls):
        from .brokers.aster_sign import AsterSigner

        return AsterSigner(cls.user_address(), cls.signer_address(), cls.signer_key())
