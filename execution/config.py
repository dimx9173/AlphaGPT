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
