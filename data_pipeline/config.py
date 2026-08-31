import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0  
    MIN_FDV = 10000000.0            
    MAX_FDV = float('inf') 
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_BASE_URL = os.getenv("BIRDEYE_BASE_URL", "https://public-api.birdeye.so")
    BASE_URL = BIRDEYE_BASE_URL
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = int(os.getenv("PIPELINE_CONCURRENCY", "20"))
    HISTORY_DAYS = int(os.getenv("PIPELINE_HISTORY_DAYS", "7"))
    CHECKPOINT_PATH = os.getenv("PIPELINE_CHECKPOINT_PATH", "data_pipeline/checkpoint.json")

    PLACEHOLDERS = {"", "password"}
    REQUIRED_KEYS = ["BIRDEYE_API_KEY", "DB_USER"]

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
