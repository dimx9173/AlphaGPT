import asyncpg
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION, 
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    PRIMARY KEY (time, address)
                );
            """)
            
            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (address) DO UPDATE 
                SET symbol = EXCLUDED.symbol, last_updated = NOW();
            """, tokens)

    async def get_existing_ohlcv_keys(self, address, since=None):
        if not self.pool:
            return set()
        async with self.pool.acquire() as conn:
            if since is not None:
                rows = await conn.fetch("SELECT time FROM ohlcv WHERE address=$1 AND time >= $2", address, since)
            else:
                rows = await conn.fetch("SELECT time FROM ohlcv WHERE address=$1", address)
            return {r['time'] for r in rows}

    async def filter_new_records(self, records):
        if not records or not self.pool:
            return records
        by_addr = {}
        for r in records:
            by_addr.setdefault(r[1], []).append(r)
        filtered = []
        for addr, recs in by_addr.items():
            existing = await self.get_existing_ohlcv_keys(addr)
            for r in recs:
                if r[0] not in existing:
                    filtered.append(r)
        return filtered

    async def batch_insert_ohlcv(self, records):
        if not records: return
        async with self.pool.acquire() as conn:
            try:
                await conn.copy_records_to_table(
                    'ohlcv',
                    records=records,
                    columns=['time', 'address', 'open', 'high', 'low', 'close', 
                             'volume', 'liquidity', 'fdv', 'source'],
                    timeout=60
                )
            except asyncpg.UniqueViolationError:
                pass # 忽略重复
            except Exception as e:
                logger.error(f"Batch insert error: {e}")