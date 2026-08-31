import asyncio
import json
import os
import aiohttp
from datetime import datetime, timedelta
from loguru import logger
from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider

class CheckpointStore:
    def __init__(self, path=None):
        self.path = path or Config.CHECKPOINT_PATH

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
        except Exception as e:
            logger.warning(f"Checkpoint load failed {self.path}: {e}")
        return set()

    def save(self, completed):
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'completed': sorted(completed), 'updated_at': datetime.now().isoformat()}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception as e:
            logger.warning(f"Checkpoint save failed {self.path}: {e}")

    def clear(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception as e:
            logger.warning(f"Checkpoint clear failed {self.path}: {e}")

class DataManager:
    def __init__(self, checkpoint_path=None):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        self.checkpoint = CheckpointStore(checkpoint_path)

    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    def _filter_records_by_existing(self, records):
        if not records:
            return records
        seen = set()
        deduped = []
        for r in records:
            key = (r[0], r[1])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    async def pipeline_sync_daily(self):
        logger.info("Step 1: Discovering trending tokens...")
        limit = 500 if Config.BIRDEYE_IS_PAID else 100
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        logger.info(f"Raw candidates found: {len(candidates)}")
        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue
            selected_tokens.append(t)
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return
        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)
        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        completed = self.checkpoint.load()
        if completed:
            before = len(selected_tokens)
            selected_tokens = [t for t in selected_tokens if t['address'] not in completed]
            logger.info(f"Checkpoint resume: skip {before - len(selected_tokens)} already-fetched, {len(selected_tokens)} remaining")
            if not selected_tokens:
                logger.info("All tokens already fetched per checkpoint; clearing checkpoint")
                self.checkpoint.clear()
                return
        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            batch_size = 20
            total_candles = 0
            completed_now = set(completed)
            for i in range(0, len(selected_tokens), batch_size):
                batch_tokens = selected_tokens[i:i+batch_size]
                tasks = [self.birdeye.get_token_history(session, t['address'], liquidity=t.get('liquidity'), fdv=t.get('fdv')) for t in batch_tokens]
                results = await asyncio.gather(*tasks)
                records = [item for sublist in results if sublist for item in sublist]
                records = self._filter_records_by_existing(records)
                if self.db.pool is not None:
                    records = await self.db.filter_new_records(records)
                await self.db.batch_insert_ohlcv(records)
                total_candles += len(records)
                for t in batch_tokens:
                    completed_now.add(t['address'])
                self.checkpoint.save(completed_now)
                logger.info(f"Processed batch {i}/{len(selected_tokens)}. Inserted {len(records)} candles.")
        self.checkpoint.clear()
        logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
