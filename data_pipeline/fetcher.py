import aiohttp
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from .config import Config

class BirdeyeFetcher:
    MAX_RETRIES = 3

    def __init__(self):
        self.headers = {
            "X-API-KEY": Config.BIRDEYE_API_KEY,
            "accept": "application/json"
        }
        self.semaphore = asyncio.Semaphore(5)

    async def get_trending_tokens(self, limit=100):
        url = f"{Config.BIRDEYE_BASE_URL}/defi/token_trending?sort_by=rank&sort_type=asc&offset=0&limit={limit}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tokens = data.get('data', {}).get('tokens', [])
                        logger.info(f"Fetched {len(tokens)} trending tokens.")
                        return tokens
                    else:
                        logger.error(f"Failed to fetch trending: {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Error fetching trending: {e}")
                return []

    async def get_token_history(self, session, address, days=30):
        time_to = int(datetime.now().timestamp())
        time_from = int((datetime.now() - timedelta(days=days)).timestamp())
        url = f"{Config.BIRDEYE_BASE_URL}/defi/ohlcv"
        params = {
            "address": address,
            "type": Config.TIMEFRAME,
            "time_from": time_from,
            "time_to": time_to
        }
        last_status = None
        last_remaining = None
        for attempt in range(self.MAX_RETRIES):
            async with self.semaphore:
                try:
                    async with session.get(url, params=params) as resp:
                        last_status = resp.status
                        try:
                            last_remaining = resp.headers.get('x-ratelimit-remaining') or resp.headers.get('X-RateLimit-Remaining') or resp.headers.get('remaining')
                        except Exception:
                            last_remaining = None
                        if resp.status == 200:
                            data = await resp.json()
                            items = data.get('data', {}).get('items', [])
                            if not items:
                                return None
                            formatted = []
                            for item in items:
                                dt = datetime.fromtimestamp(item['unixTime'])
                                formatted.append((
                                    dt,
                                    address,
                                    float(item['o']),
                                    float(item['h']),
                                    float(item['l']),
                                    float(item['c']),
                                    float(item['v']),
                                    0.0
                                ))
                            return formatted
                        elif resp.status == 429:
                            quota_msg = f" remaining={last_remaining}" if last_remaining is not None else ""
                            if attempt < self.MAX_RETRIES - 1:
                                backoff = 2 * (2 ** attempt)
                                logger.warning(f"Rate limited for {address}{quota_msg}, backoff {backoff}s attempt {attempt+1}/{self.MAX_RETRIES}")
                            else:
                                logger.warning(f"Rate limited for {address}{quota_msg}, max retries exceeded")
                        else:
                            logger.warning(f"Error {resp.status} for {address}")
                            return None
                except Exception as e:
                    logger.error(f"Exception for {address}: {e}")
                    return None
            if attempt < self.MAX_RETRIES - 1 and last_status == 429:
                backoff = 2 * (2 ** attempt)
                await asyncio.sleep(backoff)
            elif last_status != 429:
                break
        return None
