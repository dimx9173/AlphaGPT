import asyncio
import base64
import time
import aiohttp
from loguru import logger
from solders.transaction import VersionedTransaction
from .config import ExecutionConfig


class JupiterAggregator:
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=5)
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.2
    CIRCUIT_THRESHOLD = 5
    CIRCUIT_COOLDOWN = 30.0

    def __init__(self, timeout=None, max_retries=None, circuit_threshold=None, circuit_cooldown=None):
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        self.circuit_threshold = circuit_threshold if circuit_threshold is not None else self.CIRCUIT_THRESHOLD
        self.circuit_cooldown = circuit_cooldown if circuit_cooldown is not None else self.CIRCUIT_COOLDOWN
        self._consecutive_fails = 0
        self._circuit_open_until = 0.0
        self._last_quote_cache = {}
        self._last_swap_cache = {}

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    def _is_circuit_open(self):
        return time.monotonic() < self._circuit_open_until

    def _record_success(self):
        self._consecutive_fails = 0
        self._circuit_open_until = 0.0

    def _record_failure(self):
        self._consecutive_fails += 1
        if self._consecutive_fails >= self.circuit_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_cooldown
            logger.warning(f"Jupiter circuit breaker OPEN for {self.circuit_cooldown}s after {self._consecutive_fails} consecutive fails")

    async def _sleep_backoff(self, attempt, headers=None):
        if headers is not None:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                    await asyncio.sleep(min(delay, 5.0))
                    return
                except ValueError:
                    pass
        delay = self.BACKOFF_FACTOR * (2 ** attempt)
        delay = min(delay, 2.0)
        await asyncio.sleep(delay)

    def _cache_key(self, input_mint, output_mint, amount_integer):
        return (str(input_mint), str(output_mint), str(amount_integer))

    async def _fallback_quote(self, input_mint, output_mint, amount_integer):
        key = self._cache_key(input_mint, output_mint, amount_integer)
        try:
            from data_pipeline.config import Config as DPConfig
            api_key = getattr(DPConfig, "BIRDEYE_API_KEY", "")
            base_url = getattr(DPConfig, "BIRDEYE_BASE_URL", "https://public-api.birdeye.so")
            if api_key:
                timeout = aiohttp.ClientTimeout(total=3)
                url = f"{base_url}/defi/price"
                headers = {"X-API-KEY": api_key, "accept": "application/json"}
                address = str(input_mint) if str(output_mint) == ExecutionConfig.SOL_MINT else str(output_mint)
                params = {"address": address}
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as sess:
                    async with sess.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            inner = data.get("data", data)
                            value = inner.get("value") if isinstance(inner, dict) else None
                            if value is not None:
                                try:
                                    price = float(value)
                                    quote = {"outAmount": str(int(price * 1e9)), "_fallback": "birdeye", "price": price}
                                    logger.info(f"Jupiter fallback Birdeye price {price} for {address}")
                                    return quote
                                except (ValueError, TypeError):
                                    pass
        except Exception as e:
            logger.debug(f"Birdeye fallback failed: {e}")
        cached = self._last_quote_cache.get(key)
        if cached is not None:
            logger.info(f"Jupiter fallback cache hit for {key}")
            return cached
        return None

    async def get_quote(self, input_mint, output_mint, amount_integer, slippage_bps=None):
        if self._is_circuit_open():
            logger.warning("Jupiter circuit OPEN, short-circuiting get_quote to fallback")
            fb = await self._fallback_quote(input_mint, output_mint, amount_integer)
            return fb

        slippage = slippage_bps if slippage_bps else ExecutionConfig.DEFAULT_SLIPPAGE_BPS
        url = f"{self.base_url}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_integer),
            "slippageBps": str(slippage),
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                async with session.get(url, params=params, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        key = self._cache_key(input_mint, output_mint, amount_integer)
                        self._last_quote_cache[key] = data
                        self._record_success()
                        return data
                    if resp.status == 429 or 500 <= resp.status < 600:
                        text = await resp.text()
                        logger.warning(f"Jupiter Quote retryable {resp.status} attempt {attempt+1}/{self.max_retries}: {text[:200]}")
                        if attempt < self.max_retries - 1:
                            await self._sleep_backoff(attempt, resp.headers)
                            continue
                        self._record_failure()
                        break
                    text = await resp.text()
                    logger.error(f"Jupiter Quote Error {resp.status}: {text[:500]}")
                    self._record_failure()
                    break
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                logger.warning(f"Jupiter Quote timeout/client error attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await self._sleep_backoff(attempt)
                    continue
                self._record_failure()
                break
            except Exception as e:
                last_exc = e
                logger.error(f"Jupiter Quote unexpected error: {e}")
                self._record_failure()
                break

        fb = await self._fallback_quote(input_mint, output_mint, amount_integer)
        if fb is not None:
            return fb
        if last_exc is not None:
            logger.debug(f"Jupiter get_quote all retries exhausted: {last_exc}")
        return None

    async def get_swap_tx(self, quote_response):
        if self._is_circuit_open():
            logger.warning("Jupiter circuit OPEN, short-circuiting get_swap_tx")
            return None
        url = f"{self.base_url}/swap"
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": ExecutionConfig.get_wallet_address(),
            "wrapAndUnwrapSol": True,
            "computeUnitPriceMicroLamports": "auto",
            "prioritizationFeeLamports": "auto"
        }
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tx = data.get("swapTransaction")
                        self._record_success()
                        if tx:
                            self._last_swap_cache["last"] = tx
                        return tx
                    if resp.status == 429 or 500 <= resp.status < 600:
                        text = await resp.text()
                        logger.warning(f"Jupiter Swap retryable {resp.status} attempt {attempt+1}/{self.max_retries}: {text[:200]}")
                        if attempt < self.max_retries - 1:
                            await self._sleep_backoff(attempt, resp.headers)
                            continue
                        self._record_failure()
                        break
                    text = await resp.text()
                    logger.error(f"Jupiter Swap API Error {resp.status}: {text[:500]}")
                    self._record_failure()
                    break
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Jupiter Swap timeout/client error attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await self._sleep_backoff(attempt)
                    continue
                self._record_failure()
                break
            except Exception as e:
                logger.error(f"Jupiter Swap unexpected error: {e}")
                self._record_failure()
                break
        return None

    async def close(self):
        if self.session:
            await self.session.close()

    @staticmethod
    def deserialize_and_sign(b64_tx_str):
        try:
            tx_bytes = base64.b64decode(b64_tx_str)
            txn = VersionedTransaction.from_bytes(tx_bytes)
            signature = ExecutionConfig.get_payer_keypair().sign_message(txn.message.to_bytes())
            txn = VersionedTransaction.populate(txn.message, [signature])
            return txn
        except Exception as e:
            logger.error(f"Signing Error: {e}")
            raise
