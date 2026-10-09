"""ByKaranteli crypto-derivatives data provider (key-rotating client).

Reads BYKARANTELI_API_KEYS (comma-separated bk_... keys) and rotates across them,
so the 30 req/min per-key rate limit and the 15,000/month per-key quota are
spread across every key in the pool.

Only 4 endpoints answer without a key: /api/v1/public/manifest, /health,
/sitemap.json and the /api/x402 catalog. Everything under /api/public/* and
every CC0 dataset under /api/v1/public/datasets/* needs x-api-key.

Uses stdlib urllib only: the project venv ships aiohttp but not requests.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from loguru import logger

BASE = "https://bykaranteli.com"

# Endpoints that answer without a key; a key is still sent when one is present.
KEYLESS = (
    "/api/v1/public/manifest",
    "/api/v1/public/health",
    "/api/v1/public/sitemap.json",
    "/api/x402",
)

# CC0 dataset names available under /api/v1/public/datasets/<name>.{csv,json}
DATASETS = (
    "funding-daily",
    "venue-markets-daily",
    "positioning-daily",
    "liquidations-daily",
    "liquidations-largest-daily",
    "borrow-rates",
    "orderbook-daily",
    "rsi-daily",
    "metric-daily",
    "premium-daily",
    "cot-weekly",
    "etf-flows",
    "venue-share-daily",
    "venue-fees",
    "venue-settlements",
    "insurance-funds-daily",
    "options-oi-daily",
    "options-chain-hourly",
    "slippage-hourly",
    "whale-trades-daily",
    "hl-whales-daily",
    "hl-position-levels",
    "basis-daily",
    "market-profile-daily",
    "jupiter-daily",
    "solana-perps-daily",
    "listings",
    "daily-market",
    "daily-symbols",
)


class ByKaranteliError(RuntimeError):
    """Raised when every key and retry is exhausted."""


class ByKaranteliClient:
    """Round-robin client over BYKARANTELI_API_KEYS.

    Thread-safe: the rotation index and the per-key cooldown table are guarded
    by one lock, so concurrent fetchers do not pile onto the same key.
    """

    def __init__(self, keys=None, base=BASE, min_interval=0.12, timeout=30):
        raw = keys if keys is not None else os.getenv("BYKARANTELI_API_KEYS", "")
        if isinstance(raw, str):
            raw = [k.strip() for k in raw.split(",")]
        self.keys = [k for k in raw if k]
        self.base = base.rstrip("/")
        # 30 req/min per key is 2s apart; min_interval guards the burst edge.
        self.min_interval = min_interval
        self.timeout = timeout
        self._idx = 0
        self._last_used = [0.0] * len(self.keys)
        self._lock = threading.Lock()
        if not self.keys:
            logger.warning(
                "BYKARANTELI_API_KEYS is empty; only the 4 keyless endpoints work."
            )

    # --- key rotation ---------------------------------------------------

    def _next_key(self):
        """Return (key, index), advancing the rotation and honouring cooldown."""
        n = len(self.keys)
        if n == 0:
            return None, -1
        now = time.monotonic()
        with self._lock:
            for offset in range(n):
                i = (self._idx + offset) % n
                if now - self._last_used[i] >= self.min_interval:
                    self._idx = (i + 1) % n
                    self._last_used[i] = now
                    return self.keys[i], i
            # Every key is inside the cooldown: take the least recently used.
            i = min(range(n), key=lambda j: self._last_used[j])
            self._last_used[i] = now
            return self.keys[i], i

    # --- request helpers -------------------------------------------------

    def _open(self, path, params=None):
        url = f"{self.base}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        key, idx = self._next_key()
        headers = {"accept": "application/json", "user-agent": "AlphaGPT/1.0"}
        if key:
            headers["x-api-key"] = key
        req = urllib.request.Request(url, headers=headers, method="GET")
        return urllib.request.urlopen(req, timeout=self.timeout), idx

    def get_text(self, path, params=None, retries=2):
        """GET and return the body as text. Rotates on 401/403/429 and on 5xx."""
        last = None
        for attempt in range(retries + 1):
            try:
                resp, idx = self._open(path, params)
                body = resp.read().decode("utf-8", "replace")
                resp.close()
                return body
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                last = f"HTTP {e.code} {detail}"
                if e.code in (401, 403, 429):
                    # Bad, unverified or throttled key: rotate and retry.
                    time.sleep(0.6 * (attempt + 1))
                    continue
                if e.code >= 500:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ByKaranteliError(f"{path} -> {last}") from e
            except urllib.error.URLError as e:
                last = f"{type(e.reason).__name__}: {e.reason}"
                time.sleep(0.5 * (attempt + 1))
        raise ByKaranteliError(f"{path} failed after {retries + 1} tries: {last}")

    def get_json(self, path, params=None, retries=2):
        return json.loads(self.get_text(path, params, retries))

    # --- dataset / factor helpers ---------------------------------------

    def dataset_text(self, name, fmt="csv", retries=2, **params):
        """Fetch one CC0 dataset. name is given without extension."""
        ext = "" if "." in name else f".{fmt}"
        path = f"/api/v1/public/datasets/{name}{ext}"
        return self.get_text(path, params or None, retries)

    def dataset_json(self, name, **params):
        return json.loads(self.dataset_text(name, "json", **params))

    def usage(self):
        return self.get_json("/api/v1/usage")


_CLIENT = None


def get_client(**kwargs):
    """Process-wide singleton so the rotation state is shared."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ByKaranteliClient(**kwargs)
    return _CLIENT


# --- factor-oriented shortcuts ----------------------------------------

def fetch_funding_daily(**p):
    """Settled funding history per venue/symbol -> funding_rate factor."""
    return get_client().dataset_text("funding-daily", "csv", **p)


def fetch_venue_markets(**p):
    """Price, 24h volume, OI and funding per venue -> OI + funding factors."""
    return get_client().dataset_text("venue-markets-daily", "csv", **p)


def fetch_positioning(**p):
    """Long/short account and position ratios, taker buy/sell -> positioning."""
    return get_client().dataset_text("positioning-daily", "csv", **p)


def fetch_liquidations(**p):
    """Daily long/short liquidation totals -> liquidation pressure factor."""
    return get_client().dataset_text("liquidations-daily", "csv", **p)


def fetch_borrow_rates(**p):
    """Borrow/funding cost series -> cost-of-carry factor."""
    return get_client().dataset_text("borrow-rates", "csv", **p)


def fetch_funding_heatmap():
    return get_client().get_json("/api/public/heatmap")


def fetch_oi_board():
    return get_client().get_json("/api/public/oi")


def fetch_funding_arb():
    return get_client().get_json("/api/public/funding-arb")


def fetch_pressure(symbol=None):
    path = "/api/public/pressure" if not symbol else f"/api/public/pressure/{symbol}"
    return get_client().get_json(path)


def fetch_indices():
    return get_client().get_json("/api/public/indices")
