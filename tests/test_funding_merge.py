"""Tests for the merged funding source (data_pipeline/providers/funding_merge.py).

The merge has to stay honest about two things: it must never invent a funding
observation, and it must keep both sources visible so a later reader can tell
which number came from where. These tests pin both.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_pipeline.providers import funding_merge as fm


def test_binance_backbone_covers_locked_basket_coins():
    """ETC/ATOM/KAS have no ByKaranteli row at all. Binance is the only source
    for them, so the backbone must carry them or the factor silently drops 3 of
    the 5 locked basket coins."""
    for coin in ("ETC", "TRX", "ATOM", "APT", "KAS"):
        table, meta = fm.load_binance_funding(coin)
        assert meta["status"] == "ok", f"{coin}: {meta}"
        assert len(table) > 2000, f"{coin} only has {len(table)} events"


def test_binance_returns_empty_not_zero_when_missing():
    """A coin with no stored file must yield an empty table, not zeros.
    Absence of funding data is not evidence of zero funding cost."""
    table, meta = fm.load_binance_funding("NOPE_NOT_A_COIN")
    assert table == {}
    assert meta["status"] == "missing"


def test_merged_prefers_binance_and_fills_with_bykaranteli(tmp_path):
    """On a shared day Binance wins (deeper, 8h native). On a ByKaranteli-only
    day the merge must still produce a value rather than a gap."""
    bk_markets = {"BTC": {
        "2026-09-20": {"funding": 0.0005},   # shared day
        "2026-10-01": {"funding": -0.0002},  # bykar-only tail
    }}
    m = fm.merged_funding("BTC", bk_markets=bk_markets)

    assert m["funding_binance"]["2026-09-20"] != 0.0005
    assert m["funding_merged"]["2026-09-20"] == m["funding_binance"]["2026-09-20"]
    assert m["funding_merged"]["2026-10-01"] == -0.0002


def test_merged_records_provenance_for_both_sources():
    m = fm.merged_funding("BTC", bk_markets={"BTC": {"2026-10-01": {"funding": 1e-4}}})
    p = m["provenance"]
    assert p["binance"]["source"] == "binance"
    assert p["bykaranteli_days"] == 1
    assert p["binance_days"] > 0


def test_bykaranteli_percent_is_converted_to_fraction():
    """ByKaranteli publishes percent (0.01 == 1%). The merge stores fractions,
    matching the Binance history and accounting_28c.load_real_funding."""
    rows = [{"venue": "binance", "market": "perp", "symbol": "BTC",
             "date": "2026-10-01", "funding_rate_8h_pct": "0.01",
             "open_interest_usd": "100", "price_usd": "1", "volume_24h_usd": "5",
             "price_change_24h_pct": "0"}]
    fm._dataset_rows = lambda name: rows
    out = fm.load_bykaranteli_venue_markets()
    assert out["BTC"]["2026-10-01"]["funding"] == pytest.approx(0.0001)


def test_bykaranteli_row_without_funding_is_dropped_not_zeroed():
    rows = [{"venue": "binance", "market": "perp", "symbol": "BTC",
             "date": "2026-10-01", "funding_rate_8h_pct": "", "open_interest_usd": "100"}]
    fm._dataset_rows = lambda name: rows
    out = fm.load_bykaranteli_venue_markets()
    assert "funding" not in out["BTC"]["2026-10-01"]


def test_coverage_report_flags_bykaranteli_only_coins():
    rows = fm.coverage_report(coins=["BTC"], bk_markets={"BTC": {"2026-10-01": {"funding": 1e-4}}})
    assert len(rows) == 1
    assert rows[0]["bykar_days"] == 1
    assert rows[0]["merged_days"] > rows[0]["binance_days"]
