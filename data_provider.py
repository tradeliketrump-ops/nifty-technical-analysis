"""
Data provider for NIFTY 50 analysis.

Fetches ^NSEI OHLCV data from Yahoo Finance with caching (60s TTL).
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# ─── Module-level cache ────────────────────────────────────────────────
_cache: dict = {
    "data": None,
    "timestamp": 0.0,  # time.time() when cached
}

CACHE_TTL_SECONDS = 60
NIFTY_TICKER = "^NSEI"

# ─── Public API ────────────────────────────────────────────────────────


def fetch_nifty_ohlcv(
    period: str = "1mo", interval: str = "15m"
) -> pd.DataFrame:
    """
    Fetch NIFTY 50 OHLCV data from Yahoo Finance.

    Parameters
    ----------
    period : str, optional
        Data period to fetch (default "1mo").
    interval : str, optional
        Bar interval (default "15m").

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
        Index is DatetimeIndex.
    """
    ticker = yf.Ticker(NIFTY_TICKER)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {NIFTY_TICKER}")

    # Standardise column names (yfinance returns capitalised names)
    df.columns = [col.capitalize() for col in df.columns]

    return df


def get_cached_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return cached NIFTY OHLCV data, fetching fresh data if the cache
    is expired or force_refresh is True.

    Parameters
    ----------
    force_refresh : bool, optional
        If True, bypass cache and fetch new data.

    Returns
    -------
    pd.DataFrame
    """
    now = time.time()
    if not force_refresh and _cache["data"] is not None:
        if now - _cache["timestamp"] < CACHE_TTL_SECONDS:
            return _cache["data"]

    df = fetch_nifty_ohlcv()
    _cache["data"] = df
    _cache["timestamp"] = now
    return df


def get_latest_price(df: Optional[pd.DataFrame] = None) -> float:
    """
    Extract the most recent closing price.

    Parameters
    ----------
    df : pd.DataFrame or None
        OHLCV data. If None, fetches cached data.

    Returns
    -------
    float
    """
    if df is None:
        df = get_cached_data()
    return float(df["Close"].iloc[-1])


def clear_cache() -> None:
    """Force-clear the module-level cache (useful for testing)."""
    _cache["data"] = None
    _cache["timestamp"] = 0.0