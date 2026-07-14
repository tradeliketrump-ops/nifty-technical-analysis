"""
Data provider for NIFTY 50 analysis.

Fetches ^NSEI OHLCV data from Yahoo Finance with caching (60s TTL)
and automatic retry on failure (for cloud deployments like Render
where network issues may cause transient failures).
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger("nifty_data")

# ─── Module-level cache ────────────────────────────────────────────────
_cache: dict = {
    "data": None,
    "timestamp": 0.0,  # time.time() when cached
    "error_count": 0,   # consecutive failures
}

CACHE_TTL_SECONDS = 60
NIFTY_TICKER = "^NSEI"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # doubles each retry

# ─── Public API ────────────────────────────────────────────────────────


def fetch_nifty_ohlcv(
    period: str = "1mo", interval: str = "15m"
) -> pd.DataFrame:
    """
    Fetch NIFTY 50 OHLCV data from Yahoo Finance with retry logic.

    Parameters
    ----------
    period : str, optional
        Data period to fetch (default "1mo").
    interval : str, optional
        Bar interval (default "15m").

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If all retries fail or no data is returned.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ticker = yf.Ticker(NIFTY_TICKER)
            df = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                timeout=15,
            )

            if df.empty:
                raise ValueError(f"No data returned for {NIFTY_TICKER}")

            # Standardise column names (yfinance returns capitalised names)
            df.columns = [col.capitalize() for col in df.columns]

            logger.info(
                "Data fetch successful (attempt %d/%d): %d rows",
                attempt, MAX_RETRIES, len(df),
            )
            return df

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Data fetch attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.info("Retrying in %d seconds...", wait)
                time.sleep(wait)

    raise ValueError(
        f"Failed to fetch data after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def get_cached_data(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Return cached NIFTY OHLCV data, fetching fresh data if the cache
    is expired or force_refresh is True.

    Unlike the previous version, this returns None if fetch fails
    (allowing the caller to decide how to handle it) instead of
    crashing the entire analysis.

    Parameters
    ----------
    force_refresh : bool, optional
        If True, bypass cache and fetch new data.

    Returns
    -------
    pd.DataFrame or None
    """
    now = time.time()
    if not force_refresh and _cache["data"] is not None:
        if now - _cache["timestamp"] < CACHE_TTL_SECONDS:
            return _cache["data"]

    try:
        df = fetch_nifty_ohlcv()
        _cache["data"] = df
        _cache["timestamp"] = now
        _cache["error_count"] = 0
        return df
    except Exception as exc:
        _cache["error_count"] += 1
        logger.error("Fetch failed (%d consecutive): %s", _cache["error_count"], exc)
        # Return stale cache if we have it, even if expired
        if _cache["data"] is not None:
            logger.info("Returning stale cached data as fallback.")
            return _cache["data"]
        return None


def get_latest_price(df: Optional[pd.DataFrame] = None) -> Optional[float]:
    """
    Extract the most recent closing price.

    Parameters
    ----------
    df : pd.DataFrame or None
        OHLCV data. If None, fetches cached data.

    Returns
    -------
    float or None
    """
    if df is None:
        df = get_cached_data()
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def clear_cache() -> None:
    """Force-clear the module-level cache (useful for testing)."""
    _cache["data"] = None
    _cache["timestamp"] = 0.0
    _cache["error_count"] = 0
