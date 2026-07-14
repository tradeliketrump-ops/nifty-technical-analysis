"""
SMA50 Trend Filter for signal validation.

Simple rule:
  - BUY only if Close > SMA50 (price is above the 50-period moving average)
  - SELL only if Close < SMA50 (price is below the 50-period moving average)
  - HOLD if the condition for the signal direction is not met

This acts as a trend confirmation layer before the scoring engine's verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from indicators import AnalysisVerdict, IndicatorSummary, SignalVerdict

logger = logging.getLogger("nifty_sma50")

SMA50_PERIOD = 50


@dataclass
class SMA50Result:
    value: float          # The SMA50 value
    close_above: bool     # True if close > SMA50
    close_below: bool     # True if close < SMA50
    price_position: str   # "above", "below", "exactly_at"


def compute_sma50(df: pd.DataFrame) -> SMA50Result:
    """
    Compute the 50-period Simple Moving Average of the close price.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.

    Returns
    -------
    SMA50Result
    """
    if df.empty or len(df) < SMA50_PERIOD:
        return SMA50Result(
            value=0.0,
            close_above=False,
            close_below=False,
            price_position="unknown",
        )

    sma50 = df["Close"].rolling(SMA50_PERIOD).mean()
    current_close = float(df["Close"].iloc[-1])
    current_sma = float(sma50.iloc[-1])

    if current_close > current_sma:
        return SMA50Result(
            value=round(current_sma, 2),
            close_above=True,
            close_below=False,
            price_position="above",
        )
    elif current_close < current_sma:
        return SMA50Result(
            value=round(current_sma, 2),
            close_above=False,
            close_below=True,
            price_position="below",
        )
    else:
        return SMA50Result(
            value=round(current_sma, 2),
            close_above=False,
            close_below=False,
            price_position="exactly_at",
        )


def apply_sma50_filter(
    raw_signal: SignalVerdict,
    sma50: SMA50Result,
) -> SignalVerdict:
    """
    Apply the SMA50 trend filter to a raw signal.

    Rules:
      - BUY is only allowed if Close > SMA50
      - SELL is only allowed if Close < SMA50
      - If the condition is not met, downgrade to HOLD

    Parameters
    ----------
    raw_signal : SignalVerdict
        The signal before SMA50 filtering.
    sma50 : SMA50Result
        The SMA50 result.

    Returns
    -------
    SignalVerdict
        The filtered signal (may stay same or become HOLD).
    """
    if raw_signal == SignalVerdict.BUY and not sma50.close_above:
        logger.info(
            "SMA50 filter: BUY rejected — Close below SMA50 (%.2f). Downgraded to HOLD.",
            sma50.value,
        )
        return SignalVerdict.HOLD

    if raw_signal == SignalVerdict.SELL and not sma50.close_below:
        logger.info(
            "SMA50 filter: SELL rejected — Close above SMA50 (%.2f). Downgraded to HOLD.",
            sma50.value,
        )
        return SignalVerdict.HOLD

    return raw_signal