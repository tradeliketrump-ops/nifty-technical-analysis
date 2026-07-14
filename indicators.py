"""
NIFTY 50 Technical Indicators.

Pure computation functions for:
  - Coral Trend (smoothed ATR-based channel)
  - Hull Moving Average (HMA) with crossover / slope detection
  - Elliott Wave (1-5 / A-B-C) with rules-based classification
  - ATR (Average True Range) — volatility measurement
  - ADX (Average Directional Index) — trend strength indicator

All functions are pure — no I/O, no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
#  Enums & Literals
# ────────────────────────────────────────────────────────────────────────


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MomentumState(Enum):
    BULLISH_CROSS = "bullish_cross"
    BEARISH_CROSS = "bearish_cross"
    NEUTRAL = "neutral"


class WaveType(Enum):
    IMPULSE = "impulsive"
    CORRECTIVE = "corrective"


class WaveLabel(Enum):
    WAVE_1 = "Wave 1"
    WAVE_2 = "Wave 2"
    WAVE_3 = "Wave 3"
    WAVE_4 = "Wave 4"
    WAVE_5 = "Wave 5"
    WAVE_A = "Wave A"
    WAVE_B = "Wave B"
    WAVE_C = "Wave C"
    UNKNOWN = "Unknown"


class SignalVerdict(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ────────────────────────────────────────────────────────────────────────
#  Data Containers
# ────────────────────────────────────────────────────────────────────────


@dataclass
class CoralTrendResult:
    value: float
    price_position: str  # "above", "below", "crossing"
    direction: TrendDirection


@dataclass
class HMAResult:
    fast_value: float
    slow_value: float
    crossover: MomentumState
    slope: str  # "steep_up", "flat", "steep_down"


@dataclass
class ElliottWaveResult:
    wave_label: WaveLabel
    wave_type: WaveType
    confirmation_strength: float  # 0.0 to 1.0
    count_exhaustion: bool


@dataclass
class ATRResult:
    """Average True Range — volatility measurement."""
    value: float                     # Current ATR value (in price points)
    relative_strength: str           # "high", "normal", "low" vs recent history


@dataclass
class ADXResult:
    """Average Directional Index — trend strength."""
    adx_value: float                 # ADX reading (0-100)
    plus_di: float                   # +DI line
    minus_di: float                  # -DI line
    trend_strength: str              # "strong", "moderate", "weak"
    direction: str                   # "bullish" (+DI > -DI), "bearish" (-DI > +DI)


@dataclass
class IndicatorSummary:
    timestamp: datetime
    last_price: float
    coral: CoralTrendResult
    hma: HMAResult
    elliott: ElliottWaveResult
    atr: ATRResult
    adx: ADXResult


@dataclass
class AnalysisVerdict:
    signal: SignalVerdict
    core_thesis: str
    market_nuance: str
    last_price: float
    coral_summary: str
    hma_summary: str
    elliott_summary: str
    atr_summary: str
    adx_summary: str


# ────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ────────────────────────────────────────────────────────────────────────


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


# ────────────────────────────────────────────────────────────────────────
#  Coral Trend
# ────────────────────────────────────────────────────────────────────────


def compute_coral_trend(df: pd.DataFrame) -> CoralTrendResult:
    """
    Calculate Coral Trend — a smoothed ATR-based channel.

    The indicator uses an EMA as the core line and ATR to determine
    trend state by comparing price position relative to the smoothed
    line plus/minus ATR bands.

    Returns
    -------
    CoralTrendResult
    """
    if df.empty:
        return CoralTrendResult(
            value=0.0,
            price_position="unknown",
            direction=TrendDirection.NEUTRAL,
        )

    close = df["Close"]
    period = 10  # Coral period

    # Core line: EMA of close
    core = close.ewm(span=period, adjust=False).mean()

    # ATR bands
    atr = _true_range(df).rolling(14).mean()
    upper_band = core + atr
    lower_band = core - atr

    current_price = close.iloc[-1]
    current_core = core.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]

    # Determine position relative to bands
    if current_price > current_upper:
        price_position = "above"
        direction = TrendDirection.BULLISH
    elif current_price < current_lower:
        price_position = "below"
        direction = TrendDirection.BEARISH
    else:
        price_position = "crossing"
        direction = TrendDirection.NEUTRAL

    return CoralTrendResult(
        value=float(current_core),
        price_position=price_position,
        direction=direction,
    )


# ────────────────────────────────────────────────────────────────────────
#  Hull Moving Average
# ────────────────────────────────────────────────────────────────────────


def compute_hma(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 18,
) -> HMAResult:
    """
    Compute two Hull Moving Averages and detect crossover state.

    HMA formula:
      HMA = WMA(2 * WMA(period/2) - WMA(period), sqrt(period))

    Slope classification thresholds (steep if |diff| > 0.1% of price).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    fast_period : int
        Period for the fast HMA (default 9).
    slow_period : int
        Period for the slow HMA (default 18).

    Returns
    -------
    HMAResult
    """
    if df.empty:
        return HMAResult(
            fast_value=0.0,
            slow_value=0.0,
            crossover=MomentumState.NEUTRAL,
            slope="flat",
        )

    close = df["Close"]

    def _compute_hma_series(series: pd.Series, period: int) -> pd.Series:
        half_period = max(period // 2, 1)
        wma_half = _wma(series, half_period)
        wma_full = _wma(series, period)
        raw = 2 * wma_half - wma_full
        hma = _wma(raw, int(math.sqrt(period)))
        return hma

    fast_hma = _compute_hma_series(close, fast_period)
    slow_hma = _compute_hma_series(close, slow_period)

    fast_val = float(fast_hma.iloc[-1])
    slow_val = float(slow_hma.iloc[-1])

    # Crossover detection — compare the last two bars
    prev_fast = float(fast_hma.iloc[-2]) if len(fast_hma) >= 2 else fast_val
    prev_slow = float(slow_hma.iloc[-2]) if len(slow_hma) >= 2 else slow_val

    if prev_fast <= prev_slow and fast_val > slow_val:
        crossover = MomentumState.BULLISH_CROSS
    elif prev_fast >= prev_slow and fast_val < slow_val:
        crossover = MomentumState.BEARISH_CROSS
    else:
        crossover = MomentumState.NEUTRAL

    # Slope detection — use fast HMA's recent rate of change
    lookback = 3
    if len(fast_hma) > lookback:
        delta = fast_val - float(fast_hma.iloc[-1 - lookback])
        threshold = close.iloc[-1] * 0.001  # 0.1% of price
        if delta > threshold:
            slope = "steep_up"
        elif delta < -threshold:
            slope = "steep_down"
        else:
            slope = "flat"
    else:
        slope = "flat"

    return HMAResult(
        fast_value=fast_val,
        slow_value=slow_val,
        crossover=crossover,
        slope=slope,
    )


# ────────────────────────────────────────────────────────────────────────
#  Elliott Wave Detection
# ────────────────────────────────────────────────────────────────────────


def detect_swing_points(
    df: pd.DataFrame, lookback: int = 5
) -> list[dict]:
    """
    Detect swing highs and lows using local peak/trough detection.

    A swing high is a bar whose high is the highest over the
    surrounding ``lookback`` bars on each side. A swing low is a bar
    whose low is the lowest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High' and 'Low' columns.
    lookback : int
        Number of bars to look on each side (default 5).

    Returns
    -------
    list[dict]
        Each dict has keys: 'index', 'price', 'type' ('high' or 'low'),
        'timestamp'.
    """
    swings: list[dict] = []
    n = len(df)

    for i in range(lookback, n - lookback):
        # Swing high
        if all(df["High"].iloc[i] >= df["High"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["High"].iloc[i] >= df["High"].iloc[i + j] for j in range(1, lookback + 1)):
            swings.append({
                "index": i,
                "price": float(df["High"].iloc[i]),
                "type": "high",
                "timestamp": str(df.index[i]),
            })
        # Swing low
        if all(df["Low"].iloc[i] <= df["Low"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["Low"].iloc[i] <= df["Low"].iloc[i + j] for j in range(1, lookback + 1)):
            swings.append({
                "index": i,
                "price": float(df["Low"].iloc[i]),
                "type": "low",
                "timestamp": str(df.index[i]),
            })

    return swings


def classify_waves(
    swings: list[dict],
) -> tuple[list[WaveLabel], ElliottWaveResult]:
    """
    Apply Elliott Wave rules to labeled swing points.

    Rules enforced:
      1. Impulse (1-2-3-4-5): alternate high/low starting with low → high →
         low → high → low. Wave 3 is the longest and never the shortest
         (extended 3rd wave rule).
      2. Correction (A-B-C): swing high → low → high following an impulse.
      3. Retracement: Wave 2 retraces 50-61.8% of Wave 1;
         Wave 4 retraces 38.2-50% of Wave 3.
      4. Exhaustion: flagged in late Wave 5 or Wave C.

    Parameters
    ----------
    swings : list[dict]
        Detected swing points ordered by time.

    Returns
    -------
    tuple[list[WaveLabel], ElliottWaveResult]
        First element: labelled wave sequence for all identified swings.
        Second element: the current (most recent) ElliottWaveResult.
    """
    if len(swings) < 3:
        labels: list[WaveLabel] = [WaveLabel.UNKNOWN] * len(swings)
        return labels, ElliottWaveResult(
            wave_label=WaveLabel.UNKNOWN,
            wave_type=WaveType.CORRECTIVE,
            confirmation_strength=0.0,
            count_exhaustion=False,
        )

    labels: list[WaveLabel] = []
    current: WaveLabel = WaveLabel.UNKNOWN
    wave_type: WaveType = WaveType.CORRECTIVE
    exhaustion = False
    confidence = 0.3

    # Ensure alternating: must start with a low for impulse count
    first_type = swings[0]["type"]

    if first_type == "low" and len(swings) >= 5:
        # Try 5-wave impulse count
        labels = _label_impulse(swings)
        current = labels[-1]
        wave_type = WaveType.IMPULSE
        confidence = _impulse_confidence(swings, labels)

        # Check if we have room for corrective waves after impulse
        if len(swings) > len(labels):
            remaining = swings[len(labels):]
            correction_labels = _label_correction(remaining)
            labels.extend(correction_labels)
            if correction_labels:
                current = correction_labels[-1]
                wave_type = WaveType.CORRECTIVE
                # Lower confidence for correction
                confidence = max(confidence * 0.8, 0.3)
    else:
        # Could be in the middle of a formation — try correction first
        labels = _label_correction(swings)
        if labels:
            current = labels[-1]
            wave_type = WaveType.CORRECTIVE
            confidence = 0.4
        else:
            labels = [WaveLabel.UNKNOWN] * len(swings)
            current = WaveLabel.UNKNOWN

    # Exhaustion check
    if current in (WaveLabel.WAVE_5, WaveLabel.WAVE_C):
        exhaustion = True

    return labels, ElliottWaveResult(
        wave_label=current,
        wave_type=wave_type,
        confirmation_strength=round(confidence, 2),
        count_exhaustion=exhaustion,
    )


def _label_impulse(swings: list[dict]) -> list[WaveLabel]:
    """Label first 5 swings as a 1-2-3-4-5 impulse sequence."""
    wave_labels: list[WaveLabel] = []
    expected_labels = [
        WaveLabel.WAVE_1,
        WaveLabel.WAVE_2,
        WaveLabel.WAVE_3,
        WaveLabel.WAVE_4,
        WaveLabel.WAVE_5,
    ]

    # We need alternating types: low, high, low, high, low
    n = min(len(swings), 5)
    for i in range(n):
        wave_labels.append(expected_labels[i])

    # Pad if fewer than 5
    while len(wave_labels) < 5:
        wave_labels.append(WaveLabel.UNKNOWN)

    return wave_labels


def _label_correction(swings: list[dict]) -> list[WaveLabel]:
    """Label swings as A-B-C corrective pattern."""
    if len(swings) < 3:
        return []
    corr_labels: list[WaveLabel] = []
    abc_labels = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
    for i in range(min(len(swings), 3)):
        corr_labels.append(abc_labels[i])
    return corr_labels


def _impulse_confidence(swings: list[dict], labels: list[WaveLabel]) -> float:
    """
    Calculate confidence score for impulse wave labelling.

    Checks:
      - Extended 3rd wave: Wave 3 price move should be the largest
      - Retracement: Wave 2 back to 50-61.8% of Wave 1
    """
    if len(swings) < 5 or len(labels) < 5:
        return 0.3

    prices = [s["price"] for s in swings[:5]]
    # Wave magnitudes (absolute price change)
    wave_1_mag = abs(prices[1] - prices[0])
    wave_3_mag = abs(prices[3] - prices[2])
    wave_5_mag = abs(prices[4] - prices[3]) if len(prices) >= 5 else 0

    score = 0.0

    # 1) Wave 3 is the largest (> Wave 1 and Wave 5)
    if wave_3_mag > wave_1_mag and wave_3_mag > wave_5_mag:
        score += 0.4

    # 2) Wave 2 retracement: how far back toward Wave 1 start
    wave_1_range = abs(prices[1] - prices[0])
    if wave_1_range > 0:
        if swings[0]["type"] == "low":
            retrace_2 = (prices[1] - prices[2]) / wave_1_range
        else:
            retrace_2 = (prices[2] - prices[1]) / wave_1_range
        # Retrace between 0.382 and 0.786 is acceptable
        if 0.382 <= retrace_2 <= 0.786:
            score += 0.3

    # 3) Wave 4 retracement of Wave 3 (38.2-50%)
    if wave_3_mag > 0:
        if swings[2]["type"] == "low":
            retrace_4 = (prices[3] - prices[4]) / wave_3_mag
        else:
            retrace_4 = (prices[4] - prices[3]) / wave_3_mag
        if 0.382 <= retrace_4 <= 0.618:
            score += 0.2

    return min(score + 0.1, 1.0)  # Base 0.1


def compute_elliott_wave(df: pd.DataFrame) -> ElliottWaveResult:
    """
    Orchestrator: detect swing points, classify waves, return current state.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.

    Returns
    -------
    ElliottWaveResult
    """
    swings = detect_swing_points(df)
    _, result = classify_waves(swings)
    return result


# ────────────────────────────────────────────────────────────────────────
#  ATR (Average True Range)
# ────────────────────────────────────────────────────────────────────────


def compute_atr(df: pd.DataFrame, period: int = 14) -> ATRResult:
    """
    Compute Average True Range — a volatility measure.

    Uses Wilder's smoothing (modified EMA) on True Range.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    period : int
        Lookback period (default 14).

    Returns
    -------
    ATRResult
    """
    if df.empty:
        return ATRResult(value=0.0, relative_strength="normal")

    tr = _true_range(df)

    # Wilder's smoothing: first ATR is SMA, then EMA-like smoothing
    atr_values = tr.to_numpy(dtype=float, copy=True)
    atr_values[:] = 0.0
    atr_values[period] = tr.iloc[:period + 1].mean()
    for i in range(period + 1, len(atr_values)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr.iloc[i]) / period

    current_atr = float(atr_values[-1])

    # Relative strength: compare current ATR to its own 63-period median
    # to determine if volatility is high/low relative to recent history
    lookback = min(63, len(atr_values) - 1)
    if lookback > period:
        recent_atrs = atr_values[-lookback:]
        median_atr = float(np.median(recent_atrs))
        if median_atr > 0:
            ratio = current_atr / median_atr
            if ratio > 1.5:
                rel = "high"
            elif ratio < 0.7:
                rel = "low"
            else:
                rel = "normal"
        else:
            rel = "normal"
    else:
        rel = "normal"

    return ATRResult(value=round(current_atr, 2), relative_strength=rel)


# ────────────────────────────────────────────────────────────────────────
#  ADX (Average Directional Index)
# ────────────────────────────────────────────────────────────────────────


def compute_adx(df: pd.DataFrame, period: int = 14) -> ADXResult:
    """
    Compute Average Directional Index — trend strength indicator.

    ADX measures trend strength (not direction). +DI and -DI lines
    provide direction context.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    period : int
        Lookback period (default 14).

    Returns
    -------
    ADXResult
    """
    if df.empty:
        return ADXResult(
            adx_value=0.0, plus_di=0.0, minus_di=0.0,
            trend_strength="weak", direction="neutral",
        )

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(high)

    # True Range (use Wilder's)
    tr = _true_range(df).to_numpy(dtype=float)

    # Directional Movement
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            up_move[i] = up
        if down > up and down > 0:
            down_move[i] = down

    # Wilder's smoothing of TR, +DM, -DM
    atr_wild = np.zeros(n)
    plus_dm_wild = np.zeros(n)
    minus_dm_wild = np.zeros(n)

    atr_wild[period] = float(np.mean(tr[1:period + 1]))
    plus_dm_wild[period] = float(np.mean(up_move[1:period + 1]))
    minus_dm_wild[period] = float(np.mean(down_move[1:period + 1]))

    for i in range(period + 1, n):
        atr_wild[i] = (atr_wild[i - 1] * (period - 1) + tr[i]) / period
        plus_dm_wild[i] = (plus_dm_wild[i - 1] * (period - 1) + up_move[i]) / period
        minus_dm_wild[i] = (minus_dm_wild[i - 1] * (period - 1) + down_move[i]) / period

    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_wild[i] > 0:
            plus_di[i] = 100.0 * plus_dm_wild[i] / atr_wild[i]
            minus_di[i] = 100.0 * minus_dm_wild[i] / atr_wild[i]

    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX = Wilder's smoothing of DX
    adx_values = np.zeros(n)
    adx_values[period * 2] = float(np.mean(dx[period:period * 2 + 1]))
    for i in range(period * 2 + 1, n):
        adx_values[i] = (adx_values[i - 1] * (period - 1) + dx[i]) / period

    current_adx = float(adx_values[-1])
    current_plus_di = float(plus_di[-1])
    current_minus_di = float(minus_di[-1])

    # Classify trend strength
    if current_adx >= 25:
        strength = "strong"
    elif current_adx >= 20:
        strength = "moderate"
    else:
        strength = "weak"

    # Direction from +DI vs -DI
    if current_plus_di > current_minus_di:
        direction = "bullish"
    elif current_minus_di > current_plus_di:
        direction = "bearish"
    else:
        direction = "neutral"

    return ADXResult(
        adx_value=round(current_adx, 2),
        plus_di=round(current_plus_di, 2),
        minus_di=round(current_minus_di, 2),
        trend_strength=strength,
        direction=direction,
    )


# ────────────────────────────────────────────────────────────────────────
#  Top-level orchestrator
# ────────────────────────────────────────────────────────────────────────


def compute_all_indicators(df: pd.DataFrame) -> IndicatorSummary:
    """
    Run all five indicators and assemble an IndicatorSummary.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.

    Returns
    -------
    IndicatorSummary
    """
    coral = compute_coral_trend(df)
    hma = compute_hma(df)
    elliott = compute_elliott_wave(df)
    atr = compute_atr(df)
    adx = compute_adx(df)
    last_price = float(df["Close"].iloc[-1])

    return IndicatorSummary(
        timestamp=datetime.now(),
        last_price=last_price,
        coral=coral,
        hma=hma,
        elliott=elliott,
        atr=atr,
        adx=adx,
    )


def format_indicator_summary(summary: IndicatorSummary) -> dict[str, Any]:
    """
    Serialize an IndicatorSummary into a plain dict for JSON / MCP output.

    Parameters
    ----------
    summary : IndicatorSummary

    Returns
    -------
    dict
    """
    return {
        "timestamp": summary.timestamp.isoformat(),
        "last_price": summary.last_price,
        "coral": {
            "value": round(summary.coral.value, 2),
            "price_position": summary.coral.price_position,
            "direction": summary.coral.direction.value,
        },
        "hma": {
            "fast_value": round(summary.hma.fast_value, 2),
            "slow_value": round(summary.hma.slow_value, 2),
            "crossover": summary.hma.crossover.value,
            "slope": summary.hma.slope,
        },
        "elliott": {
            "wave_label": summary.elliott.wave_label.value,
            "wave_type": summary.elliott.wave_type.value,
            "confirmation_strength": summary.elliott.confirmation_strength,
            "count_exhaustion": summary.elliott.count_exhaustion,
        },
        "atr": {
            "value": summary.atr.value,
            "relative_strength": summary.atr.relative_strength,
        },
        "adx": {
            "adx_value": summary.adx.adx_value,
            "plus_di": summary.adx.plus_di,
            "minus_di": summary.adx.minus_di,
            "trend_strength": summary.adx.trend_strength,
            "direction": summary.adx.direction,
        },
    }
