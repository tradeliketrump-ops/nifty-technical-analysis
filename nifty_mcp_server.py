"""
NIFTY 50 MCP Server.

Exposes 3 tools via stdio transport:
  1. get_nifty_ohlcv        — raw OHLCV JSON
  2. get_indicator_summary   — Coral, HMA, Elliott state
  3. get_nifty_analysis      — full BUY/SELL/HOLD verdict
"""

from __future__ import annotations

import json

import pandas as pd

from mcp.server.fastmcp import FastMCP

from data_provider import get_cached_data, get_latest_price
from indicators import (
    AnalysisVerdict,
    CoralTrendResult,
    ElliottWaveResult,
    HMAResult,
    ATRResult,
    ADXResult,
    IndicatorSummary,
    SignalVerdict,
    TrendDirection,
    WaveLabel,
    WaveType,
    compute_all_indicators,
    format_indicator_summary,
)

# ─── MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "nifty-technical-analyst",
    instructions="NIFTY 50 technical analysis server providing OHLCV data, "
    "Coral Trend, HMA, Elliott Wave, ATR (volatility), "
    "and ADX (trend strength) indicators.",
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _ohlcv_to_json(df: pd.DataFrame) -> str:
    """Convert OHLCV DataFrame to JSON string."""
    # Reset index so timestamp becomes a column, round values
    result = df.reset_index()
    result["index"] = result["index"].astype(str)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in result.columns:
            result[col] = result[col].round(2)
    return result.to_json(orient="records", date_format="iso")


def _build_analysis_verdict(summary: IndicatorSummary) -> AnalysisVerdict:
    """
    Build a BUY/SELL/HOLD verdict from the indicator summary.

    Incorporates Coral Trend, HMA, Elliott Wave, ADX trend strength,
    and ATR volatility into a unified score.
    """
    coral = summary.coral
    hma = summary.hma
    elliott = summary.elliott
    atr = summary.atr
    adx = summary.adx

    # Score-based aggregation (positive = bullish, negative = bearish)
    score = 0

    # Coral contribution (±2)
    if coral.direction == TrendDirection.BULLISH:
        score += 2
    elif coral.direction == TrendDirection.BEARISH:
        score -= 2

    # HMA contribution (±2)
    if hma.crossover.value == "bullish_cross":
        score += 2
    elif hma.crossover.value == "bearish_cross":
        score -= 2

    # HMA slope contribution (±1)
    if hma.slope == "steep_up":
        score += 1
    elif hma.slope == "steep_down":
        score -= 1

    # Elliott exhaustion (penalty)
    if elliott.count_exhaustion:
        score -= 1

    # Coral position nuance
    if coral.price_position == "above":
        score += 1
    elif coral.price_position == "below":
        score -= 1

    # ADX contribution: strong trend amplifies direction
    if adx.trend_strength == "strong":
        score += 1 if adx.direction == "bullish" else -1
    elif adx.trend_strength == "weak":
        score -= 1  # Weak trend reduces conviction

    # ATR contribution: high volatility adds caution
    if atr.relative_strength == "high":
        score -= 1  # Caution in high volatility
    elif atr.relative_strength == "low":
        score += 1  # Low vol supports trend continuation

    # ── Verdict ──────────────────────────────────────────────────────
    if score >= 3:
        signal = SignalVerdict.BUY
        core_thesis = "Bullish alignment across trend, momentum, ADX, and wave structure."
        market_nuance = (
            f"Strength from Coral Trend, HMA crossover, and {adx.trend_strength} "
            f"{adx.direction} ADX. Volatility is {atr.relative_strength}. "
            "Monitor for volume confirmation."
        )
    elif score <= -3:
        signal = SignalVerdict.SELL
        core_thesis = "Bearish alignment — trend, momentum, and ADX are all negative."
        market_nuance = (
            f"Coral and HMA both bearish with {adx.trend_strength} {adx.direction} ADX. "
            f"ATR is {atr.relative_strength}. "
            "Consider protective puts or reduced exposure."
        )
    else:
        signal = SignalVerdict.HOLD
        core_thesis = "Mixed signals — no clear directional edge."
        market_nuance = (
            f"Coral and HMA show conflicting or neutral readings. "
            f"ADX is {adx.trend_strength} ({adx.adx_value}), "
            f"volatility is {atr.relative_strength}. "
            "Wait for clearer setup."
        )

    # ── Text summaries ───────────────────────────────────────────────
    coral_summary = (
        f"Coral Trend is {coral.direction.value} "
        f"(price {coral.price_position} the line at {coral.value:.2f})."
    )
    hma_summary = (
        f"HMA fast={hma.fast_value:.2f}, slow={hma.slow_value:.2f}, "
        f"crossover={hma.crossover.value}, slope={hma.slope}."
    )
    elliott_summary = (
        f"Elliott Wave: {elliott.wave_label.value} ({elliott.wave_type.value}), "
        f"confidence={elliott.confirmation_strength:.2f}, "
        f"exhaustion={'yes' if elliott.count_exhaustion else 'no'}."
    )
    atr_summary = (
        f"ATR is {atr.value:.2f} ({atr.relative_strength} volatility)."
    )
    adx_summary = (
        f"ADX is {adx.adx_value:.2f} ({adx.trend_strength}), "
        f"{adx.direction.capitalize()} direction "
        f"(+DI={adx.plus_di:.2f}, -DI={adx.minus_di:.2f})."
    )

    return AnalysisVerdict(
        signal=signal,
        core_thesis=core_thesis,
        market_nuance=market_nuance,
        last_price=summary.last_price,
        coral_summary=coral_summary,
        hma_summary=hma_summary,
        elliott_summary=elliott_summary,
        atr_summary=atr_summary,
        adx_summary=adx_summary,
    )


# ─── MCP Tools ─────────────────────────────────────────────────────────


@mcp.tool(
    name="get_nifty_ohlcv",
    description="Returns raw OHLCV (Open, High, Low, Close, Volume) data for NIFTY 50 (^NSEI) as JSON.",
)
def handle_get_nifty_ohlcv() -> str:
    """Tool handler: fetch OHLCV and return as JSON."""
    df: pd.DataFrame = get_cached_data()
    return _ohlcv_to_json(df)


@mcp.tool(
    name="get_indicator_summary",
    description="Returns the current Coral Trend, HMA, and Elliott Wave indicator states as JSON.",
)
def handle_get_indicator_summary() -> str:
    """Tool handler: compute indicators and return serialised summary."""
    df: pd.DataFrame = get_cached_data()
    summary = compute_all_indicators(df)
    return json.dumps(format_indicator_summary(summary), indent=2)


@mcp.tool(
    name="get_nifty_analysis",
    description="Returns the full BUY/SELL/HOLD analysis verdict with commentary for NIFTY 50.",
)
def handle_get_nifty_analysis() -> str:
    """Tool handler: compute indicators and return the full analysis verdict."""
    df: pd.DataFrame = get_cached_data()
    summary = compute_all_indicators(df)
    verdict = _build_analysis_verdict(summary)

    return json.dumps(
        {
            "signal": verdict.signal.value,
            "last_price": verdict.last_price,
            "core_thesis": verdict.core_thesis,
            "market_nuance": verdict.market_nuance,
            "coral_summary": verdict.coral_summary,
            "hma_summary": verdict.hma_summary,
            "elliott_summary": verdict.elliott_summary,
            "atr_summary": verdict.atr_summary,
            "adx_summary": verdict.adx_summary,
        },
        indent=2,
    )


# ─── Entry Point ───────────────────────────────────────────────────────


def main() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()