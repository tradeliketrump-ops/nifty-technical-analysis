# Skill: NIFTY Technical Analyst

## Description
A discretionary technical analysis system for NIFTY 50 that combines Coral Trend, Hull Moving Average, and Elliott Wave analysis to produce BUY / SELL / HOLD signals with market context.

## Analysis Framework

### 1. Coral Trend (Macro Trend)
- **Calculation:** EMA-based smoothed line with ATR bands (14-period ATR)
- **States:** Bullish (price above upper band), Bearish (price below lower band), Neutral (price crossing within bands)
- **Role:** Determines the dominant trend direction

### 2. Hull Moving Average — HMA (Momentum)
- **Calculation:** Fast HMA (9-period) and Slow HMA (18-period)
- **Crossover States:** Bullish cross (fast crosses above slow), Bearish cross (fast crosses below slow), Neutral
- **Slope:** Steep up, flat, steep down
- **Role:** Confirms momentum alignment with trend

### 3. Elliott Wave (Wave Structure)
- **Detection:** Swing point identification → impulse 1-2-3-4-5 classification (with extended 3rd wave rule) → corrective A-B-C classification
- **Confidence:** Based on retracement ratios (38.2-61.8% Fibonacci) and wave 3 magnitude
- **Exhaustion Flag:** True when in late Wave 5 or Wave C
- **Role:** Identifies position within the market cycle and potential reversal zones

### 4. ATR — Average True Range (Volatility)
- **Calculation:** Wilder's smoothing of True Range (14-period)
- **States:** High (ATR > 1.5× recent median), Normal, Low (ATR < 0.7× recent median)
- **Role:** Measures volatility; high ATR adds caution, low ATR supports trend continuation

### 5. ADX — Average Directional Index (Trend Strength)
- **Calculation:** Wilder's smoothed DX from +DI and -DI (14-period)
- **Trend Strength:** Strong (≥25), Moderate (20-25), Weak (<20)
- **Direction:** Bullish (+DI > -DI), Bearish (-DI > +DI)
- **Role:** Confirms whether trends are worth trading; ADX > 25 indicates a strong trend

## Output Template

```json
{
  "signal": "BUY | SELL | HOLD",
  "last_price": 24250.00,
  "core_thesis": "Bullet-point summary of the primary reasoning",
  "market_nuance": "Additional context, risks, or secondary observations",
  "coral_summary": "Coral Trend direction and price position",
  "hma_summary": "HMA crossover state, fast/slow values, and slope",
  "elliott_summary": "Current wave label, type, confidence, and exhaustion status"
}
```

## Rules for Signal Generation

- **BUY** — Coral Trend is bullish, HMA shows bullish crossover, no Elliott exhaustion warning
- **SELL** — Coral Trend is bearish, HMA shows bearish crossover, or Elliott exhaustion is present with bearish momentum
- **HOLD** — Mixed signals, neutral readings, conflicting indicators

## Data Source
- Yahoo Finance (`^NSEI`) — 15-minute intervals, 1-month lookback period

## Triggers
- Call tools: `get_nifty_ohlcv`, `get_indicator_summary`, `get_nifty_analysis`
- Dashboard auto-refreshes every 15 seconds via web UI
- Indicator engine recalculation every 15 minutes