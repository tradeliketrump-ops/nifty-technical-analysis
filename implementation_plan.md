# Implementation Plan

Build a NIFTY 50 Discretionary Technical Analysis system with a Python MCP server that calculates Coral Trend, Hull Moving Average (HMA), and Elliott Wave (1-5/A-B-C) indicators, exposes them as MCP tools for LLM consumption, and serves a scheduled 15-minute refresh web dashboard via FastAPI.

The system lives under `C:\Users\mikhu\Desktop\NIFTY_AI\` and is entirely self-contained in Python, reusing the `yfinance` data pipeline already established in the existing TRADE-DS project. The MCP server communicates via stdio transport (the standard MCP pattern) and provides three tools: `get_nifty_ohlcv()` for raw market data, `get_indicator_summary()` for the three pre-calculated indicator states, and `get_nifty_analysis()` which returns the full analysis verdict ready for the LLM skill template. A FastAPI web server runs alongside, scheduling recalculation every 15 minutes via APScheduler and serving a live dashboard at `http://localhost:8000`.

[Types]

Define the core data structures that flow through the system — indicator outputs, wave labels, and the final analysis verdict.

```python
# ─── Enums & Literals ────────────────────────────────────────────────
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

# ─── Data Containers ─────────────────────────────────────────────────
@dataclass
class CoralTrendResult:
    value: float                         # Current Coral Trend line value
    price_position: str                  # "above", "below", "crossing"
    direction: TrendDirection

@dataclass
class HMAResult:
    fast_value: float                    # Fast HMA (e.g. 9-period)
    slow_value: float                    # Slow HMA (e.g. 18-period)
    crossover: MomentumState
    slope: str                           # "steep_up", "flat", "steep_down"

@dataclass
class ElliottWaveResult:
    wave_label: WaveLabel
    wave_type: WaveType
    confirmation_strength: float         # 0.0 to 1.0 confidence
    count_exhaustion: bool               # True if in late Wave 5 / Wave C

@dataclass
class IndicatorSummary:
    timestamp: datetime
    last_price: float
    coral: CoralTrendResult
    hma: HMAResult
    elliott: ElliottWaveResult

@dataclass
class AnalysisVerdict:
    signal: SignalVerdict
    core_thesis: str
    market_nuance: str
    last_price: float
    coral_summary: str
    hma_summary: str
    elliott_summary: str
```

[Files]

Create 7 new files under `C:\Users\mikhu\Desktop\NIFTY_AI\`; no existing files need modification since this is a greenfield project.

| # | File Path | Purpose |
|---|-----------|---------|
| 1 | `C:\Users\mikhu\Desktop\NIFTY_AI\requirements.txt` | Python dependencies: yfinance, pandas, numpy, fastapi, uvicorn, apscheduler, mcp |
| 2 | `C:\Users\mikhu\Desktop\NIFTY_AI\indicators.py` | All three indicator calculators (Coral Trend, HMA, Elliott Wave) as pure functions + types module |
| 3 | `C:\Users\mikhu\Desktop\NIFTY_AI\data_provider.py` | yfinance wrapper: fetches ^NSEI OHLCV, caches latest data, returns DataFrames |
| 4 | `C:\Users\mikhu\Desktop\NIFTY_AI\nifty_mcp_server.py` | Python MCP server using `mcp` library: registers 3 tools, loads indicator engine, runs on stdio |
| 5 | `C:\Users\mikhu\Desktop\NIFTY_AI\web_server.py` | FastAPI application: health endpoint, /api/analysis endpoint, schedule 15-min recalc |
| 6 | `C:\Users\mikhu\Desktop\NIFTY_AI\templates\dashboard.html` | Single-page dashboard: displays last price, Coral/HMA/Elliott state, verdict, commentary |
| 7 | `C:\Users\mikhu\Desktop\NIFTY_AI\skill_nifty_technical_analyst.md` | The LLM skill definition (the user-provided skill document, saved as Markdown) |

No files need to be deleted or moved.

[Functions]

Implement 14 functions across 4 modules. All functions are pure (no side effects) except data-fetching and server lifecycle.

### indicators.py (7 functions)

| Function | Signature | File | Purpose |
|----------|-----------|------|---------|
| `compute_coral_trend(df: pd.DataFrame) -> CoralTrendResult` | `indicators.py` | Calculate Coral Trend line using smoothed ATR-based channel. Returns direction, position relative to price. |
| `compute_hma(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 18) -> HMAResult` | `indicators.py` | Compute two Hull Moving Averages and detect crossover state and slope steepness. |
| `detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> list[dict]` | `indicators.py` | Find swing highs/lows using local peak/trough detection over a rolling window. Returns list of swing points with price, index, type. |
| `classify_waves(swings: list[dict]) -> tuple[list[WaveLabel], ElliottWaveResult]` | `indicators.py` | Apply Elliott Wave rules to labeled swing points: detect 5-wave impulses (3rd extended), 3-wave corrections, retracement ratios. |
| `compute_elliott_wave(df: pd.DataFrame) -> ElliottWaveResult` | `indicators.py` | Orchestrator: calls detect_swing_points then classify_waves, returns current wave label, type, and exhaustion flag. |
| `compute_all_indicators(df: pd.DataFrame) -> IndicatorSummary` | `indicators.py` | Top-level orchestrator: runs all three indicators and assembles an IndicatorSummary dataclass. |
| `format_indicator_summary(summary: IndicatorSummary) -> dict` | `indicators.py` | Serialize the IndicatorSummary into a plain dict for JSON/MCP output. |

### data_provider.py (3 functions)

| Function | Signature | File | Purpose |
|----------|-----------|------|---------|
| `fetch_nifty_ohlcv(period: str = "1mo", interval: str = "15m") -> pd.DataFrame` | `data_provider.py` | Call yfinance on `^NSEI`, return clean OHLCV DataFrame. Period defaults to 1 month, interval to 15 minutes. |
| `get_cached_data(force_refresh: bool = False) -> pd.DataFrame` | `data_provider.py` | Module-level cache that avoids refetching if data is < 60s old. Used by both MCP and web servers. |
| `get_latest_price(df: pd.DataFrame) -> float` | `data_provider.py` | Extract the most recent closing price from the DataFrame. |

### nifty_mcp_server.py (4 functions — 1 lifecycle + 3 tool handlers)

| Function | Signature | File | Purpose |
|----------|-----------|------|---------|
| `main()` | `nifty_mcp_server.py` | Server entry point: initializes MCP server, registers handlers, connects stdio transport. |
| `handle_get_nifty_ohlcv(args) -> list[TextContent]` | `nifty_mcp_server.py` | Tool handler: returns OHLCV as JSON text. |
| `handle_get_indicator_summary(args) -> list[TextContent]` | `nifty_mcp_server.py` | Tool handler: returns serialized IndicatorSummary. |
| `handle_get_nifty_analysis(args) -> list[TextContent]` | `nifty_mcp_server.py` | Tool handler: returns full analysis verdict formatted per skill template. |

### web_server.py (3 functions)

| Function | Signature | File | Purpose |
|----------|-----------|------|---------|
| `scheduled_analysis()` | `web_server.py` | APScheduler callback: refreshes cached data and recomputes indicators. Runs every 15 min. |
| `serve_dashboard() -> HTMLResponse` | `web_server.py` | FastAPI route: renders dashboard.html with current analysis state. |
| `api_analysis() -> JSONResponse` | `web_server.py` | FastAPI route: returns analysis as JSON for AJAX polling by dashboard. |

[Classes]

No new classes are strictly necessary. The existing `dataclass` types defined in the [Types] section handle all data modeling. If the MCP library requires a class-based server, the `main()` function in `nifty_mcp_server.py` can instantiate the SDK's `Server` class with appropriate capabilities. No custom classes beyond the MCP SDK's built-in `Server`.

[Dependencies]

Add 1 new dependency to the project; the rest are already familiar from the existing TRADE-DS codebase.

```
mcp>=1.0.0           # MCP SDK for Python server (installed via pip install mcp)
yfinance==1.4.1      # Already used in TRADE-DS
pandas>=2.0.0        # Already used
numpy>=1.24.0        # Already used
fastapi>=0.104.0     # Already used
uvicorn>=0.24.0      # Already used
apscheduler>=3.10.0  # Already used
```

Installation: `pip install mcp yfinance pandas numpy fastapi uvicorn apscheduler`

[Testing]

No formal test suite is planned for this phase. Validation will be manual:
1. Run `python nifty_mcp_server.py` — verify the MCP server starts and connects to stdio without errors.
2. Run `python web_server.py` — verify FastAPI starts on port 8000, dashboard loads.
3. Call `get_nifty_ohlcv()` via MCP — confirm OHLCV JSON is returned with recent data.
4. Call `get_indicator_summary()` — confirm Coral, HMA, Elliott fields are populated.
5. Call `get_nifty_analysis()` — confirm full verdict with commentary is returned.
6. Let the scheduler run once — confirm dashboard auto-refreshes.

[Implementation Order]

Implement in dependency order, bottom-up: data layer → indicator engine → MCP server → web server → dashboard → MCP configuration.

1. **Create project directory + requirements.txt** — scaffold `C:\Users\mikhu\Desktop\NIFTY_AI\`, install deps.
2. **Implement `data_provider.py`** — yfinance wrapper with caching. Test fetch with `python -c`.
3. **Implement `indicators.py`** — all 7 functions. Start with Coral Trend (simplest), then HMA, then Elliott Wave (most complex).
4. **Implement `nifty_mcp_server.py`** — MCP server with 3 tools. Test by running server and checking tool listing.
5. **Implement `web_server.py`** — FastAPI + APScheduler. Start server, hit `/api/analysis`.
6. **Create `templates/dashboard.html`** — HTML/CSS/JS dashboard with auto-polling.
7. **Write `skill_nifty_technical_analyst.md`** — save the user-provided skill document for reference.
8. **Configure MCP settings** — add the NIFTY MCP server to `cline_mcp_settings.json`.
9. **Final integration test** — run MCP server + web server, verify end-to-end.