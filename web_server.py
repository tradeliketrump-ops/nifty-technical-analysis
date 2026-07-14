"""
NIFTY 50 Web Server — FastAPI + APScheduler

Serves a live dashboard at http://localhost:8000 with 15-minute
auto-refresh of indicators via APScheduler.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

# IST timezone (UTC +5:30)
IST = timezone(timedelta(hours=5, minutes=30))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler

from data_provider import get_cached_data, clear_cache
from indicators import (
    AnalysisVerdict,
    ATRResult,
    ADXResult,
    IndicatorSummary,
    compute_all_indicators,
    format_indicator_summary,
)
from alerter import check_and_alert, is_configured
from signal_history import get_stable_signal, add_history_entry, get_history, get_current_signal_info

# ─── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nifty_web")

# ─── Paths ─────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent / "templates"
DASHBOARD_PATH = TEMPLATES_DIR / "dashboard.html"

# ─── Global state ──────────────────────────────────────────────────────

_active_summary: IndicatorSummary | None = None
_active_verdict: AnalysisVerdict | None = None
_scheduler: BackgroundScheduler | None = None


# ─── Scheduler ─────────────────────────────────────────────────────────


def scheduled_analysis() -> None:
    """APScheduler callback: refresh data and recompute indicators."""
    global _active_summary, _active_verdict
    try:
        logger.info("Running scheduled analysis...")
        clear_cache()
        df = get_cached_data(force_refresh=True)

        if df is None or df.empty:
            logger.warning("No data returned — keeping previous analysis state.")
            # Still allow health check to pass; don't overwrite state
            if _active_summary is None:
                # First run with no data — API will return 503 gracefully
                logger.warning("No cached data available yet.")
            return

        _active_summary = compute_all_indicators(df)
        _active_verdict = _build_verdict(_active_summary)
        logger.info(
            "Analysis complete — price=%.2f, raw_signal=%s",
            _active_summary.last_price,
            _active_verdict.signal.value,
        )

        # Apply signal stabilization (prevents flip-flopping)
        changed, stable_signal, prev_signal = get_stable_signal(_active_verdict)

        # Log the stable signal info
        s_info = get_current_signal_info()
        logger.info(
            "Stable signal: %s (raw=%s, pending=%s, count=%d/%d)",
            s_info["current_signal"],
            s_info["raw_signal"],
            s_info["candidate_signal"] or "-",
            s_info["stabilization_count"],
            s_info["stabilization_needed"],
        )

        # Record in history
        add_history_entry(_active_verdict, changed, stable_signal, prev_signal)

        # Send alert only if signal was confirmed changed
        if changed and _active_verdict:
            logger.info("Signal CHANGE confirmed: %s → %s", prev_signal, stable_signal)
            check_and_alert(_active_verdict)
    except Exception as exc:
        logger.error("Scheduled analysis failed: %s", exc)


def _build_verdict(summary: IndicatorSummary) -> AnalysisVerdict:
    """Build verdict incorporating Coral, HMA, Elliott, ADX, and ATR."""
    from indicators import SignalVerdict, TrendDirection

    coral = summary.coral
    hma = summary.hma
    elliott = summary.elliott
    atr = summary.atr
    adx = summary.adx

    score = 0
    if coral.direction == TrendDirection.BULLISH:
        score += 2
    elif coral.direction == TrendDirection.BEARISH:
        score -= 2

    if hma.crossover.value == "bullish_cross":
        score += 2
    elif hma.crossover.value == "bearish_cross":
        score -= 2

    if hma.slope == "steep_up":
        score += 1
    elif hma.slope == "steep_down":
        score -= 1

    if elliott.count_exhaustion:
        score -= 1

    if coral.price_position == "above":
        score += 1
    elif coral.price_position == "below":
        score -= 1

    # ADX contribution
    if adx.trend_strength == "strong":
        score += 1 if adx.direction == "bullish" else -1
    elif adx.trend_strength == "weak":
        score -= 1

    # ATR contribution
    if atr.relative_strength == "high":
        score -= 1
    elif atr.relative_strength == "low":
        score += 1

    if score >= 3:
        signal = SignalVerdict.BUY
        thesis = "Bullish alignment across trend, momentum, ADX, and wave structure."
        nuance = (
            f"Strength from Coral Trend, HMA crossover, and {adx.trend_strength} "
            f"{adx.direction} ADX. Volatility is {atr.relative_strength}. "
            "Monitor for volume confirmation."
        )
    elif score <= -3:
        signal = SignalVerdict.SELL
        thesis = "Bearish alignment — trend, momentum, and ADX are all negative."
        nuance = (
            f"Coral and HMA both bearish with {adx.trend_strength} {adx.direction} ADX. "
            f"ATR is {atr.relative_strength}. "
            "Consider protective puts or reduced exposure."
        )
    else:
        signal = SignalVerdict.HOLD
        thesis = "Mixed signals — no clear directional edge."
        nuance = (
            f"Coral and HMA show conflicting or neutral readings. "
            f"ADX is {adx.trend_strength} ({adx.adx_value}), "
            f"volatility is {atr.relative_strength}. "
            "Wait for clearer setup."
        )

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
        core_thesis=thesis,
        market_nuance=nuance,
        last_price=summary.last_price,
        coral_summary=coral_summary,
        hma_summary=hma_summary,
        elliott_summary=elliott_summary,
        atr_summary=atr_summary,
        adx_summary=adx_summary,
    )


# ─── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, shut down on exit."""
    global _scheduler
    logger.info("Starting NIFTY Web Server...")

    # Run initial analysis immediately
    scheduled_analysis()

    # Schedule every 15 minutes
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        scheduled_analysis,
        "interval",
        minutes=15,
        id="nifty_analysis",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("APScheduler started — analysis runs every 15 minutes.")

    yield

    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler shut down.")


# ─── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="NIFTY 50 Technical Analysis",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Routes ────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> str:
    """Serve the dashboard HTML."""
    if not DASHBOARD_PATH.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return DASHBOARD_PATH.read_text(encoding="utf-8")


@app.get("/api/analysis")
async def api_analysis() -> JSONResponse:
    """Return the current analysis as JSON for AJAX polling."""
    if _active_summary is None or _active_verdict is None:
        return JSONResponse(
            {"status": "pending", "message": "Analysis not yet complete. Retry in a moment."},
            status_code=503,
        )

    indicator_dict = format_indicator_summary(_active_summary)
    signal_info = get_current_signal_info()
    return JSONResponse(
        {
            "status": "ok",
            "timestamp": datetime.now(IST).isoformat(),
            "last_price": _active_verdict.last_price,
            "signal": _active_verdict.signal.value,
            "stable_signal": signal_info["current_signal"],
            "previous_stable_signal": signal_info["previous_signal"],
            "raw_signal": signal_info["raw_signal"],
            "core_thesis": _active_verdict.core_thesis,
            "market_nuance": _active_verdict.market_nuance,
            "coral": indicator_dict["coral"],
            "hma": indicator_dict["hma"],
            "elliott": indicator_dict["elliott"],
            "atr": indicator_dict["atr"],
            "adx": indicator_dict["adx"],
            "coral_summary": _active_verdict.coral_summary,
            "hma_summary": _active_verdict.hma_summary,
            "elliott_summary": _active_verdict.elliott_summary,
            "atr_summary": _active_verdict.atr_summary,
            "adx_summary": _active_verdict.adx_summary,
            "signal_info": signal_info,
        }
    )


@app.get("/api/history")
async def api_history() -> JSONResponse:
    """Return signal change history."""
    return JSONResponse({"history": get_history(limit=50)})


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "nifty-technical-analysis",
            "version": "1.0.0",
        }
    )


# ─── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
