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

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


def is_market_open() -> bool:
    """Check if current IST time is within NSE market hours (Mon-Fri, 9:15 AM - 3:30 PM)."""
    now = datetime.now(IST)
    # Weekend check (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Time check
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return market_open <= now <= market_close


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
from trade_journal import (
    record_signal, check_pending_trades, get_performance_stats,
    get_recent_trades, get_strategy_suggestion,
)
from sma50_filter import compute_sma50, apply_sma50_filter, SMA50Result
from ai_analyst import run_ai_analysis, get_latest_ai_verdict

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

    # Skip analysis if market is closed
    if not is_market_open():
        logger.info("Market closed — skipping analysis.")
        return

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
        _active_verdict = _build_verdict(_active_summary, df)
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

        # Record BUY/SELL signals in trade journal
        if _active_verdict and _active_summary:
            record_signal(_active_verdict, _active_summary)

        # Check pending trades for 1-hour price movement
        updated = check_pending_trades()
        if updated > 0:
            logger.info("Trade journal: %d pending trades updated.", updated)

        # Run AI analysis in background
        if _active_verdict and _active_summary:
            try:
                ai_result = run_ai_analysis(_active_verdict, _active_summary)
                if ai_result:
                    agrees = "AGREES" if ai_result.get("agrees_with_mechanical") else "DIFFERS"
                    logger.info("AI: %s (confidence=%s, %s)", ai_result.get("ai_signal", "?"), ai_result.get("ai_confidence", "?"), agrees)
            except Exception as ai_exc:
                logger.warning("AI analysis failed: %s", ai_exc)
    except Exception as exc:
        logger.error("Scheduled analysis failed: %s", exc)


def _build_verdict(summary: IndicatorSummary, df: pd.DataFrame = None) -> AnalysisVerdict:
    """Build verdict incorporating Coral, HMA, Elliott, ADX, ATR, and SMA50."""
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
        raw_signal = SignalVerdict.BUY
        thesis = "Bullish alignment across trend, momentum, ADX, and wave structure."
        nuance = (
            f"Strength from Coral Trend, HMA crossover, and {adx.trend_strength} "
            f"{adx.direction} ADX. Volatility is {atr.relative_strength}. "
            "Monitor for volume confirmation."
        )
    elif score <= -3:
        raw_signal = SignalVerdict.SELL
        thesis = "Bearish alignment — trend, momentum, and ADX are all negative."
        nuance = (
            f"Coral and HMA both bearish with {adx.trend_strength} {adx.direction} ADX. "
            f"ATR is {atr.relative_strength}. "
            "Consider protective puts or reduced exposure."
        )
    else:
        raw_signal = SignalVerdict.HOLD
        thesis = "Mixed signals — no clear directional edge."
        nuance = (
            f"Coral and HMA show conflicting or neutral readings. "
            f"ADX is {adx.trend_strength} ({adx.adx_value}), "
            f"volatility is {atr.relative_strength}. "
            "Wait for clearer setup."
        )

    # ── SMA50 Trend Filter ──────────────────────────────────────────
    sma50_filter_result = None
    sma50_summary = ""
    if df is not None and not df.empty:
        sma50_result = compute_sma50(df)
        sma50_filter_result = sma50_result
        filtered_signal = apply_sma50_filter(raw_signal, sma50_result)
        if filtered_signal != raw_signal:
            signal = filtered_signal
            sma50_summary = (
                f"SMA50 filter overrode {raw_signal.value} to HOLD "
                f"(Close {sma50_result.price_position} SMA50 at {sma50_result.value:.2f})."
            )
            thesis = f"{thesis} SMA50 filter applied — Close is {sma50_result.price_position} SMA50."
            nuance = (
                f"Original signal was {raw_signal.value} but price at {summary.last_price:.2f} "
                f"is {sma50_result.price_position} the 50-period MA ({sma50_result.value:.2f}). "
                f"Downgraded to HOLD for trend alignment."
            )
            logger.info("SMA50 filter applied: %s → HOLD (price=%s SMA50)", raw_signal.value, sma50_result.price_position)
        else:
            signal = raw_signal
            sma50_summary = (
                f"SMA50 confirmed: Close is {sma50_result.price_position} SMA50 at {sma50_result.value:.2f}."
            )
    else:
        signal = raw_signal

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
            "market_open": is_market_open(),
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


@app.get("/api/llm-analysis")
async def api_llm_analysis() -> JSONResponse:
    """Return a comprehensive analysis package designed for LLM consumption.

    This endpoint packages ALL indicator data + mechanical verdict + reasoning
    framework into a single JSON response. An LLM can use this to produce its
    own independent verdict by applying the rules in skill_nifty_technical_analyst.md.
    """
    if _active_summary is None or _active_verdict is None:
        return JSONResponse(
            {"status": "pending", "message": "Analysis not yet complete."},
            status_code=503,
        )

    now_ist = datetime.now(IST)
    df = get_cached_data()

    # Compute SMA50
    sma50_val = 0.0
    sma50_pos = "unknown"
    if df is not None and not df.empty:
        sma50_res = compute_sma50(df)
        sma50_val = sma50_res.value
        sma50_pos = sma50_res.price_position

    indicator_dict = format_indicator_summary(_active_summary)
    signal_info = get_current_signal_info()
    strategy = get_strategy_suggestion(_active_verdict, _active_summary)
    perf = get_performance_stats()

    llm_package = {
        "query": "Analyze the following NIFTY 50 indicator data and produce your own BUY/SELL/HOLD verdict. Use the reasoning framework from skill_nifty_technical_analyst.md.",
        "timestamp": now_ist.isoformat(),
        "market_open": is_market_open(),
        "mechanical_verdict": {
            "signal": _active_verdict.signal.value,
            "stable_signal": signal_info["current_signal"],
            "core_thesis": _active_verdict.core_thesis,
            "market_nuance": _active_verdict.market_nuance,
        },
        "current_price": _active_verdict.last_price,
        "indicators": {
            "coral_trend": indicator_dict["coral"],
            "hull_moving_average": indicator_dict["hma"],
            "elliott_wave": indicator_dict["elliott"],
            "average_true_range": indicator_dict["atr"],
            "average_directional_index": indicator_dict["adx"],
        },
        "sma50": {
            "value": sma50_val,
            "price_position": sma50_pos,
        },
        "indicator_summaries": {
            "coral": _active_verdict.coral_summary,
            "hma": _active_verdict.hma_summary,
            "elliott": _active_verdict.elliott_summary,
            "atr": _active_verdict.atr_summary,
            "adx": _active_verdict.adx_summary,
        },
        "options_strategy": {
            "suggested_strategy": strategy.get("suggested_strategy", ""),
            "rationale": strategy.get("rationale", ""),
            "direction": strategy.get("direction", ""),
        },
        "signal_stabilization": signal_info,
        "historical_performance": {
            "win_rate_pct": perf.get("win_rate_pct", 0),
            "total_signals": perf.get("completed_checks", 0),
            "avg_move_points": perf.get("avg_move_points", 0),
        },
        "reasoning_framework": {
            "reference": "skill_nifty_technical_analyst.md",
            "key_rules": [
                "ADX < 20 = ranging market, avoid trend signals",
                "ADX >= 25 = strong trend, increase conviction",
                "Elliott exhaustion (Wave 5/C) trumps alignment",
                "Coral crossing = no-trade zone, wait for confirmation",
                "HMA slope: steep = committed, flat = hesitant",
                "SMA50: BUY only above, SELL only below",
                "High ATR = reduce conviction, widen stops",
                "Low ATR + strong ADX = high conviction trend continuation",
            ],
        },
        "your_task": "Apply the reasoning framework above to the indicators provided. Consider confluences and conflicts between indicators. Produce STRONG_BUY/BUY/CAUTIOUS_BUY/HOLD/CAUTIOUS_SELL/SELL/STRONG_SELL with detailed reasoning.",
    }

    return JSONResponse(llm_package)


@app.get("/api/ai-verdict")
async def api_ai_verdict() -> JSONResponse:
    """Return the latest AI-generated verdict for dashboard display."""
    verdict = get_latest_ai_verdict()
    if verdict is None:
        return JSONResponse({"ai_available": False, "ai_signal": "NONE", "reasoning": "AI analysis not yet run."})
    return JSONResponse({
        "ai_available": True,
        "ai_signal": verdict.get("ai_signal", "N/A"),
        "ai_confidence": verdict.get("ai_confidence", "LOW"),
        "mechanical_signal": verdict.get("mechanical_signal", "N/A"),
        "agrees": verdict.get("agrees_with_mechanical", True),
        "reasoning": verdict.get("reasoning", ""),
        "key_conflict": verdict.get("key_conflict", ""),
        "suggested_strategy": verdict.get("suggested_strategy", ""),
        "model_used": verdict.get("model_used", ""),
        "elapsed_ms": verdict.get("elapsed_ms", 0),
        "timestamp": verdict.get("timestamp", ""),
    })


@app.get("/api/performance")
async def api_performance() -> JSONResponse:
    """Return trade journal performance stats and strategy suggestion."""
    stats = get_performance_stats()
    if _active_verdict and _active_summary:
        strategy = get_strategy_suggestion(_active_verdict, _active_summary)
    else:
        strategy = {}
    return JSONResponse({
        "performance": stats,
        "strategy": strategy,
    })


@app.get("/api/trades")
async def api_trades() -> JSONResponse:
    """Return recent trade records."""
    return JSONResponse({"trades": get_recent_trades(limit=30)})


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
