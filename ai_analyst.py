"""
AI Analyst — calls OpenAI API to produce an independent trading verdict.

Runs every 15 minutes alongside the mechanical analysis.
The AI verdict is stored and displayed alongside the mechanical signal
on the dashboard for comparison.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

from indicators import AnalysisVerdict, IndicatorSummary, SignalVerdict
from sma50_filter import compute_sma50
from trade_journal import get_performance_stats, get_strategy_suggestion
from data_provider import get_cached_data

logger = logging.getLogger("nifty_ai")

# ─── Constants ─────────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))
AI_DATA_FILE = Path(__file__).parent / "ai_verdict.json"

OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Fast & cheap (~$0.15/1M input tokens)
AI_SYSTEM_PROMPT = """You are a senior NIFTY 50 technical analyst.

Your job is to analyze the provided indicator data and produce an independent
BUY/SELL/HOLD verdict. You must think step by step using the framework below.

REASONING FRAMEWORK:
1. MACRO (Coral + ADX): Are they aligned? Is ADX >= 25 (strong trend)?
2. MOMENTUM (HMA): Is HMA confirming or diverging? Slope steep or flat?
3. CYCLE (Elliott): Are we in impulse or correction? Exhaustion flagged?
4. VOLATILITY (ATR): High/normal/low — adjust conviction accordingly.
5. TREND FILTER (SMA50): BUY only above SMA50, SELL only below SMA50.
6. SYNTHESIS: Combine all factors into a final verdict.

RULES:
- ADX < 20 = ranging market. Avoid trend signals. Prefer HOLD.
- Elliott exhaustion (Wave 5 or Wave C) overrides bullish alignment.
- HOLD is the default if signals are conflicting or unclear.
- Consider the options strategy suggestion in your reasoning.

OUTPUT FORMAT (ONLY valid JSON, no other text):
{
  "ai_signal": "STRONG_BUY | BUY | CAUTIOUS_BUY | HOLD | CAUTIOUS_SELL | SELL | STRONG_SELL",
  "ai_confidence": "HIGH | MODERATE | LOW",
  "reasoning": "Step-by-step reasoning covering all 6 framework steps",
  "key_conflict": "Main conflicting indicators or reason for caution",
  "agrees_with_mechanical": true|false,
  "suggested_strategy": "Bull Put Credit Spread | Bear Call Credit Spread | Iron Condor | Wait"
}"""


# ─── Data structures ───────────────────────────────────────────────────


@dataclass
class AIVerdict:
    timestamp: str
    mechanical_signal: str
    ai_signal: str
    ai_confidence: str
    reasoning: str
    key_conflict: str
    agrees_with_mechanical: bool
    suggested_strategy: str
    model_used: str
    elapsed_ms: int


# ─── Persistence ───────────────────────────────────────────────────────


def _load_ai_verdict() -> Optional[dict]:
    """Load the latest AI verdict from JSON file."""
    if not AI_DATA_FILE.exists():
        return None
    try:
        return json.loads(AI_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_ai_verdict(verdict: dict) -> None:
    """Save AI verdict to JSON file."""
    try:
        AI_DATA_FILE.write_text(
            json.dumps(verdict, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not save AI verdict: %s", exc)


# ─── OpenAI API call ──────────────────────────────────────────────────


def _call_openai(analysis_data: dict) -> Optional[dict]:
    """
    Call OpenAI API to get an AI-generated verdict.

    Parameters
    ----------
    analysis_data : dict
        The full analysis package from /api/llm-analysis.

    Returns
    -------
    dict or None
        Parsed AI verdict JSON, or None on failure.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — skipping AI analysis.")
        return None

    # Build a concise prompt from the analysis data
    indicators = analysis_data.get("indicators", {})
    summaries = analysis_data.get("indicator_summaries", {})
    mechanical = analysis_data.get("mechanical_verdict", {})
    perf = analysis_data.get("historical_performance", {})
    strategy = analysis_data.get("options_strategy", {})

    user_prompt = f"""Current NIFTY 50 Analysis:

Price: {analysis_data.get('current_price', 'N/A')}
Market: {'OPEN' if analysis_data.get('market_open') else 'CLOSED'}

INDICATORS:
- Coral Trend: {summaries.get('coral', 'N/A')}
- HMA: {summaries.get('hma', 'N/A')}
- Elliott Wave: {summaries.get('elliott', 'N/A')}
- ATR: {summaries.get('atr', 'N/A')}
- ADX: {summaries.get('adx', 'N/A')}
- SMA50: Price is {analysis_data.get('sma50', {}).get('price_position', 'unknown')} SMA50 at {analysis_data.get('sma50', {}).get('value', 0)}

Coral details: {json.dumps(indicators.get('coral_trend', {}))}
HMA details: {json.dumps(indicators.get('hull_moving_average', {}))}
ADX details: {json.dumps(indicators.get('average_directional_index', {}))}
Elliott details: {json.dumps(indicators.get('elliott_wave', {}))}

MECHANICAL VERDICT: {mechanical.get('signal', 'N/A')} - {mechanical.get('core_thesis', '')}

OPTIONS STRATEGY SUGGESTION: {strategy.get('suggested_strategy', 'N/A')}
HISTORICAL WIN RATE: {perf.get('win_rate_pct', 0)}% ({perf.get('total_signals', 0)} signals)

Your task: Apply the reasoning framework and produce your independent verdict."""

    start = time.time()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": AI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 800,
            },
            timeout=30,
        )

        elapsed = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            logger.warning("OpenAI API error: %s %s", resp.status_code, resp.text)
            return None

        result = resp.json()
        content = result["choices"][0]["message"]["content"]

        # Extract JSON from the response (handle markdown-wrapped JSON)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in AI response: %s", content[:200])
            return None

        parsed = json.loads(json_match.group())

        return {
            "timestamp": datetime.now(IST).isoformat(),
            "mechanical_signal": mechanical.get("signal", "N/A"),
            "ai_signal": parsed.get("ai_signal", "HOLD"),
            "ai_confidence": parsed.get("ai_confidence", "LOW"),
            "reasoning": parsed.get("reasoning", ""),
            "key_conflict": parsed.get("key_conflict", ""),
            "agrees_with_mechanical": parsed.get("agrees_with_mechanical", True),
            "suggested_strategy": parsed.get("suggested_strategy", strategy.get("suggested_strategy", "")),
            "model_used": OPENAI_MODEL,
            "elapsed_ms": elapsed,
        }

    except requests.RequestException as exc:
        logger.warning("OpenAI request failed: %s", exc)
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse AI response: %s", exc)
        return None


# ─── Main entry point ─────────────────────────────────────────────────


def run_ai_analysis(
    mechanical_verdict: AnalysisVerdict,
    indicator_summary: Optional[IndicatorSummary] = None,
) -> dict:
    """
    Run AI analysis and return the verdict.

    Parameters
    ----------
    mechanical_verdict : AnalysisVerdict
        The current mechanical verdict.
    indicator_summary : IndicatorSummary or None
        The current indicator summary.

    Returns
    -------
    dict
        AI verdict dict, or fallback dict if AI is unavailable.
    """
    # Build the analysis package (same as /api/llm-analysis)
    analysis_data = _build_analysis_package(mechanical_verdict, indicator_summary)

    # Call OpenAI
    ai_result = _call_openai(analysis_data)

    if ai_result:
        _save_ai_verdict(ai_result)
        logger.info(
            "AI verdict: %s (confidence=%s, agrees=%s, %dms)",
            ai_result["ai_signal"],
            ai_result["ai_confidence"],
            ai_result["agrees_with_mechanical"],
            ai_result["elapsed_ms"],
        )
        return ai_result

    # Fallback: return last cached verdict or default
    cached = _load_ai_verdict()
    if cached:
        logger.info("Using cached AI verdict from %s", cached.get("timestamp", "?"))
        return cached

    fallback = {
        "timestamp": datetime.now(IST).isoformat(),
        "mechanical_signal": mechanical_verdict.signal.value,
        "ai_signal": "UNAVAILABLE",
        "ai_confidence": "NONE",
        "reasoning": "AI analysis unavailable. Check OPENAI_API_KEY environment variable.",
        "key_conflict": "",
        "agrees_with_mechanical": True,
        "suggested_strategy": "",
        "model_used": "none",
        "elapsed_ms": 0,
    }
    _save_ai_verdict(fallback)
    return fallback


def get_latest_ai_verdict() -> Optional[dict]:
    """Get the latest AI verdict from cache."""
    return _load_ai_verdict()


def _build_analysis_package(
    verdict: AnalysisVerdict,
    summary: Optional[IndicatorSummary] = None,
) -> dict:
    """Build the analysis package for AI consumption."""
    now_ist = datetime.now(IST)
    df = get_cached_data()

    sma50_val = 0.0
    sma50_pos = "unknown"
    if df is not None and not df.empty:
        sma50_res = compute_sma50(df)
        sma50_val = sma50_res.value
        sma50_pos = sma50_res.price_position

    indicator_dict = {}
    signal_info = {}
    strategy = {}
    perf = {}

    try:
        from indicators import format_indicator_summary
        if summary:
            indicator_dict = format_indicator_summary(summary)
    except Exception:
        pass

    try:
        from signal_history import get_current_signal_info
        signal_info = get_current_signal_info()
    except Exception:
        pass

    try:
        strategy = get_strategy_suggestion(verdict, summary)
        perf = get_performance_stats()
    except Exception:
        pass

    return {
        "timestamp": now_ist.isoformat(),
        "market_open": datetime.now(IST).weekday() < 5 and (
            datetime.now(IST).hour >= 9 and datetime.now(IST).minute >= 15
        ) and datetime.now(IST).hour < 15 or (datetime.now(IST).hour == 15 and datetime.now(IST).minute <= 30),
        "current_price": verdict.last_price,
        "mechanical_verdict": {
            "signal": verdict.signal.value,
            "core_thesis": verdict.core_thesis,
            "market_nuance": verdict.market_nuance,
        },
        "indicators": {
            "coral_trend": indicator_dict.get("coral", {}),
            "hull_moving_average": indicator_dict.get("hma", {}),
            "elliott_wave": indicator_dict.get("elliott", {}),
            "average_true_range": indicator_dict.get("atr", {}),
            "average_directional_index": indicator_dict.get("adx", {}),
        },
        "sma50": {"value": sma50_val, "price_position": sma50_pos},
        "indicator_summaries": {
            "coral": verdict.coral_summary,
            "hma": verdict.hma_summary,
            "elliott": verdict.elliott_summary,
            "atr": verdict.atr_summary,
            "adx": verdict.adx_summary,
        },
        "options_strategy": strategy,
        "historical_performance": perf,
    }