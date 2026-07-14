"""
Signal history tracker and stabilizer.

Stores signal history in memory and provides stabilization
to prevent flip-flopping from noisy data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from indicators import AnalysisVerdict, SignalVerdict

logger = logging.getLogger("nifty_signal")

# ─── History entry ─────────────────────────────────────────────────────

_STABILIZATION_WINDOW = 2  # Number of consecutive same-signal readings needed


@dataclass
class SignalEntry:
    timestamp: str
    signal: str
    previous_signal: str
    is_change: bool
    last_price: float
    core_thesis: str
    market_nuance: str
    coral_summary: str
    hma_summary: str
    elliott_summary: str
    atr_summary: str
    adx_summary: str


# ─── Signal tracker ────────────────────────────────────────────────────

_signal_history: list[SignalEntry] = []
_candidate_signal: Optional[str] = None  # Signal waiting for confirmation
_consecutive_count: int = 0
_current_stable_signal: str = "HOLD"
_last_raw_signal: str = "HOLD"
_previous_stable_signal: str = "HOLD"


def get_stable_signal(raw_verdict: AnalysisVerdict) -> tuple[bool, str, str]:
    """
    Apply signal stabilization.

    A signal change is only accepted after being seen for
    ``_STABILIZATION_WINDOW`` consecutive cycles.

    Parameters
    ----------
    raw_verdict : AnalysisVerdict
        The current analysis verdict from the scoring engine.

    Returns
    -------
    tuple[bool, str, str]
        (changed, new_signal, previous_signal)
    """
    global _candidate_signal, _consecutive_count, _current_stable_signal
    global _previous_stable_signal, _last_raw_signal

    raw_signal = raw_verdict.signal.value
    _last_raw_signal = raw_signal

    if raw_signal == _current_stable_signal:
        # Still in the same signal — reset candidate
        _candidate_signal = None
        _consecutive_count = 0
        return (False, _current_stable_signal, _previous_stable_signal)

    # Raw signal differs from stable signal
    if _candidate_signal == raw_signal:
        _consecutive_count += 1
    else:
        _candidate_signal = raw_signal
        _consecutive_count = 1

    if _consecutive_count >= _STABILIZATION_WINDOW:
        # Confirmed change
        _previous_stable_signal = _current_stable_signal
        _current_stable_signal = raw_signal
        _candidate_signal = None
        _consecutive_count = 0
        return (True, _current_stable_signal, _previous_stable_signal)

    return (False, _current_stable_signal, _previous_stable_signal)


def add_history_entry(verdict: AnalysisVerdict, changed: bool, stable_signal: str, prev_signal: str) -> SignalEntry:
    """Add a new entry to the signal history."""
    entry = SignalEntry(
        timestamp=datetime.now().isoformat(),
        signal=stable_signal,
        previous_signal=prev_signal,
        is_change=changed,
        last_price=verdict.last_price,
        core_thesis=verdict.core_thesis,
        market_nuance=verdict.market_nuance,
        coral_summary=verdict.coral_summary,
        hma_summary=verdict.hma_summary,
        elliott_summary=verdict.elliott_summary,
        atr_summary=verdict.atr_summary,
        adx_summary=verdict.adx_summary,
    )
    _signal_history.append(entry)
    # Keep last 100 entries
    if len(_signal_history) > 100:
        _signal_history.pop(0)
    return entry


def get_history(limit: int = 50) -> list[dict]:
    """Return signal history as a list of dicts (newest first)."""
    entries = list(reversed(_signal_history[-limit:]))
    return [asdict(e) for e in entries]


def get_current_signal_info() -> dict:
    """Return current signal state info for dashboard display."""
    return {
        "current_signal": _current_stable_signal,
        "previous_signal": _previous_stable_signal,
        "raw_signal": _last_raw_signal,
        "candidate_signal": _candidate_signal,
        "stabilization_count": _consecutive_count,
        "stabilization_needed": _STABILIZATION_WINDOW,
        "history_count": len(_signal_history),
    }


def get_status_json(verdict: AnalysisVerdict) -> dict:
    """
    Get the full status including signal info and history for the API.
    """
    return {
        "signal_info": get_current_signal_info(),
        "history": get_history(limit=20),
    }