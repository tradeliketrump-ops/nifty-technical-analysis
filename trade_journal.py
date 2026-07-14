"""
Trade Journal — tracks signal performance.

Records every signal, checks 1-hour price movement against
a 150-point threshold, and computes win rates.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from data_provider import get_cached_data
from indicators import AnalysisVerdict, IndicatorSummary

logger = logging.getLogger("nifty_journal")

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# ─── Constants ─────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
TRADE_DATA_FILE = DATA_DIR / "trade_data.json"
WIN_THRESHOLD_POINTS = 150  # Minimum price move to count as a win
FOLLOWUP_HOURS = 1          # Check price movement after this many hours

# ─── Data structures ───────────────────────────────────────────────────


@dataclass
class TradeRecord:
    id: str
    timestamp: str
    signal: str
    entry_price: float
    coral_direction: str
    hma_crossover: str
    elliott_label: str
    elliott_exhaustion: bool
    atr_value: float
    adx_value: float
    adx_trend_strength: str
    adx_direction: str
    checked_1h: bool = False
    price_1h: Optional[float] = None
    move_1h: Optional[float] = None  # positive = up, negative = down
    is_win: Optional[bool] = None
    pnl_label: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Persistence ───────────────────────────────────────────────────────


def _load_trades() -> list[dict]:
    """Load trade records from JSON file."""
    if not TRADE_DATA_FILE.exists():
        return []
    try:
        data = json.loads(TRADE_DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load trade data: %s", exc)
        return []


def _save_trades(trades: list[dict]) -> None:
    """Save trade records to JSON file."""
    try:
        # Keep only last 500 entries to prevent file bloat
        if len(trades) > 500:
            trades = trades[-500:]
        TRADE_DATA_FILE.write_text(
            json.dumps(trades, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not save trade data: %s", exc)


# ─── Core Functions ────────────────────────────────────────────────────


def _generate_id() -> str:
    """Generate a unique trade ID."""
    return datetime.now(IST).strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1_000_000) % 1_000_000)


def record_signal(verdict: AnalysisVerdict, summary: Optional[IndicatorSummary] = None) -> None:
    """
    Record a new signal in the trade journal.

    Parameters
    ----------
    verdict : AnalysisVerdict
        The current analysis verdict.
    summary : IndicatorSummary or None
        The raw indicator summary for detailed indicator tracking.
    """
    trades = _load_trades()

    # Don't record HOLD signals (no trade action)
    if verdict.signal.value == "HOLD":
        logger.debug("HOLD signal — skipping trade record.")
        _save_trades(trades)
        return

    # Get indicator details from summary if available
    coral_dir = ""
    hma_cross = ""
    ell_label = ""
    ell_exhaust = False
    atr_val = 0.0
    adx_val = 0.0
    adx_strength = ""
    adx_dir = ""

    if summary:
        coral_dir = summary.coral.direction.value
        hma_cross = summary.hma.crossover.value
        ell_label = summary.elliott.wave_label.value
        ell_exhaust = summary.elliott.count_exhaustion
        atr_val = round(summary.atr.value, 2)
        adx_val = round(summary.adx.adx_value, 2)
        adx_strength = summary.adx.trend_strength
        adx_dir = summary.adx.direction

    record = TradeRecord(
        id=_generate_id(),
        timestamp=datetime.now(IST).isoformat(),
        signal=verdict.signal.value,
        entry_price=round(verdict.last_price, 2),
        coral_direction=coral_dir,
        hma_crossover=hma_cross,
        elliott_label=ell_label,
        elliott_exhaustion=ell_exhaust,
        atr_value=atr_val,
        adx_value=adx_val,
        adx_trend_strength=adx_strength,
        adx_direction=adx_dir,
    )

    trades.append(record.to_dict())
    _save_trades(trades)
    logger.info(
        "Trade recorded: %s at %.2f (signal=%s)",
        record.id, record.entry_price, record.signal,
    )


def check_pending_trades() -> int:
    """
    Check all pending trades for 1-hour price movement.

    Runs on every scheduler cycle (every 15 minutes).
    Checks if enough time has elapsed since the trade was recorded.

    Returns
    -------
    int
        Number of trades that were updated.
    """
    trades = _load_trades()
    now = datetime.now(IST)
    updated = 0

    for trade in trades:
        if trade.get("checked_1h", False):
            continue

        # Parse trade timestamp
        try:
            trade_time = datetime.fromisoformat(trade["timestamp"])
        except (ValueError, KeyError):
            continue

        # Check if 1 hour has passed
        elapsed = (now - trade_time).total_seconds()
        if elapsed < FOLLOWUP_HOURS * 3600:
            continue

        # 1 hour has passed — check price movement
        signal = trade.get("signal", "")
        entry_price = trade.get("entry_price", 0)

        # Fetch current price
        df = get_cached_data()
        if df is None or df.empty:
            logger.warning("Cannot check trade %s: no data", trade.get("id"))
            continue

        current_price = float(df["Close"].iloc[-1])
        move = current_price - entry_price  # positive = price UP

        trade["price_1h"] = round(current_price, 2)
        trade["move_1h"] = round(move, 2)
        trade["checked_1h"] = True

        # Determine win/loss based on signal direction
        if signal == "BUY":
            if move >= WIN_THRESHOLD_POINTS:
                trade["is_win"] = True
                trade["pnl_label"] = "WIN"
            elif move <= -WIN_THRESHOLD_POINTS:
                trade["is_win"] = False
                trade["pnl_label"] = "LOSS"
            else:
                trade["is_win"] = False
                trade["pnl_label"] = "NO_MOVE"
        elif signal == "SELL":
            if move <= -WIN_THRESHOLD_POINTS:
                trade["is_win"] = True
                trade["pnl_label"] = "WIN"
            elif move >= WIN_THRESHOLD_POINTS:
                trade["is_win"] = False
                trade["pnl_label"] = "LOSS"
            else:
                trade["is_win"] = False
                trade["pnl_label"] = "NO_MOVE"

        updated += 1
        logger.info(
            "Trade %s: entry=%.2f, now=%.2f, move=%.2f, result=%s",
            trade.get("id", "?"), entry_price, current_price, move,
            trade.get("pnl_label", "?"),
        )

    if updated > 0:
        _save_trades(trades)

    return updated


def get_performance_stats() -> dict:
    """
    Calculate performance statistics from the trade journal.

    Returns
    -------
    dict
        Performance statistics.
    """
    trades = _load_trades()
    completed = [t for t in trades if t.get("checked_1h", False)]

    total = len(completed)
    wins = sum(1 for t in completed if t.get("is_win"))
    losses = sum(1 for t in completed if t.get("is_win") is False and t.get("pnl_label") == "LOSS")
    no_moves = sum(1 for t in completed if t.get("pnl_label") == "NO_MOVE")

    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0

    # Average move
    moves = [t.get("move_1h", 0) or 0 for t in completed]
    avg_move = round(sum(moves) / len(moves), 2) if moves else 0.0

    # Average win/loss magnitude
    win_moves = [t.get("move_1h", 0) or 0 for t in completed if t.get("is_win")]
    loss_moves = [t.get("move_1h", 0) or 0 for t in completed if t.get("is_win") is False and t.get("pnl_label") == "LOSS"]

    avg_win = round(sum(win_moves) / len(win_moves), 2) if win_moves else 0.0
    avg_loss = round(sum(loss_moves) / len(loss_moves), 2) if loss_moves else 0.0

    # Win rate by signal type
    buy_trades = [t for t in completed if t.get("signal") == "BUY"]
    sell_trades = [t for t in completed if t.get("signal") == "SELL"]
    buy_wins = sum(1 for t in buy_trades if t.get("is_win"))
    sell_wins = sum(1 for t in sell_trades if t.get("is_win"))

    buy_win_rate = round(buy_wins / len(buy_trades) * 100, 1) if buy_trades else 0.0
    sell_win_rate = round(sell_wins / len(sell_trades) * 100, 1) if sell_trades else 0.0

    # Performance by ADX strength
    strong_trades = [t for t in completed if t.get("adx_trend_strength") == "strong"]
    weak_trades = [t for t in completed if t.get("adx_trend_strength") == "weak"]
    strong_wins = sum(1 for t in strong_trades if t.get("is_win"))
    weak_wins = sum(1 for t in weak_trades if t.get("is_win"))
    strong_win_rate = round(strong_wins / len(strong_trades) * 100, 1) if strong_trades else 0.0
    weak_win_rate = round(weak_wins / len(weak_trades) * 100, 1) if weak_trades else 0.0

    return {
        "total_signals": len(trades),
        "completed_checks": total,
        "wins": wins,
        "losses": losses,
        "no_moves": no_moves,
        "win_rate_pct": win_rate,
        "avg_move_points": avg_move,
        "avg_win_points": avg_win,
        "avg_loss_points": avg_loss,
        "win_rate_buy_pct": buy_win_rate,
        "win_rate_sell_pct": sell_win_rate,
        "buy_count": len(buy_trades),
        "sell_count": len(sell_trades),
        "strong_adx_win_rate_pct": strong_win_rate,
        "weak_adx_win_rate_pct": weak_win_rate,
        "strong_adx_count": len(strong_trades),
        "weak_adx_count": len(weak_trades),
        "threshold_points": WIN_THRESHOLD_POINTS,
        "followup_hours": FOLLOWUP_HOURS,
        "last_updated": datetime.now(IST).isoformat(),
    }


def get_recent_trades(limit: int = 20) -> list[dict]:
    """Return recent trade records (newest first)."""
    trades = _load_trades()
    return list(reversed(trades[-limit:]))


def get_strategy_suggestion(verdict: AnalysisVerdict, summary: Optional[IndicatorSummary] = None) -> dict:
    """
    Suggest an options strategy based on current indicators.

    Parameters
    ----------
    verdict : AnalysisVerdict
        Current verdict.
    summary : IndicatorSummary or None
        Current indicator summary.

    Returns
    -------
    dict
        Strategy suggestion with rationale.
    """
    signal = verdict.signal.value

    if summary is None:
        return {
            "suggested_strategy": "Not enough data",
            "rationale": "Indicators not available.",
        }

    coral = summary.coral
    elliott = summary.elliott
    atr = summary.atr
    adx = summary.adx

    is_bullish = coral.direction.value == "bullish" or adx.direction == "bullish"
    is_bearish = coral.direction.value == "bearish" or adx.direction == "bearish"
    is_ranging = adx.trend_strength == "weak"
    is_volatile = atr.relative_strength == "high"
    is_exhausted = elliott.count_exhaustion

    if is_ranging or signal == "HOLD":
        return {
            "suggested_strategy": "Iron Condor / Short Strangle",
            "rationale": (
                f"ADX {adx.adx_value} ({adx.trend_strength}) indicates low momentum. "
                f"Range-bound market favors theta strategies. "
                f"Sell OTM call and OTM put credit spreads.",
            ),
            "direction": "neutral",
            "adx_context": adx.trend_strength,
            "volatility_context": atr.relative_strength,
            "exhaustion_warning": "Trend exhaustion detected — reversal possible" if is_exhausted else None,
        }

    if is_bullish and not is_bearish:
        return {
            "suggested_strategy": "Bull Put Credit Spread",
            "rationale": (
                f"Bullish bias from {coral.direction.value} Coral and {adx.direction} ADX. "
                f"Sell OTM Put, buy further OTM Put for defined risk.",
            ),
            "direction": "bullish",
            "adx_context": adx.trend_strength,
            "volatility_context": atr.relative_strength,
            "exhaustion_warning": "Wave C exhaustion — trend may reverse soon" if is_exhausted else None,
        }

    if is_bearish and not is_bullish:
        return {
            "suggested_strategy": "Bear Call Credit Spread",
            "rationale": (
                f"Bearish bias from {coral.direction.value} Coral and {adx.direction} ADX. "
                f"Sell OTM Call, buy further OTM Call for defined risk.",
            ),
            "direction": "bearish",
            "adx_context": adx.trend_strength,
            "volatility_context": atr.relative_strength,
            "exhaustion_warning": "Wave C exhaustion — trend may reverse soon" if is_exhausted else None,
        }

    return {
        "suggested_strategy": "Wait / Cash",
        "rationale": "Mixed signals across indicators. No clear setup.",
        "direction": "neutral",
        "adx_context": adx.trend_strength,
        "volatility_context": atr.relative_strength,
    }