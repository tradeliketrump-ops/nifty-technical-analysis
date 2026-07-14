"""
Alert system for NIFTY Technical Analysis.

Sends Telegram notifications when the trading signal changes
(BUY / SELL / HOLD transitions).

Uses the simple HTTP API (no python-telegram-bot dependency).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

from indicators import AnalysisVerdict

logger = logging.getLogger("nifty_alerter")

# ─── File paths ─────────────────────────────────────────────────────────

ALERTS_DIR = Path(__file__).parent
PREVIOUS_SIGNAL_FILE = ALERTS_DIR / ".previous_signal.json"

# ─── Telegram config from env vars ──────────────────────────────────────

TELEGRAM_BOT_TOKEN: Optional[str] = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: Optional[str] = os.environ.get("TELEGRAM_CHAT_ID")

# ─── Signal tracking ────────────────────────────────────────────────────


def _signal_file_path() -> Path:
    return PREVIOUS_SIGNAL_FILE


def load_previous_signal() -> Optional[str]:
    """
    Load the last known signal from disk.

    Returns
    -------
    str or None
        "BUY", "SELL", "HOLD", or None if no history exists.
    """
    path = _signal_file_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("signal")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_previous_signal(signal: str) -> None:
    """Persist the current signal to disk."""
    path = _signal_file_path()
    try:
        path.write_text(
            json.dumps({"signal": signal, "timestamp": time.time()}),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not save previous signal: %s", exc)


# ─── Telegram sender ────────────────────────────────────────────────────


def send_telegram_message(message: str) -> bool:
    """
    Send a text message via Telegram Bot API.

    Returns True if sent successfully, False otherwise.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram alert sent successfully.")
            return True
        else:
            logger.warning(
                "Telegram API error: %s %s",
                resp.status_code,
                resp.text,
            )
            return False
    except requests.RequestException as exc:
        logger.warning("Telegram request failed: %s", exc)
        return False


def build_alert_message(
    old_signal: str,
    new_signal: str,
    verdict: AnalysisVerdict,
) -> str:
    """
    Format a Telegram alert message from the verdict data.
    """
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    old_emoji = emoji_map.get(old_signal, "⚪")
    new_emoji = emoji_map.get(new_signal, "⚪")

    lines = [
        f"{new_emoji} <b>NIFTY Signal Changed</b>",
        "",
        f"{old_emoji} {old_signal}  →  {new_emoji} <b>{new_signal}</b>",
        "",
        f"💰 Price: {verdict.last_price:,.2f}",
        "",
        f"{verdict.coral_summary}",
        f"{verdict.hma_summary}",
        f"{verdict.elliott_summary}",
        f"{verdict.atr_summary}",
        f"{verdict.adx_summary}",
        "",
        f"📋 {verdict.core_thesis}",
    ]
    return "\n".join(lines)


def build_status_message(verdict: AnalysisVerdict) -> str:
    """
    Format a periodic status update message.
    """
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    emoji = emoji_map.get(verdict.signal.value, "⚪")

    lines = [
        f"{emoji} <b>NIFTY Status Update</b>",
        "",
        f"💰 Price: {verdict.last_price:,.2f}",
        f"📊 Signal: <b>{verdict.signal.value}</b>",
        "",
        f"{verdict.coral_summary}",
        f"{verdict.hma_summary}",
        f"{verdict.elliott_summary}",
        f"{verdict.atr_summary}",
        f"{verdict.adx_summary}",
    ]
    return "\n".join(lines)


# ─── Main alert function ────────────────────────────────────────────────


def check_and_alert(
    verdict: AnalysisVerdict,
    force_notify: bool = False,
) -> None:
    """
    Compare the current signal with the previous one and send alerts.

    Parameters
    ----------
    verdict : AnalysisVerdict
        The latest analysis verdict.
    force_notify : bool
        If True, send a status update even if signal hasn't changed.
    """
    current_signal = verdict.signal.value
    previous_signal = load_previous_signal()

    # Save current signal for next comparison
    save_previous_signal(current_signal)

    if previous_signal is None:
        # First run — just save, don't alert
        logger.info("Initial signal saved: %s", current_signal)
        # But send a welcome message if Telegram is configured
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            msg = build_status_message(verdict)
            send_telegram_message(
                f"🤖 <b>NIFTY Bot Started</b>\n\n"
                f"Monitoring signal changes.\n\n"
                + msg
            )
        return

    if previous_signal != current_signal:
        # Signal changed → alert!
        logger.info(
            "Signal changed: %s → %s",
            previous_signal,
            current_signal,
        )
        msg = build_alert_message(previous_signal, current_signal, verdict)
        send_telegram_message(msg)
    elif force_notify:
        # Send periodic status update
        msg = build_status_message(verdict)
        send_telegram_message(msg)


def is_configured() -> bool:
    """Check if Telegram alerting is configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)