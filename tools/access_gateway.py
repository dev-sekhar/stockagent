"""
access_gateway.py — External Access Gatekeeper
-----------------------------------------------
Single control point for ALL outbound network calls made by the agent.

Categories
----------
  LLM          Groq LLM API (chat completions)
  MARKET_DATA  yfinance (price / history / company metadata)
  POLYGON      Polygon.io search API
  NEWS_RSS     Yahoo Finance / Google News RSS feeds

Usage
-----
  from tools.access_gateway import gateway

  # Guard only — raises AccessDeniedError if disabled
  gateway.check("MARKET_DATA")

  # Guard + execute + log
  result = gateway.call("MARKET_DATA", yf.Ticker, "AAPL")

Configuration (env vars in .env, all default to enabled)
---------------------------------------------------------
  GATEWAY_LLM_ENABLED=false          # disable Groq calls
  GATEWAY_MARKET_DATA_ENABLED=false  # disable yfinance
  GATEWAY_POLYGON_ENABLED=false      # disable Polygon search
  GATEWAY_NEWS_RSS_ENABLED=false     # disable RSS news feeds
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from config import (
    GATEWAY_LLM_ENABLED,
    GATEWAY_MARKET_DATA_ENABLED,
    GATEWAY_POLYGON_ENABLED,
    GATEWAY_NEWS_RSS_ENABLED,
)

logger = logging.getLogger(__name__)

# Human-readable labels for console output
_LABELS: dict[str, str] = {
    "LLM":         "Groq LLM API",
    "MARKET_DATA": "yfinance / market data",
    "POLYGON":     "Polygon.io search",
    "NEWS_RSS":    "RSS news feeds",
}


class AccessDeniedError(RuntimeError):
    """Raised when a disabled external-access category is requested."""


class AccessGateway:
    """
    Lightweight gatekeeper for external network calls.

    All external calls in the codebase should go through this gateway so
    there is a single place to enable/disable categories, log usage, and
    add future policy controls (quota, caching, circuit-breaking).
    """

    def __init__(self) -> None:
        self._enabled: dict[str, bool] = {
            "LLM":         GATEWAY_LLM_ENABLED,
            "MARKET_DATA": GATEWAY_MARKET_DATA_ENABLED,
            "POLYGON":     GATEWAY_POLYGON_ENABLED,
            "NEWS_RSS":    GATEWAY_NEWS_RSS_ENABLED,
        }
        self._call_counts: dict[str, int] = {k: 0 for k in self._enabled}

    # ── public interface ──────────────────────────────────────────────────────

    def check(self, category: str) -> None:
        """
        Assert that *category* is currently enabled.

        Raises AccessDeniedError if it has been disabled via env var.
        Call this at the top of any function that makes an outbound request.
        """
        cat = category.upper()
        if not self._enabled.get(cat, True):
            label = _LABELS.get(cat, cat)
            raise AccessDeniedError(
                f"External access to '{label}' is disabled "
                f"(set GATEWAY_{cat}_ENABLED=true to re-enable)."
            )

    def call(self, category: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Check access, then call *fn* with the supplied arguments.

        Logs the call (DEBUG) and increments the call counter.
        Re-raises any exception from *fn* unchanged.
        """
        self.check(category)
        cat = category.upper()
        self._call_counts[cat] = self._call_counts.get(cat, 0) + 1
        label = _LABELS.get(cat, cat)
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - start
            logger.debug(
                "[Gateway] %s | %s | %.3fs | call #%d",
                cat, fn.__name__ if hasattr(fn, "__name__") else str(fn),
                elapsed, self._call_counts[cat],
            )
            return result
        except Exception:
            elapsed = time.monotonic() - start
            logger.debug(
                "[Gateway] %s | %s | FAILED | %.3fs",
                cat, fn.__name__ if hasattr(fn, "__name__") else str(fn), elapsed,
            )
            raise

    def status(self) -> dict[str, bool]:
        """Return a copy of the current enabled/disabled state per category."""
        return dict(self._enabled)

    def report(self) -> str:
        """Return a one-line status summary suitable for startup logging."""
        parts = []
        for cat, enabled in self._enabled.items():
            icon  = "✅" if enabled else "🚫"
            label = _LABELS.get(cat, cat)
            parts.append(f"{icon} {label}")
        return "  🛡️  [AccessGateway] " + " | ".join(parts)


# Module-level singleton — import and use everywhere
gateway = AccessGateway()
