"""
DataValidatorAgent
------------------
Single purpose: validate raw price data before it is consumed by other agents.

Checks performed
----------------
Current price data
  • Freshness    – trade_date is not older than MAX_STALE_DAYS calendar days
  • Price sanity – price > 0
  • Change sanity– |change_pct| < ANOMALY_PCT_THRESHOLD (flags extreme moves)
  • Volume       – volume > 0 (zero volume may indicate a data gap)

Historical / date-range data (per row)
  • OHLC consistency  – high >= low, high >= open, high >= close,
                        low  <= open, low  <= close
  • Price sanity      – all price columns > 0
  • Gap detection     – flags if >GAP_THRESHOLD % of expected trading days
                        are missing from the series

The validator does NOT raise exceptions.  It returns a ValidationResult dict:
  {
    "valid":    bool,           # False only if hard errors exist
    "errors":   [str, ...],     # Must-fix problems
    "warnings": [str, ...],     # Advisory notices
  }
which is merged into the tool result so the LLM can surface issues naturally.
"""

from datetime import date, timedelta
from typing import Any

# ── Thresholds (tune as needed) ───────────────────────────────────────────────
MAX_STALE_DAYS        = 5    # Current price older than this → stale warning
ANOMALY_PCT_THRESHOLD = 25   # |change%| above this → anomaly warning
GAP_THRESHOLD_PCT     = 20   # >20% missing rows in a series → gap warning


class DataValidator:
    """Pure-Python validator — no LLM, no I/O, deterministic."""

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_current_price(self, data: dict) -> dict:
        """Validate a get_current_price() result dict."""
        errors, warnings = [], []

        if "error" in data:
            return self._result(False, errors=[f"Tool error: {data['error']}"])

        # Freshness
        trade_date = self._parse_date(data.get("trade_date"))
        if trade_date:
            age = (date.today() - trade_date).days
            if age > MAX_STALE_DAYS:
                warnings.append(
                    f"Data may be stale: last trade date is {trade_date} "
                    f"({age} days ago). Market may have been closed."
                )

        # Price sanity
        price = data.get("price")
        if price is not None and price <= 0:
            errors.append(f"Invalid price: {price} (must be > 0)")

        # Anomaly detection
        change_pct = data.get("change_pct")
        if change_pct is not None and abs(change_pct) > ANOMALY_PCT_THRESHOLD:
            warnings.append(
                f"Unusually large price move: {change_pct:+.2f}% — "
                "verify this is not a data error (split, dividend, etc.)."
            )

        # Volume
        volume = data.get("volume", 0)
        if volume == 0:
            warnings.append("Volume is 0 — the market may have been closed or data is incomplete.")

        return self._result(not errors, errors=errors, warnings=warnings)

    def validate_historical_prices(self, data: dict) -> dict:
        """Validate a get_historical_prices() or get_price_range_between_dates() result."""
        errors, warnings = [], []

        if "error" in data:
            return self._result(False, errors=[f"Tool error: {data['error']}"])

        rows = data.get("data", [])
        if not rows:
            errors.append("No rows returned in historical data.")
            return self._result(False, errors=errors)

        # Per-row OHLC consistency
        bad_rows = []
        for r in rows:
            o, h, l, c = r.get("open"), r.get("high"), r.get("low"), r.get("close")
            if None in (o, h, l, c):
                continue
            issues = []
            if h < l:
                issues.append("high < low")
            if h < o:
                issues.append("high < open")
            if h < c:
                issues.append("high < close")
            if l > o:
                issues.append("low > open")
            if l > c:
                issues.append("low > close")
            if any(v <= 0 for v in (o, h, l, c)):
                issues.append("non-positive price")
            if issues:
                bad_rows.append(f"{r['date']}: {', '.join(issues)}")

        if bad_rows:
            # Report first 3 bad rows to keep message readable
            sample = bad_rows[:3]
            suffix = f" (and {len(bad_rows)-3} more)" if len(bad_rows) > 3 else ""
            errors.append(
                f"OHLC consistency errors in {len(bad_rows)} row(s): "
                + "; ".join(sample) + suffix
            )

        # Gap detection (approximate — assumes 5-day trading weeks)
        if len(rows) >= 2:
            first = self._parse_date(rows[0]["date"])
            last  = self._parse_date(rows[-1]["date"])
            if first and last:
                calendar_days  = (last - first).days + 1
                expected_rows  = max(1, int(calendar_days * 5 / 7))
                actual_rows    = len(rows)
                gap_pct        = max(0, (expected_rows - actual_rows) / expected_rows * 100)
                if gap_pct > GAP_THRESHOLD_PCT:
                    warnings.append(
                        f"~{gap_pct:.0f}% of expected trading days are missing "
                        f"({actual_rows} rows vs ~{expected_rows} expected). "
                        "Possible data gaps or holidays."
                    )

        return self._result(not errors, errors=errors, warnings=warnings)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(value: Any):
        if not value:
            return None
        try:
            if isinstance(value, date):
                return value
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    @staticmethod
    def _result(valid: bool, errors: list = None, warnings: list = None) -> dict:
        return {
            "valid":    valid,
            "errors":   errors   or [],
            "warnings": warnings or [],
        }
