"""
Single-purpose tools: fetch historical OHLCV data for a ticker.
  • get_historical_prices         – relative period (1mo, 1y, ytd …)
  • get_price_range_between_dates – explicit start/end calendar dates
"""
import yfinance as yf
from tabulate import tabulate
from tools.data_validator  import DataValidator
from tools.access_gateway  import gateway

_validator = DataValidator()

HISTORICAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_historical_prices",
        "description": (
            "Fetch OHLCV data for a relative time period such as "
            "'last month', 'last year', 'year-to-date', '5 years', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."},
                "period": {
                    "type": "string",
                    "description": "yfinance period: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max",
                    "default": "1mo",
                },
                "interval": {
                    "type": "string",
                    "description": "Bar interval: 1d (default) 1wk 1mo",
                    "default": "1d",
                },
            },
            "required": ["ticker"],
        },
    },
}

DATE_RANGE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_price_range_between_dates",
        "description": (
            "Fetch daily OHLCV data between two specific calendar dates, "
            "e.g. 'from Jan 2024 to June 2024'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."},
                "start":  {"type": "string", "description": "Start date YYYY-MM-DD."},
                "end":    {"type": "string", "description": "End date YYYY-MM-DD."},
            },
            "required": ["ticker", "start", "end"],
        },
    },
}


def _to_table(hist, is_intraday: bool = False) -> tuple:
    hist = hist.round(4)
    fmt = "%Y-%m-%d %H:%M" if is_intraday else "%Y-%m-%d"
    hist.index = hist.index.strftime(fmt)
    rows = [
        {
            "date":   str(date),
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  float(row["Close"]),
            "volume": int(row["Volume"]),
        }
        for date, row in hist.iterrows()
    ]
    table = tabulate(
        [[r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in rows],
        headers=["Date", "Open", "High", "Low", "Close", "Volume"],
        tablefmt="rounded_outline",
    )
    return rows, table


def _summarise(rows: list, ticker: str, period: str) -> dict:
    """
    Compute period statistics from OHLCV rows.
    Returns a compact dict (~200 tokens) suitable for LLM narration.
    """
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    closes   = [r["close"] for r in rows]
    volumes  = [r["volume"] for r in rows]
    high_row = max(rows, key=lambda r: r["high"])
    low_row  = min(rows, key=lambda r: r["low"])
    pct_chg  = ((last["close"] - first["open"]) / first["open"] * 100) if first["open"] else 0

    return {
        "ticker":          ticker,
        "period":          period,
        "trading_days":    len(rows),
        "start_date":      first["date"],
        "end_date":        last["date"],
        "open_price":      first["open"],
        "close_price":     last["close"],
        "period_high":     high_row["high"],
        "period_high_date":high_row["date"],
        "period_low":      low_row["low"],
        "period_low_date": low_row["date"],
        "avg_volume":      int(sum(volumes) / len(volumes)),
        "pct_change":      round(pct_chg, 2),
        "trend":           "up" if pct_chg > 0 else "down" if pct_chg < 0 else "flat",
        # Recent 5 rows so LLM can show latest movement
        "recent_rows":     rows[-5:],
    }


def get_historical_prices(ticker: str, period: str = "1mo", interval: str = "1d") -> dict:
    try:
        gateway.check("MARKET_DATA")
        hist = yf.Ticker(ticker.upper()).history(period=period, interval=interval)
        if hist.empty:
            return {"error": f"No historical data for {ticker} (period={period})"}
        is_intraday = interval.endswith("m") or interval.endswith("h")
        rows, table = _to_table(hist, is_intraday)

        summary = _summarise(rows, ticker.upper(), period)
        result  = {**summary, "interval": interval}

        # Only attach the full table for short periods (≤ 1 month = ≤ 31 rows)
        if len(rows) <= 31:
            result["table"] = table

        validation = _validator.validate_historical_prices(
            {"ticker": ticker.upper(), "period": period, "data": rows}
        )
        if validation["errors"] or validation["warnings"]:
            result["validation"] = validation

        return result
    except Exception as exc:
        return {"error": str(exc)}


def get_price_range_between_dates(ticker: str, start: str, end: str) -> dict:
    try:
        gateway.check("MARKET_DATA")
        hist = yf.Ticker(ticker.upper()).history(start=start, end=end, interval="1d")
        if hist.empty:
            return {"error": f"No data for {ticker} between {start} and {end}"}
        rows, table = _to_table(hist)

        summary = _summarise(rows, ticker.upper(), f"{start} to {end}")
        result  = {**summary}

        if len(rows) <= 31:
            result["table"] = table

        validation = _validator.validate_historical_prices(
            {"ticker": ticker.upper(), "data": rows}
        )
        if validation["errors"] or validation["warnings"]:
            result["validation"] = validation

        return result
    except Exception as exc:
        return {"error": str(exc)}
