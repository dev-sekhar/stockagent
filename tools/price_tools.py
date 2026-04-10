"""
Single-purpose tool: fetch the latest traded price for a stock ticker.
"""
import yfinance as yf
from tools.data_validator  import DataValidator
from tools.access_gateway  import gateway

_validator = DataValidator()

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_price",
        "description": "Get the last traded price, change, change%, and volume for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, TSLA, MSFT).",
                }
            },
            "required": ["ticker"],
        },
    },
}


def get_current_price(ticker: str) -> dict:
    """Return last traded price, change, change%, volume, and trade date."""
    try:
        gateway.check("MARKET_DATA")
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="2d")
        if hist.empty:
            return {"error": f"No price data found for '{ticker}'"}

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else None
        price = round(float(last["Close"]), 4)
        change = round(float(last["Close"]) - float(prev["Close"]), 4) if prev is not None else None
        change_pct = round(change / float(prev["Close"]) * 100, 2) if change is not None else None

        result = {
            "ticker": ticker.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": int(last.get("Volume", 0)),
            "trade_date": str(last.name.date()),
            "currency": getattr(t.fast_info, "currency", "USD"),
        }

        validation = _validator.validate_current_price(result)
        if validation["errors"] or validation["warnings"]:
            result["validation"] = validation

        return result
    except Exception as exc:
        return {"error": str(exc)}
