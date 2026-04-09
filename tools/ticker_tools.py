"""
Single-purpose tool: resolve a company name to its stock ticker symbol.
Returns multiple candidates so the LLM can pick the right exchange/listing.
"""
import yfinance as yf

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_ticker",
        "description": (
            "Search for stock ticker symbols matching a company name. "
            "Returns up to 5 candidates with ticker, name, exchange, and listing type "
            "so the caller can pick the best match for the requested market/exchange."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name to search for (e.g. 'Apple', 'HDFC Bank').",
                }
            },
            "required": ["company_name"],
        },
    },
}

# Maps yfinance exchange codes to human-readable names
_EXCHANGE_LABELS = {
    "NSI": "NSE (India)",
    "BSE": "BSE (India)",
    "NYQ": "NYSE (US)",
    "NMS": "NASDAQ (US)",
    "LSE": "LSE (UK)",
    "TOR": "TSX (Canada)",
    "ASX": "ASX (Australia)",
    "HKG": "HKEX (Hong Kong)",
    "SHH": "SSE (Shanghai)",
    "SHZ": "SZSE (Shenzhen)",
    "FRA": "Frankfurt (Germany)",
    "PAR": "Euronext Paris",
    "AMS": "Euronext Amsterdam",
}


def verify_ticker(ticker: str) -> bool:
    """
    Return True if ticker resolves to live price data on Yahoo Finance.
    Used to distinguish real tickers (AAPL) from ambiguous abbreviations (HDFC).
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period="5d")
        return not hist.empty
    except Exception:
        return False


def search_ticker(company_name: str, max_candidates: int = 5, max_results: int = 30) -> dict:
    """
    Return up to max_candidates ticker candidates for a company name.

    Strategy:
      1. Fetch up to max_results raw results from yfinance (funds dominate short
         queries like 'hdfc', so we need to look deeper to find actual equities).
      2. Split into equities and everything else.
      3. Return top-N equities; fall back to top-N non-equities only if no
         equity was found at all.

    Each candidate carries ticker, name, exchange label, type, and ADR flag.
    """
    try:
        results = yf.Search(company_name, max_results=max_results)
        quotes = results.quotes
        if not quotes:
            return {"error": f"No ticker found for '{company_name}'"}

        _EQUITY_TYPES = {"EQUITY", "ADR", "ADRC"}
        _SKIP_TYPES   = {"MUTUALFUND", "ETF", "INDEX", "FUTURE", "OPTION", "CURRENCY"}

        equities, others = [], []
        for q in quotes:
            qt = q.get("quoteType", "").upper()
            if qt in _SKIP_TYPES:
                others.append(q)
            else:
                equities.append(q)

        pool = equities[:max_candidates] if equities else others[:max_candidates]

        candidates = []
        for q in pool:
            exchange_code = q.get("exchange", "")
            quote_type    = q.get("quoteType", "")
            candidates.append({
                "ticker":       q.get("symbol", ""),
                "name":         q.get("longname") or q.get("shortname", ""),
                "exchange":     _EXCHANGE_LABELS.get(exchange_code, exchange_code),
                "raw_exchange": exchange_code,
                "type":         quote_type,
                "is_adr":       quote_type.upper() in ("ADR", "ADRC"),
            })

        return {"candidates": candidates}
    except Exception as exc:
        return {"error": str(exc)}
