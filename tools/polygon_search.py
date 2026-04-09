"""
PolygonSearchAgent
------------------
Single purpose: search Polygon.io's reference ticker database for companies
matching a name or partial name.

Uses the free Polygon.io REST API:
  GET https://api.polygon.io/v3/reference/tickers
      ?search=<query>&active=true&market=stocks&limit=20&apiKey=<key>

Free tier covers US markets (NYSE, NASDAQ) very well.
Indian NSE/BSE coverage varies — the orchestrator supplements with a direct
yfinance search to fill any gaps.

Sign up for a free API key at: https://polygon.io/dashboard/signup
Add POLYGON_API_KEY to your .env file.
"""
import os
import requests

_POLYGON_BASE = "https://api.polygon.io/v3/reference/tickers"

# Map Polygon exchange MIC codes to human-readable labels
_EXCHANGE_MAP = {
    "XNAS": "NASDAQ (US)",
    "XNYS": "NYSE (US)",
    "XNSE": "NSE (India)",
    "XBOM": "BSE (India)",
    "XLON": "LSE (UK)",
    "XFRA": "Frankfurt (Germany)",
    "XHKG": "HKEX (Hong Kong)",
    "XTSX": "TSX (Canada)",
    "XASX": "ASX (Australia)",
}


class PolygonSearchAgent:
    """
    Searches Polygon.io reference database for companies matching a query.
    Returns a structured list compatible with ExchangeDisambiguatorAgent.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """
        Return [{name, ticker, exchange, type, is_adr, raw_exchange}, ...].
        Returns empty list on error (caller should fall back to yfinance search).
        """
        try:
            params = {
                "search":   query,
                "active":   "true",
                "market":   "stocks",
                "limit":    limit,
                "apiKey":   self.api_key,
            }
            resp = requests.get(_POLYGON_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                return []

            candidates = []
            for r in results:
                exchange_mic = r.get("primary_exchange", "")
                ticker       = r.get("ticker", "")
                name         = r.get("name", "")
                asset_type   = r.get("type", "CS")    # CS = Common Stock

                # Skip non-equity types (ETF, FUND, WARRANT, UNIT, etc.)
                if asset_type not in ("CS", "ADRC", "ADR"):
                    continue

                candidates.append({
                    "ticker":       ticker,
                    "name":         name,
                    "exchange":     _EXCHANGE_MAP.get(exchange_mic, exchange_mic),
                    "raw_exchange": exchange_mic,
                    "type":         "EQUITY" if asset_type == "CS" else asset_type,
                    "is_adr":       asset_type in ("ADR", "ADRC"),
                })

            print(
                f"  📡 [PolygonSearchAgent] Found {len(candidates)} candidate(s) for '{query}'"
                + (f": {', '.join(c['name'][:25] for c in candidates[:3])}…"
                   if len(candidates) > 3 else
                   f": {', '.join(c['name'] for c in candidates)}" if candidates else "")
            )
            return candidates

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                print(f"  ⚠️  [PolygonSearchAgent] API key invalid or free-tier limit hit")
            else:
                print(f"  ⚠️  [PolygonSearchAgent] HTTP error: {e}")
            return []
        except Exception as exc:
            print(f"  ⚠️  [PolygonSearchAgent] Search failed ({exc})")
            return []

    @staticmethod
    def from_env() -> "PolygonSearchAgent | None":
        """Return an agent if POLYGON_API_KEY is set, else None."""
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            return None
        return PolygonSearchAgent(key)
