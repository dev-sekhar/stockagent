"""
SectorIndexAgent
----------------
AI Agent: given a stock ticker, identify its industry sector and return the
most appropriate sector-specific benchmark index to overlay on a chart.

This is a genuine AI agent — it uses an LLM to reason about sector mapping
when the fast keyword lookup does not produce a match.

Resolution order:
  1. Fast path: yfinance sector/industry → keyword lookup → same-exchange index
  2. LLM fallback: ask the model to map the sector string to a known key
  3. Cross-exchange fallback: any exchange that has a match
  4. None if still unresolved
"""
import re
import yfinance as yf
from groq import Groq

# ── Sector index map ──────────────────────────────────────────────────────────
# Structure: sector_key → [(exchange_suffix, index_ticker, index_label), ...]
# exchange_suffix="" matches US (no-suffix) tickers.
# Try entries in order; first one that yfinance can actually fetch wins.

_SECTOR_MAP: dict[str, list[tuple[str, str, str]]] = {
    "healthcare": [
        (".NS",  "^CNXPHARMA",    "Nifty Pharma"),
        (".BO",  "^CNXPHARMA",    "Nifty Pharma"),
        ("",     "XLV",           "S&P Health ETF"),
        ("",     "^SP500-35",     "S&P Healthcare"),
    ],
    "pharma": [
        (".NS",  "^CNXPHARMA",    "Nifty Pharma"),
        (".BO",  "^CNXPHARMA",    "Nifty Pharma"),
        ("",     "XLV",           "S&P Health ETF"),
    ],
    "technology": [
        (".NS",  "^CNXIT",        "Nifty IT"),
        (".BO",  "^CNXIT",        "Nifty IT"),
        ("",     "XLK",           "S&P Tech ETF"),
        ("",     "^IXIC",         "NASDAQ"),
    ],
    "financial services": [
        (".NS",  "^NSEBANK",      "Nifty Bank"),
        (".BO",  "^NSEBANK",      "Nifty Bank"),
        ("",     "XLF",           "S&P Financials ETF"),
    ],
    "banking": [
        (".NS",  "^NSEBANK",      "Nifty Bank"),
        (".BO",  "^NSEBANK",      "Nifty Bank"),
        ("",     "XLF",           "S&P Financials ETF"),
    ],
    "consumer defensive": [
        (".NS",  "^CNXFMCG",      "Nifty FMCG"),
        (".BO",  "^CNXFMCG",      "Nifty FMCG"),
        ("",     "XLP",           "S&P Staples ETF"),
    ],
    "consumer cyclical": [
        (".NS",  "^CNXAUTO",      "Nifty Auto"),
        (".BO",  "^CNXAUTO",      "Nifty Auto"),
        ("",     "XLY",           "S&P Discret. ETF"),
    ],
    "automobiles": [
        (".NS",  "^CNXAUTO",      "Nifty Auto"),
        (".BO",  "^CNXAUTO",      "Nifty Auto"),
        ("",     "XLY",           "S&P Discret. ETF"),
    ],
    "basic materials": [
        (".NS",  "^CNXMETAL",     "Nifty Metal"),
        (".BO",  "^CNXMETAL",     "Nifty Metal"),
        ("",     "XLB",           "S&P Materials ETF"),
    ],
    "metals": [
        (".NS",  "^CNXMETAL",     "Nifty Metal"),
        (".BO",  "^CNXMETAL",     "Nifty Metal"),
        ("",     "XLB",           "S&P Materials ETF"),
    ],
    "energy": [
        (".NS",  "^CNXENERGY",    "Nifty Energy"),
        (".BO",  "^CNXENERGY",    "Nifty Energy"),
        ("",     "XLE",           "S&P Energy ETF"),
    ],
    "utilities": [
        (".NS",  "^CNXINFRA",     "Nifty Infra"),
        (".BO",  "^CNXINFRA",     "Nifty Infra"),
        ("",     "XLU",           "S&P Utilities ETF"),
    ],
    "industrials": [
        (".NS",  "^CNXINFRA",     "Nifty Infra"),
        (".BO",  "^CNXINFRA",     "Nifty Infra"),
        ("",     "XLI",           "S&P Industrials ETF"),
    ],
    "real estate": [
        (".NS",  "^CNXREALTY",    "Nifty Realty"),
        (".BO",  "^CNXREALTY",    "Nifty Realty"),
        ("",     "XLRE",          "S&P Real Estate ETF"),
    ],
    "communication services": [
        (".NS",  "^CNXMEDIA",     "Nifty Media"),
        (".BO",  "^CNXMEDIA",     "Nifty Media"),
        ("",     "XLC",           "S&P Comm. ETF"),
    ],
    "media": [
        (".NS",  "^CNXMEDIA",     "Nifty Media"),
        (".BO",  "^CNXMEDIA",     "Nifty Media"),
        ("",     "XLC",           "S&P Comm. ETF"),
    ],
}

# Keyword → canonical sector key (handles yfinance sector/industry variations)
_KEYWORD_MAP: dict[str, str] = {
    "drug": "pharma",
    "pharmaceutical": "pharma",
    "biotech": "healthcare",
    "health": "healthcare",
    "medical": "healthcare",
    "hospital": "healthcare",
    "software": "technology",
    "tech": "technology",
    "semiconductor": "technology",
    "hardware": "technology",
    "bank": "banking",
    "insurance": "financial services",
    "finance": "financial services",
    "nbfc": "financial services",
    "fmcg": "consumer defensive",
    "food": "consumer defensive",
    "beverage": "consumer defensive",
    "tobacco": "consumer defensive",
    "household": "consumer defensive",
    "auto": "automobiles",
    "vehicle": "automobiles",
    "tyre": "automobiles",
    "steel": "metals",
    "metal": "metals",
    "mining": "basic materials",
    "cement": "basic materials",
    "chemical": "basic materials",
    "oil": "energy",
    "gas": "energy",
    "petroleum": "energy",
    "power": "utilities",
    "electric": "utilities",
    "utility": "utilities",
    "construction": "industrials",
    "infrastructure": "industrials",
    "engineering": "industrials",
    "defence": "industrials",
    "defense": "industrials",
    "aerospace": "industrials",
    "realty": "real estate",
    "property": "real estate",
    "telecom": "communication services",
    "media": "media",
    "entertainment": "media",
}


def _detect_suffix(ticker: str) -> str:
    """Return the exchange suffix of the ticker (e.g. '.NS', '.BO', '')."""
    upper = ticker.upper()
    for sfx in (".NS", ".BO", ".L", ".DE", ".F", ".TO", ".AX", ".HK", ".SI", ".PA"):
        if upper.endswith(sfx.upper()):
            return sfx.lower()
    return ""


def _sector_key(sector: str, industry: str) -> str | None:
    """Map yfinance sector/industry strings to a key in _SECTOR_MAP."""
    combined = f"{sector} {industry}".lower()
    # Direct sector match first
    for key in _SECTOR_MAP:
        if key in combined:
            return key
    # Keyword fallback
    for kw, key in _KEYWORD_MAP.items():
        if kw in combined:
            return key
    return None


class SectorIndexAgent:
    """
    AI agent that identifies a stock's sector and returns the best sector index.

    Uses a fast keyword lookup first. When that fails, delegates to an LLM
    to reason about the correct sector mapping from the yfinance metadata.
    """

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, groq_client: Groq | None = None):
        self._client = groq_client

    def find(self, ticker: str) -> tuple[str, str] | tuple[None, None]:
        """
        Return (index_ticker, index_label) or (None, None) if not found.

        Lookup order:
          1. Same-exchange sector index
          2. Any-exchange sector index
          3. None
        """
        try:
            info     = yf.Ticker(ticker.upper()).info
            sector   = info.get("sector",   "") or ""
            industry = info.get("industry", "") or ""
        except Exception:
            return None, None

        if not sector and not industry:
            print(f"  ⚠️  [SectorIndexAgent] No sector metadata for {ticker}")
            return None, None

        key = _sector_key(sector, industry)

        # ── LLM fallback: ask the model when keyword lookup fails ─────────────
        if not key and self._client:
            key = self._llm_identify_sector(ticker, sector, industry)

        if not key:
            print(f"  ℹ️  [SectorIndexAgent] No sector index mapped for "
                  f"sector='{sector}' industry='{industry}'")
            return None, None

        suffix    = _detect_suffix(ticker)
        candidates = _SECTOR_MAP[key]

        # Pass 1: same exchange
        for sfx, idx_ticker, idx_label in candidates:
            if sfx == suffix:
                print(f"  🏭 [SectorIndexAgent] {sector} / {industry} → {idx_label} ({idx_ticker})")
                return idx_ticker, idx_label

        # Pass 2: any exchange
        for sfx, idx_ticker, idx_label in candidates:
            print(f"  🏭 [SectorIndexAgent] {sector} / {industry} → {idx_label} ({idx_ticker}) "
                  f"[cross-exchange]")
            return idx_ticker, idx_label

        return None, None

    def _llm_identify_sector(
        self, ticker: str, sector: str, industry: str
    ) -> str | None:
        """
        Ask the LLM to map an unrecognised sector/industry to a known sector key.

        The LLM reasons about which sector in _SECTOR_MAP best matches the
        yfinance metadata — it adapts to new industry strings yfinance may add
        without requiring a keyword-map update.
        """
        keys = list(_SECTOR_MAP.keys())
        prompt = (
            f"Stock: {ticker}\n"
            f"Yahoo Finance sector: '{sector}'\n"
            f"Yahoo Finance industry: '{industry}'\n\n"
            f"Available sector keys: {keys}\n\n"
            f"Which ONE key from the list best describes this stock's industry? "
            f"Reply with ONLY the exact key string (e.g. 'healthcare'), "
            f"or 'none' if none of the keys apply."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content":
                     "You map stock sector/industry metadata to a standardised sector key. "
                     "Reply with exactly ONE key from the provided list, or 'none'."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip().lower()
            # Strip punctuation / quotes the model might add
            answer = re.sub(r"[\"'\s]", "", answer)
            if answer and answer != "none" and answer in _SECTOR_MAP:
                print(f"  🤖 [SectorIndexAgent] LLM mapped '{sector}/{industry}' → '{answer}'")
                return answer
            print(f"  ℹ️  [SectorIndexAgent] LLM could not map '{sector}/{industry}' to a sector key")
        except Exception as exc:
            print(f"  ⚠️  [SectorIndexAgent] LLM fallback failed: {exc}")
        return None
