"""
IndustryIndexAgent
------------------
Single purpose: given a stock ticker, identify its sector and return the
appropriate sector/industry index ticker for overlaying on a price chart.

Uses yfinance's info dict (no LLM needed — yfinance returns structured sector
data directly). Falls back gracefully to None if no sector index is found.

Priority:
  1. Exchange-specific sector index (e.g. HINDUNILVR.NS → Nifty FMCG)
  2. Any-exchange sector index if exchange-specific not available
  3. None — caller should skip the sector overlay

Supported exchanges:
  Indian (.NS / .BO) → NSE/BSE Nifty sector indices
  US (no suffix)     → SPDR S&P sector ETFs (XLV, XLK, XLF …)
  Other exchanges    → US sector ETFs as proxy
"""
import yfinance as yf

# ── Indian (NSE/BSE) sector indices ──────────────────────────────────────────
_NSE_SECTOR_INDEX: dict[str, tuple[str, str]] = {
    "Healthcare":              ("^CNXPHARMA",  "Nifty Pharma"),
    "Technology":              ("^CNXIT",       "Nifty IT"),
    "Financial Services":      ("^NSEBANK",     "Nifty Bank"),
    "Consumer Defensive":      ("^CNXFMCG",     "Nifty FMCG"),
    "Basic Materials":         ("^CNXMETAL",    "Nifty Metal"),
    "Consumer Cyclical":       ("^CNXAUTO",     "Nifty Auto"),
    "Energy":                  ("^CNXENERGY",   "Nifty Energy"),
    "Real Estate":             ("^CNXREALTY",   "Nifty Realty"),
    "Industrials":             ("^CNXINFRA",    "Nifty Infra"),
    "Communication Services":  ("^CNXMEDIA",    "Nifty Media"),
    "Utilities":               ("^CNXINFRA",    "Nifty Infra"),
}

# ── US / global sector ETFs (SPDR) ───────────────────────────────────────────
_US_SECTOR_INDEX: dict[str, tuple[str, str]] = {
    "Healthcare":              ("XLV",  "S&P Healthcare"),
    "Technology":              ("XLK",  "S&P Technology"),
    "Financial Services":      ("XLF",  "S&P Financials"),
    "Energy":                  ("XLE",  "S&P Energy"),
    "Basic Materials":         ("XLB",  "S&P Materials"),
    "Industrials":             ("XLI",  "S&P Industrials"),
    "Consumer Defensive":      ("XLP",  "S&P Consumer Staples"),
    "Consumer Cyclical":       ("XLY",  "S&P Consumer Discret."),
    "Utilities":               ("XLU",  "S&P Utilities"),
    "Real Estate":             ("XLRE", "S&P Real Estate"),
    "Communication Services":  ("XLC",  "S&P Communication"),
}

_INDIAN_SUFFIXES = (".NS", ".BO")


class IndustryIndexAgent:
    """
    Identifies a stock's sector via yfinance and maps it to a sector index.
    No LLM — pure yfinance metadata lookup.
    """

    def get_sector_index(self, ticker: str) -> tuple[str, str] | None:
        """
        Return (sector_index_ticker, sector_index_name) or None.

        Tries the exchange-native sector index first (e.g. Nifty Pharma for .NS
        stocks); falls back to US sector ETFs as a global proxy if not found.
        """
        try:
            info   = yf.Ticker(ticker).fast_info   # fast_info is lighter than full info
            sector = getattr(info, "sector", None)

            # fast_info may not have sector — fall back to full info if needed
            if not sector:
                full   = yf.Ticker(ticker).info
                sector = full.get("sector", "")

            if not sector:
                print(f"  ℹ️  [IndustryIndexAgent] No sector data for {ticker}")
                return None

            print(f"  🏭 [IndustryIndexAgent] {ticker} → sector: {sector}")

            is_indian = ticker.upper().endswith(_INDIAN_SUFFIXES)

            if is_indian:
                result = _NSE_SECTOR_INDEX.get(sector)
                if result:
                    return result
                # Fall back to US proxy if no Indian sector index mapped
                result = _US_SECTOR_INDEX.get(sector)
                if result:
                    print(f"  ℹ️  [IndustryIndexAgent] No Indian sector index for '{sector}' "
                          f"— using {result[1]} as proxy")
                    return result
            else:
                result = _US_SECTOR_INDEX.get(sector)
                if result:
                    return result

            print(f"  ℹ️  [IndustryIndexAgent] No sector index mapped for '{sector}'")
            return None

        except Exception as exc:
            print(f"  ⚠️  [IndustryIndexAgent] Lookup failed ({exc})")
            return None
