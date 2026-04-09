"""
company_data.py
---------------
CompanyDataTool — pure yfinance data-fetch tool.

Fetches three data groups for a given ticker:
  profile      — identity  (name, sector, industry, country, exchange, description)
  fundamentals — valuation (market cap, P/E, EPS, 52-week range, beta, dividend)
  financials   — health    (revenue, margins, debt, cash flow, growth rates)

No LLM is used.  NaN/Inf values are cleaned to None for JSON safety.
"""

import math
from typing import Any, Dict, Optional

import yfinance as yf


# ── NaN / Inf guard ───────────────────────────────────────────────────────────

def _clean(val: Any) -> Any:
    """Return None for NaN / Inf / None; otherwise return val unchanged."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _pct(val: Any) -> Optional[float]:
    """Convert a 0-1 fraction to a rounded percentage, or None."""
    v = _clean(val)
    return round(v * 100, 2) if v is not None else None


# ── Tool ──────────────────────────────────────────────────────────────────────

class CompanyDataTool:
    """
    Fetches structured company data from yfinance.

    Usage
    -----
      data = CompanyDataTool().fetch("INFY.NS")
      # data["profile"]["name"]          → "Infosys Limited"
      # data["fundamentals"]["pe_trailing"] → 24.3
      # data["financials"]["net_margin"]    → 17.2  (%)
    """

    def fetch(self, ticker: str) -> Dict[str, Any]:
        """
        Return a cleaned dict with profile / fundamentals / financials.
        On failure, returns {"ticker": ..., "error": "..."}.
        """
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

        if not info.get("longName") and not info.get("shortName"):
            return {
                "ticker": ticker,
                "error":  f"No company data available for {ticker}",
            }

        profile = self._extract_profile(info)
        fundamentals = self._extract_fundamentals(info)
        financials   = self._extract_financials(info)

        return {
            "ticker":       ticker,
            "profile":      profile,
            "fundamentals": fundamentals,
            "financials":   financials,
            "error":        None,
        }

    # ── Profile ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_profile(info: dict) -> dict:
        desc = info.get("longBusinessSummary") or ""
        return {
            "name":        info.get("longName") or info.get("shortName", ""),
            "sector":      info.get("sector"),
            "industry":    info.get("industry"),
            "country":     info.get("country"),
            "exchange":    info.get("exchange"),
            "website":     info.get("website"),
            "city":        info.get("city"),
            "description": desc[:500] + ("…" if len(desc) > 500 else ""),
            "employees":   info.get("fullTimeEmployees"),
        }

    # ── Fundamentals ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_fundamentals(info: dict) -> dict:
        div_yield = _clean(info.get("dividendYield"))
        return {
            "currency":           info.get("currency", ""),
            "market_cap":         _clean(info.get("marketCap")),
            "enterprise_value":   _clean(info.get("enterpriseValue")),
            "pe_trailing":        _clean(info.get("trailingPE")),
            "pe_forward":         _clean(info.get("forwardPE")),
            "pb":                 _clean(info.get("priceToBook")),
            "ps":                 _clean(info.get("priceToSalesTrailing12Months")),
            "eps_trailing":       _clean(info.get("trailingEps")),
            "eps_forward":        _clean(info.get("forwardEps")),
            "dividend_yield_pct": round(div_yield * 100, 2) if div_yield else None,
            "dividend_rate":      _clean(info.get("dividendRate")),
            "ex_dividend_date":   str(info.get("exDividendDate") or ""),
            "beta":               _clean(info.get("beta")),
            "week_52_high":       _clean(info.get("fiftyTwoWeekHigh")),
            "week_52_low":        _clean(info.get("fiftyTwoWeekLow")),
            "avg_volume":         _clean(info.get("averageVolume")),
            "avg_volume_10d":     _clean(info.get("averageVolume10days")),
            "shares_outstanding": _clean(info.get("sharesOutstanding")),
            "float_shares":       _clean(info.get("floatShares")),
        }

    # ── Financials ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_financials(info: dict) -> dict:
        return {
            "revenue":          _clean(info.get("totalRevenue")),
            "gross_profit":     _clean(info.get("grossProfits")),
            "net_income":       _clean(info.get("netIncomeToCommon")),
            "gross_margin_pct": _pct(info.get("grossMargins")),
            "operating_margin_pct": _pct(info.get("operatingMargins")),
            "net_margin_pct":   _pct(info.get("profitMargins")),
            "roe_pct":          _pct(info.get("returnOnEquity")),
            "roa_pct":          _pct(info.get("returnOnAssets")),
            "total_debt":       _clean(info.get("totalDebt")),
            "total_cash":       _clean(info.get("totalCash")),
            "debt_to_equity":   _clean(info.get("debtToEquity")),
            "current_ratio":    _clean(info.get("currentRatio")),
            "free_cash_flow":   _clean(info.get("freeCashflow")),
            "revenue_growth_pct":   _pct(info.get("revenueGrowth")),
            "earnings_growth_pct":  _pct(info.get("earningsGrowth")),
        }
