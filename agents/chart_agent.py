"""
ChartAgent
----------
AI orchestrator: coordinates data-fetch tools and ChartCompiler to produce
a single-stock interactive price chart with benchmark and sector overlays.

Pipeline
--------
  1. StockCurveTool    → fetches OHLCV for the requested ticker
  2. IndexCurveTool    → fetches the broad-market benchmark (exchange-based)
  3. SectorIndexAgent  → AI decision: which sector index to include (if any)
     IndexCurveTool    → fetches that sector index (if found & different from benchmark)
  4. ChartCompiler     → aligns dates, rebases to 100, renders Plotly HTML

Data flows through CurveSpec objects. This class owns NO data-fetching or
rendering logic — it reasons about WHAT to fetch and passes it to tools.
"""
import os
import webbrowser

from tools.stock_curve     import StockCurveTool
from tools.index_curve     import IndexCurveTool
from agents.sector_index   import SectorIndexAgent
from tools.chart_compiler  import ChartCompiler

# ── Exchange suffix → (broad-market index ticker, display name) ───────────────
_EXCHANGE_INDEX: dict[str, tuple[str, str]] = {
    ".NS":  ("^NSEI",   "Nifty 50"),
    ".BO":  ("^BSESN",  "Sensex"),
    ".L":   ("^FTSE",   "FTSE 100"),
    ".DE":  ("^GDAXI",  "DAX"),
    ".F":   ("^GDAXI",  "DAX"),
    ".TO":  ("^GSPTSE", "S&P/TSX"),
    ".AX":  ("^AXJO",   "ASX 200"),
    ".HK":  ("^HSI",    "Hang Seng"),
    ".SI":  ("^STI",    "STI"),
    ".PA":  ("^FCHI",   "CAC 40"),
    ".AS":  ("^AEX",    "AEX"),
    ".SW":  ("^SSMI",   "SMI"),
    ".T":   ("^N225",   "Nikkei 225"),
}
_DEFAULT_INDEX = ("^GSPC", "S&P 500")


def _detect_index(ticker: str) -> tuple[str, str]:
    upper = ticker.upper()
    for sfx, info in _EXCHANGE_INDEX.items():
        if upper.endswith(sfx.upper()):
            return info
    return _DEFAULT_INDEX


class ChartAgent:
    """
    AI orchestrator: decides which indices to include, then delegates data
    fetching to tools and rendering to ChartCompiler.
    """

    def __init__(self, groq_client=None):
        self.stock_tool    = StockCurveTool()
        self.index_tool    = IndexCurveTool()
        self.sector_finder = SectorIndexAgent(groq_client)
        self.compiler      = ChartCompiler()

    def plot(
        self,
        ticker:       str,
        period:       str = "1mo",
        interval:     str = "1d",
        start:        str | None = None,
        end:          str | None = None,
        company_data: dict | None = None,
        news:         list | None = None,
    ) -> str:
        """
        Fetch all curve data and render an interactive HTML chart.

        Optional company_data + news (from CompanyDataTool / NewsAgent)
        are forwarded to ChartCompiler, which injects a company-snapshot
        panel below the Plotly chart.
        """
        label = f"{start} → {end}" if (start and end) else period
        kw    = dict(period=period, interval=interval, start=start, end=end)

        specs = []

        # ── Tool: fetch stock price data ──────────────────────────────────────
        stock_spec = self.stock_tool.run(ticker, **kw)
        specs.append(stock_spec)

        # ── Tool: fetch broad-market benchmark ────────────────────────────────
        idx_ticker, idx_name = _detect_index(ticker)
        bench_spec = self.index_tool.run(
            idx_ticker, idx_name, role="benchmark", **kw
        )
        specs.append(bench_spec)

        # ── Agent decision: which sector index to include? ────────────────────
        sec_ticker, sec_name = self.sector_finder.find(ticker)
        if sec_ticker and sec_ticker != idx_ticker:
            sector_spec = self.index_tool.run(
                sec_ticker, sec_name, role="sector", **kw
            )
            specs.append(sector_spec)

        # ── Compile all curves into chart ──────────────────────────────────────
        pct_chg = stock_spec.meta.get("pct_chg", 0)
        sign    = "▲" if pct_chg >= 0 else "▼"
        title   = (f"{ticker.upper()}  vs  {idx_name}"
                   + (f"  +  {sec_name}" if sec_ticker and sec_ticker != idx_ticker else "")
                   + f"  ·  {label}  ·  ({sign}{abs(pct_chg):.1f}%)")

        return self.compiler.compile(
            specs, title=title, company_data=company_data, news=news
        )


