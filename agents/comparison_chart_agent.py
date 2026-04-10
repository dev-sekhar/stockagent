"""
ComparisonChartAgent
--------------------
AI orchestrator: coordinates data-fetch tools to compare multiple stocks.

Pipeline
--------
  For each stock ticker:
    StockCurveTool → fetches that stock's OHLCV → CurveSpec(role='stock')

  For each UNIQUE exchange represented (deduplicated):
    IndexCurveTool → fetches that exchange's benchmark → CurveSpec(role='benchmark')

  In arbitrage mode (same base ticker on ≥2 exchanges):
    SpreadCurveTool → computes % spread between the two primary listings

  ChartCompiler → aligns dates, rebases to 100 (or uses actual prices for
                   arbitrage), renders Plotly HTML with crosshair

Sector indices are NOT plotted in comparison mode — too cluttered.
This class owns NO data-fetching or rendering logic.
"""
from tools.stock_curve          import StockCurveTool
from tools.index_curve          import IndexCurveTool
from tools.spread_curve         import SpreadCurveTool
from tools.chart_compiler       import ChartCompiler
from config import DEFAULT_BENCHMARK, DEFAULT_BENCHMARK_NAME

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
_DEFAULT_INDEX = (DEFAULT_BENCHMARK, DEFAULT_BENCHMARK_NAME)


def _exchange_index(ticker: str) -> tuple[str, str]:
    upper = ticker.upper()
    for sfx, info in _EXCHANGE_INDEX.items():
        if upper.endswith(sfx.upper()):
            return info
    return _DEFAULT_INDEX


class ComparisonChartAgent:
    """
    Orchestrates curve agents for multi-stock comparison.
    """

    def __init__(self):
        self.stock_tool   = StockCurveTool()
        self.index_tool   = IndexCurveTool()
        self.spread_tool  = SpreadCurveTool()
        self.compiler     = ChartCompiler()

    def plot(
        self,
        tickers:   list[str],
        period:    str = "1mo",
        interval:  str = "1d",
        start:     str | None = None,
        end:       str | None = None,
        arbitrage: bool = False,
    ) -> str:
        """
        Fetch all curve data and render a comparison HTML chart.

        Parameters
        ----------
        tickers   : verified ticker symbols (≥ 2)
        arbitrage : True → actual prices + spread panel; False → rebased + volume
        """
        label = f"{start} → {end}" if (start and end) else period
        kw    = dict(period=period, interval=interval, start=start, end=end)

        specs = []

        print(f"  📊 [ComparisonChartAgent] Fetching {len(tickers)} stock(s)…")

        # ── Curve agents: one per stock ────────────────────────────────────────
        stock_specs = []
        for ticker in tickers:
            spec = self.stock_tool.run(ticker, **kw)
            specs.append(spec)
            if not spec.error:
                stock_specs.append(spec)

        if not stock_specs:
            return "❌  No data returned for any of the requested tickers."

        # ── Tools: one index per unique exchange (skip in arbitrage mode) ──────
        if not arbitrage:
            seen_indices: dict[str, str] = {}
            for ticker in tickers:
                it, iname = _exchange_index(ticker)
                if it not in seen_indices:
                    seen_indices[it] = iname

            for it, iname in seen_indices.items():
                spec = self.index_tool.run(it, iname, role="benchmark", **kw)
                specs.append(spec)

        # ── Tool: spread computation for arbitrage view ────────────────────────
        if arbitrage and len(stock_specs) >= 2:
            spread_spec = self.spread_tool.run(stock_specs[0], stock_specs[1])
            specs.append(spread_spec)

        # ── Compile all curves into chart ──────────────────────────────────────
        active_names = [s.name for s in stock_specs]
        mode_str     = "Arbitrage" if arbitrage else "Comparison (rebased to 100)"
        title        = f"{' vs '.join(active_names)}  ·  {label}  ·  {mode_str}"

        return self.compiler.compile(specs, title=title, arbitrage=arbitrage)
