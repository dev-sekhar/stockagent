"""
CurveSpec
---------
Standard data contract passed between chart tools and ChartCompiler.

Every data-fetch tool (StockCurveTool, IndexCurveTool, SpreadCurveTool) returns
a CurveSpec.  ChartCompiler receives a list of them, aligns dates, and renders.

Roles
-----
  stock      : stock close price + volumes (one per ticker being charted)
  benchmark  : broad-market exchange index  (Nifty 50, S&P 500 …)
  sector     : industry-specific index      (Nifty Pharma, XLV …)
  spread     : % price difference between two stocks (arbitrage panel)
"""
from dataclasses import dataclass, field


@dataclass
class CurveSpec:
    """
    Standardised output from any data-fetch tool.

    Immutable contract — changes here affect ALL tools and ChartCompiler.
    Dates are always ISO-8601 strings (tz-naive). Values are raw (not rebased).
    """
    name:    str                    # Legend label shown in the chart
    role:    str                    # 'stock' | 'benchmark' | 'sector' | 'spread'
    ticker:  str                    # Source ticker (for traceability / governance)
    dates:   list[str]              # ISO-8601 date strings, tz-naive, normalised
    values:  list[float]            # Close prices OR spread percentages
    volumes: list[float] | None = None   # Only populated for 'stock' role

    # Styling hints — compiler may override for palette consistency
    color:   str | None = None      # Hex colour suggestion (None = auto-assign)

    # Placement in multi-panel subplot
    row: int = 1

    # Graceful failure: non-None means this fetch failed; compiler skips silently
    error:   str | None = None

    # Arbitrary metadata for the compiler (pct_change, high/low dates …)
    meta:    dict = field(default_factory=dict)
