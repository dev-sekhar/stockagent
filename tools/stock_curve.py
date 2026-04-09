"""
StockCurveTool
--------------
Data-fetch tool: retrieve OHLCV data for ONE stock ticker via yfinance.

This is a TOOL, not an AI agent — it performs a deterministic fetch with no
reasoning.  It is called BY agents (ChartAgent, ComparisonChartAgent) and
returns a CurveSpec for ChartCompiler to render.

Responsibilities
----------------
  • Fetch yfinance history for the given ticker + period/dates
  • Strip timezone from the DatetimeIndex (UTC-normalise)
  • Return dates, close prices, volumes, and meta stats in a CurveSpec

Does NOT rebase, does NOT create Plotly traces — those are the compiler's job.
"""
import yfinance as yf
from tools.curve_spec import CurveSpec


class StockCurveTool:
    """Fetches price + volume data for a single stock ticker."""

    def run(
        self,
        ticker:   str,
        period:   str = "1mo",
        interval: str = "1d",
        start:    str | None = None,
        end:      str | None = None,
    ) -> CurveSpec:
        """
        Fetch OHLCV and return a CurveSpec.

        Returns a CurveSpec with error set (and empty lists) on failure so
        ChartCompiler can continue with the remaining curves.
        """
        label = f"{start}→{end}" if (start and end) else period
        print(f"  📈 [StockCurveTool] Fetching {ticker.upper()} ({label})…")
        try:
            t  = yf.Ticker(ticker.upper())
            df = (t.history(period=period, interval=interval)
                  if not (start and end)
                  else t.history(start=start, end=end, interval=interval))

            if df.empty:
                return CurveSpec(
                    name=ticker.upper(), role="stock", ticker=ticker.upper(),
                    dates=[], values=[],
                    error=f"No data returned for {ticker}",
                )

            # Timezone-strip + date-normalise
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            df.index = idx.normalize()

            dates   = [d.strftime("%Y-%m-%d") for d in df.index]
            closes  = df["Close"].values.tolist()
            volumes = df["Volume"].values.tolist()
            highs   = df["High"].values.tolist()
            lows    = df["Low"].values.tolist()

            c0, c1  = closes[0], closes[-1]
            pct_chg = (c1 - c0) / c0 * 100 if c0 else 0

            high_val  = max(highs)
            low_val   = min(lows)
            high_date = df["High"].idxmax().strftime("%Y-%m-%d")
            low_date  = df["Low"].idxmin().strftime("%Y-%m-%d")

            print(f"  ✅ [StockCurveTool] {ticker.upper()} — {len(dates)} bars  "
                  f"({'▲' if pct_chg >= 0 else '▼'}{abs(pct_chg):.2f}%)")

            return CurveSpec(
                name=ticker.upper(), role="stock", ticker=ticker.upper(),
                dates=dates, values=closes, volumes=volumes,
                meta={
                    "pct_chg":   pct_chg,
                    "open":      closes[0],
                    "close":     closes[-1],
                    "high":      high_val,
                    "low":       low_val,
                    "high_date": high_date,
                    "low_date":  low_date,
                },
            )
        except Exception as exc:
            print(f"  ❌ [StockCurveTool] {ticker} failed: {exc}")
            return CurveSpec(
                name=ticker.upper(), role="stock", ticker=ticker.upper(),
                dates=[], values=[], error=str(exc),
            )
