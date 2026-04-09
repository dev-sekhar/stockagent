"""
IndexCurveTool
--------------
Data-fetch tool: retrieve price data for ONE index/ETF ticker via yfinance.

This is a TOOL, not an AI agent — it performs a deterministic fetch with no
reasoning.  Used for both broad-market benchmarks (role='benchmark') and
sector indices (role='sector').  The caller decides which role to assign.

Does NOT rebase, does NOT create Plotly traces — those are the compiler's job.
"""
import yfinance as yf
from tools.curve_spec import CurveSpec


class IndexCurveTool:
    """Fetches price data for an index or ETF ticker."""

    def run(
        self,
        index_ticker: str,
        index_name:   str,
        role:         str = "benchmark",   # 'benchmark' | 'sector'
        period:       str = "1mo",
        interval:     str = "1d",
        start:        str | None = None,
        end:          str | None = None,
    ) -> CurveSpec:
        """
        Fetch OHLCV for index_ticker and return a CurveSpec with the given role.

        Parameters
        ----------
        index_ticker : Yahoo Finance symbol (e.g. '^NSEI', 'XLV')
        index_name   : Human-readable label (e.g. 'Nifty 50')
        role         : 'benchmark' (exchange-wide) or 'sector' (industry-specific)
        """
        label = f"{start}→{end}" if (start and end) else period
        print(f"  📊 [IndexCurveTool] Fetching {role}: {index_name} ({index_ticker}) {label}…")
        try:
            t  = yf.Ticker(index_ticker)
            df = (t.history(period=period, interval=interval)
                  if not (start and end)
                  else t.history(start=start, end=end, interval=interval))

            if df.empty:
                print(f"  ⚠️  [IndexCurveTool] No data for {index_ticker}")
                return CurveSpec(
                    name=index_name, role=role, ticker=index_ticker,
                    dates=[], values=[],
                    error=f"No data for index {index_ticker}",
                )

            # Timezone-strip + date-normalise
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            df.index = idx.normalize()

            dates  = [d.strftime("%Y-%m-%d") for d in df.index]
            closes = df["Close"].values.tolist()
            c0, c1  = closes[0], closes[-1]
            pct_chg = (c1 - c0) / c0 * 100 if c0 else 0

            print(f"  ✅ [IndexCurveTool] {index_name} — {len(dates)} bars  "
                  f"({'▲' if pct_chg >= 0 else '▼'}{abs(pct_chg):.2f}%)")

            return CurveSpec(
                name=index_name, role=role, ticker=index_ticker,
                dates=dates, values=closes,
                meta={"pct_chg": pct_chg},
            )
        except Exception as exc:
            print(f"  ❌ [IndexCurveTool] {index_name} ({index_ticker}) failed: {exc}")
            return CurveSpec(
                name=index_name, role=role, ticker=index_ticker,
                dates=[], values=[], error=str(exc),
            )
