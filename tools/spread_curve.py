"""
SpreadCurveTool
---------------
Computation tool: calculate the % price spread between two CurveSpec instances.

This is a TOOL, not an AI agent — it performs a deterministic mathematical
computation with no reasoning.  Used in arbitrage mode to quantify how much
one cross-listed stock diverges from another on a day-by-day basis.

Input:  two CurveSpec objects (typically role='stock')
Output: one CurveSpec with role='spread'
        values[i] = (price_a[i] / price_b[i] - 1) × 100
        positive  = spec_a is MORE expensive than spec_b on that date
"""
import pandas as pd
from tools.curve_spec import CurveSpec


class SpreadCurveTool:
    """Computes the % price spread between two price series."""

    def run(self, spec_a: CurveSpec, spec_b: CurveSpec) -> CurveSpec:
        """
        Return a CurveSpec containing the daily % spread.

        Dates are aligned by intersection. Returns an error CurveSpec
        if either input is empty or has no overlapping dates.
        """
        name = f"Spread  {spec_a.ticker} vs {spec_b.ticker}"
        print(f"  🔀 [SpreadCurveTool] Computing {name}…")

        if spec_a.error or not spec_a.dates:
            return CurveSpec(name=name, role="spread", ticker="",
                             dates=[], values=[],
                             error=f"{spec_a.ticker} has no data for spread")
        if spec_b.error or not spec_b.dates:
            return CurveSpec(name=name, role="spread", ticker="",
                             dates=[], values=[],
                             error=f"{spec_b.ticker} has no data for spread")
        try:
            idx_a = pd.DatetimeIndex([pd.Timestamp(d) for d in spec_a.dates])
            idx_b = pd.DatetimeIndex([pd.Timestamp(d) for d in spec_b.dates])
            s_a   = pd.Series(spec_a.values, index=idx_a)
            s_b   = pd.Series(spec_b.values, index=idx_b)

            common = idx_a.intersection(idx_b)
            if len(common) == 0:
                return CurveSpec(name=name, role="spread", ticker="",
                                 dates=[], values=[],
                                 error="No overlapping dates between the two series")

            spread  = (s_a[common] / s_b[common] - 1) * 100
            dates   = [d.strftime("%Y-%m-%d") for d in spread.index]
            values  = [round(v, 4) for v in spread.values.tolist()]
            max_abs = max(abs(v) for v in values)

            print(f"  ✅ [SpreadCurveTool] {len(dates)} data points  "
                  f"max spread: {max_abs:.3f}%")

            return CurveSpec(
                name=name, role="spread", ticker="",
                dates=dates, values=values, row=2,
                meta={
                    "max_spread":  max_abs,
                    "mean_spread": sum(values) / len(values),
                    "ticker_a":    spec_a.ticker,
                    "ticker_b":    spec_b.ticker,
                },
            )
        except Exception as exc:
            print(f"  ❌ [SpreadCurveTool] Failed: {exc}")
            return CurveSpec(name=name, role="spread", ticker="",
                             dates=[], values=[], error=str(exc))
