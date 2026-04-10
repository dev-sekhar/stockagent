"""
HistoricalPriceAgent
--------------------
Single purpose: fetch OHLCV data for a relative period (1mo, 1y, ytd …).
"""
from agents.base_agent import BaseAgent
from tools.history_tools import HISTORICAL_TOOL_DEFINITION, get_historical_prices


class HistoricalPriceAgent(BaseAgent):
    system_prompt = (
        "Role: Summarise historical OHLCV data for a stock.\n"
        "Tool: call get_historical_prices(ticker, period, interval).\n"
        "Output: markdown report with:\n"
        "  • Opening/closing prices for the period\n"
        "  • Period high/low with dates\n"
        "  • Overall % change and trend direction\n"
        "  • Average volume\n"
        "  • Last 5 days closing prices\n"
        "  • Full data table if provided by the tool\n"
        "Constraint: Report only what the tool returns. Max 200 words."
    )
    tools = [HISTORICAL_TOOL_DEFINITION]
    tool_fn_map = {"get_historical_prices": get_historical_prices}

    def fetch(self, ticker: str, period: str = "1mo", interval: str = "1d") -> str:
        """Return a formatted OHLCV table + trend summary for the given period."""
        return self.run(
            f"Get historical prices for {ticker}, period={period}, interval={interval}"
        )
