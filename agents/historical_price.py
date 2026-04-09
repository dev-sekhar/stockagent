"""
HistoricalPriceAgent
--------------------
Single purpose: fetch OHLCV data for a relative period (1mo, 1y, ytd …).
"""
from agents.base_agent import BaseAgent
from tools.history_tools import HISTORICAL_TOOL_DEFINITION, get_historical_prices


class HistoricalPriceAgent(BaseAgent):
    system_prompt = (
        "You summarise historical stock price data from tool results. "
        "Call get_historical_prices with the supplied ticker, period, and interval. "
        "Present: opening price, closing price, period high/low (with dates), "
        "overall % change, trend direction, average volume, and the last few days' prices. "
        "If a full table is included show it; otherwise narrate from the summary statistics. "
        "Keep the response concise."
    )
    tools = [HISTORICAL_TOOL_DEFINITION]
    tool_fn_map = {"get_historical_prices": get_historical_prices}

    def fetch(self, ticker: str, period: str = "1mo", interval: str = "1d") -> str:
        """Return a formatted OHLCV table + trend summary for the given period."""
        return self.run(
            f"Get historical prices for {ticker}, period={period}, interval={interval}"
        )
