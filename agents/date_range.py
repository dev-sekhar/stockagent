"""
DateRangeAgent
--------------
Single purpose: fetch OHLCV data between two explicit calendar dates.
"""
from agents.base_agent import BaseAgent
from tools.history_tools import DATE_RANGE_TOOL_DEFINITION, get_price_range_between_dates


class DateRangeAgent(BaseAgent):
    system_prompt = (
        "You fetch stock price data between two specific dates. "
        "Call get_price_range_between_dates with the supplied ticker, start, and end dates. "
        "Always include the full OHLCV table, followed by a brief summary."
    )
    tools = [DATE_RANGE_TOOL_DEFINITION]
    tool_fn_map = {"get_price_range_between_dates": get_price_range_between_dates}

    def fetch(self, ticker: str, start: str, end: str) -> str:
        """Return a formatted OHLCV table + summary for a specific date range."""
        return self.run(f"Get prices for {ticker} from {start} to {end}")
