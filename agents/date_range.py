"""
DateRangeAgent
--------------
Single purpose: fetch OHLCV data between two explicit calendar dates.
"""
from agents.base_agent import BaseAgent
from tools.history_tools import DATE_RANGE_TOOL_DEFINITION, get_price_range_between_dates


class DateRangeAgent(BaseAgent):
    system_prompt = (
        "Role: Fetch and display OHLCV data between two dates.\n"
        "Tool: call get_price_range_between_dates(ticker, start, end).\n"
        "Output: markdown table (Date | Open | High | Low | Close | Volume) "
        "followed by a summary (max 100 words: % change, key price movements, notable dates).\n"
        "Constraint: Report only what the tool returns. Never estimate missing data."
    )
    tools = [DATE_RANGE_TOOL_DEFINITION]
    tool_fn_map = {"get_price_range_between_dates": get_price_range_between_dates}

    def fetch(self, ticker: str, start: str, end: str) -> str:
        """Return a formatted OHLCV table + summary for a specific date range."""
        return self.run(f"Get prices for {ticker} from {start} to {end}")
