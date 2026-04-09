"""
CurrentPriceAgent
-----------------
Single purpose: fetch and summarise the last traded price for a ticker symbol.
"""
from agents.base_agent import BaseAgent
from tools.price_tools import TOOL_DEFINITION, get_current_price


class CurrentPriceAgent(BaseAgent):
    system_prompt = (
        "You report the current stock price. "
        "Given a ticker, call get_current_price and present the result clearly: "
        "show the price with currency, the day's change and change%, volume, "
        "and the trade date. Keep it concise. "
        "IMPORTANT: if the tool returns an 'error' field, report that error honestly — "
        "never invent, estimate, or guess a price value."
    )
    tools = [TOOL_DEFINITION]
    tool_fn_map = {"get_current_price": get_current_price}

    def fetch(self, ticker: str) -> str:
        """Return a formatted current-price summary for the given ticker."""
        return self.run(f"Get the current price for: {ticker}")
