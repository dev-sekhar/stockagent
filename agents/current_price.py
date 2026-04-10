"""
CurrentPriceAgent
-----------------
Single purpose: fetch and summarise the last traded price for a ticker symbol.
"""
from agents.base_agent import BaseAgent
from tools.price_tools import TOOL_DEFINITION, get_current_price


class CurrentPriceAgent(BaseAgent):
    system_prompt = (
        "Role: Fetch and report the current stock price.\n"
        "Tool: call get_current_price.\n"
        "Output format:\n"
        "  Price: [value] [currency]\n"
        "  Change: [±value] ([±%])\n"
        "  Volume: [volume]\n"
        "  Trade Date: [YYYY-MM-DD]\n"
        "Constraint: Never invent or estimate prices. "
        "If the tool returns an 'error' field, report it verbatim."
    )
    tools = [TOOL_DEFINITION]
    tool_fn_map = {"get_current_price": get_current_price}

    def fetch(self, ticker: str) -> str:
        """Return a formatted current-price summary for the given ticker."""
        return self.run(f"Get the current price for: {ticker}")
