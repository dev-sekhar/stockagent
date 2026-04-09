"""
TickerResolverAgent
-------------------
AI Agent: resolves an ambiguous company name or partial ticker to a verified
Yahoo Finance symbol using LLM-powered reasoning.

This is a genuine AI agent — it uses an LLM with the search_ticker tool to:
  • Understand typos and common name variations ("appl" → "AAPL")
  • Rank multiple candidates by likelihood given the user's phrasing
  • Return the single best ticker with a confidence note

Falls back to the orchestrator's full resolution pipeline for complex cases.
"""
from agents.base_agent import BaseAgent
from tools.ticker_tools import TOOL_DEFINITION, search_ticker, verify_ticker


class TickerResolverAgent(BaseAgent):
    system_prompt = (
        "You are a stock ticker resolution specialist. "
        "Given a company name, abbreviation, or ticker symbol, call search_ticker "
        "to find matching candidates. "
        "Then identify the SINGLE most likely ticker the user meant based on:\n"
        "  • Exact or near-exact name match\n"
        "  • The most liquid / well-known listing for that company\n"
        "  • Preference for primary exchange over ADRs\n"
        "Respond concisely: 'Best match: TICKER (Company Name, Exchange)'. "
        "If truly ambiguous, list the top 3 and explain the ambiguity."
    )
    tools = [TOOL_DEFINITION]
    tool_fn_map = {"search_ticker": search_ticker}

    def resolve(self, company_or_ticker: str) -> str:
        """
        Use LLM reasoning to identify the best ticker for the given query.
        Returns the LLM's natural-language response (may name multiple candidates).
        """
        return self.run(f"Find the best stock ticker for: '{company_or_ticker}'")

    def get_candidates(self, company_name: str) -> dict:
        """Direct tool call — returns raw candidates without LLM reasoning."""
        return search_ticker(company_name)

