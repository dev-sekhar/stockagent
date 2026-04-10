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
        "Role: Resolve a company name, abbreviation, or ticker to the best stock ticker.\n"
        "Tool: call search_ticker.\n"
        "Output format:\n"
        "  Best match: TICKER (Company Name, Exchange)\n"
        "  Confidence: high|medium|low\n"
        "  Reason: one sentence\n"
        "Constraint: Prefer primary exchange over ADRs. "
        "If genuinely ambiguous, list top 3 with tickers and exchanges."
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

