"""
ExchangeDisambiguatorAgent
--------------------------
AI Agent: given candidate stock listings, resolve which one the user means.

The agent uses LLM reasoning — not hardcoded Python menus — to decide:

  • If one listing is clearly the right answer → return it automatically.
  • If multiple distinct companies match → call ask_user so the user picks.
  • If one company but listed on multiple exchanges → call ask_user so the
    user picks the exchange.

The LLM drives ALL disambiguation.  Interactive prompts are issued via the
ask_user tool, which the LLM calls only when genuine ambiguity exists.
No hardcoded if-len>1 branching or input() menus anywhere in this file.
"""
import re
from groq import Groq

from agents.base_agent import BaseAgent
from tools.ask_user_tool import ask_user, TOOL_DEFINITION as ASK_USER_TOOL
from tools.ticker_tools import search_ticker


_SYSTEM = """\
Role: Resolve one verified ticker from a list of candidate stock listings.
Tool: call ask_user when genuine ambiguity exists.

Rules:
1. Single listing → return ticker immediately. No tool call.
2. One company, multiple exchanges → ask_user to choose exchange.
3. Multiple distinct companies → ask_user to choose company, then exchange if needed.
4. Fully resolved → output ONLY the ticker symbol (e.g. INFY.NS). No prose.

Constraints:
- Well-known company with one clear primary listing → auto-select.
- Multiple exchanges for the same company → ALWAYS ask.
- Multiple companies that could match → ALWAYS ask.
"""


def _normalise_name(name: str) -> str:
    """Lower-case, strip legal suffixes, ADR/listing tags, and parenthetical
    qualifiers for fuzzy company-identity comparison."""
    name = name.lower()
    # Strip anything in parentheses — always a qualifier, never core identity:
    # "(Old)", "(GDR)", "(ADR)", "(NS)", "(Formerly XYZ)" etc.
    name = re.sub(r"\(.*?\)", "", name)
    for suffix in (
        "limited", "ltd", "inc", "corp", "plc", "llc", "n.v.", "s.a.", "ag",
        # ADR / depositary-receipt listing tags
        "adr", "adrc", "ads", "sponsored adr", "unsponsored adr",
        # Historical naming variants (e.g. Infosys → Infosys Technologies)
        "technologies", "technology",
    ):
        name = re.sub(rf"\b{re.escape(suffix)}\b\.?", "", name)
    return re.sub(r"\s+", " ", name).strip()


class ExchangeDisambiguatorAgent(BaseAgent):
    """
    AI agent that resolves the intended stock listing from a list of candidates.

    Inherits the BaseAgent tool-calling loop.  The only tool available to the
    LLM is ask_user, which it calls whenever it cannot auto-resolve.
    """

    system_prompt = _SYSTEM
    tools         = [ASK_USER_TOOL]
    tool_fn_map   = {"ask_user": ask_user}

    def __init__(self, groq_client: Groq | None = None):
        # BaseAgent requires a client; store optionally for None-safe resolution
        self._has_client = groq_client is not None
        if groq_client:
            super().__init__(groq_client)

    # ── public API ────────────────────────────────────────────────────────────

    def resolve(self, company_name: str) -> str:
        """Discover candidates via search_ticker, then disambiguate."""
        result = search_ticker(company_name)
        if "error" in result:
            raise ValueError(result["error"])
        candidates = result.get("candidates", [])
        if not candidates:
            raise ValueError(f"No listings found for '{company_name}'")
        return self.resolve_from_candidates(company_name, candidates)

    def resolve_from_candidates(self, query: str, candidates: list) -> str:
        """
        Main entry point.  Groups candidates by company + exchange, formats
        them for the LLM, then runs the tool-calling loop.  The LLM calls
        ask_user whenever it cannot auto-resolve.
        """
        if not candidates:
            raise ValueError(f"No listings found for '{query}'")

        company_groups = self._group_by_company(candidates)
        message        = self._format_for_llm(query, company_groups)

        print(f"  🤖 [ExchangeDisambiguatorAgent] Resolving '{query}'…")

        if not self._has_client:
            # No LLM available — return first candidate as best effort
            first = list(company_groups.values())[0][0]
            print(f"  ✅ [ExchangeDisambiguatorAgent] No LLM — defaulting to {first['ticker']}")
            return first["ticker"]

        response = self.run(message)
        ticker   = self._extract_ticker(response, candidates)
        if ticker:
            print(f"  ✅ [ExchangeDisambiguatorAgent] Resolved → {ticker}")
            return ticker

        raise ValueError(
            f"Could not extract a valid ticker from LLM response for '{query}'. "
            f"Response was: {response!r}"
        )

    # ── formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_for_llm(query: str, company_groups: dict) -> str:
        """Render candidate groups as a readable prompt for the LLM."""
        lines = [f"User query: '{query}'\n", "Matching stock listings:\n"]
        for group in company_groups.values():
            by_exchange   = ExchangeDisambiguatorAgent._best_per_exchange(group)
            company_label = group[0]["name"]
            if len(by_exchange) == 1:
                c   = by_exchange[0]
                adr = "  [ADR]" if c.get("is_adr") else ""
                lines.append(f"  Company: {company_label}")
                lines.append(f"    {c['ticker']:<18} {c['exchange']:<25}{adr}")
            else:
                lines.append(
                    f"  Company: {company_label}  ({len(by_exchange)} exchanges)"
                )
                for c in by_exchange:
                    adr = "  [ADR]" if c.get("is_adr") else ""
                    lines.append(
                        f"    {c['ticker']:<18} {c['exchange']:<25}{adr}"
                    )
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _extract_ticker(response: str, candidates: list) -> str | None:
        """Parse the LLM's final text response to extract the ticker symbol."""
        known = {c["ticker"].upper() for c in candidates}
        clean = response.strip().upper()
        # Exact match — LLM output just the ticker
        if clean in known:
            return clean
        # Partial match — find first ticker-shaped word in the response
        for m in re.finditer(r"\b([A-Z]{1,6}(?:\.[A-Z]{1,2})?)\b", clean):
            if m.group(1) in known:
                return m.group(1)
        return None

    # ── grouping helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _group_by_company(candidates: list) -> dict:
        """
        Returns {normalised_name: [candidates]} preserving insertion order.
        Within each group, equities are promoted to the front.
        """
        groups: dict = {}
        for c in candidates:
            key = _normalise_name(c["name"])
            groups.setdefault(key, []).append(c)
        for key in groups:
            groups[key].sort(
                key=lambda c: 0 if c.get("type", "").upper() == "EQUITY" else 1
            )
        return groups

    @staticmethod
    def _best_per_exchange(candidates: list) -> list:
        """One entry per exchange; prefer EQUITY within each exchange."""
        by_exchange: dict = {}
        for c in candidates:
            by_exchange.setdefault(c["exchange"], []).append(c)
        result = []
        for group in by_exchange.values():
            equities = [c for c in group if c.get("type", "").upper() == "EQUITY"]
            result.append(equities[0] if equities else group[0])
        return result

    @staticmethod
    def _find_group_for_ticker(ticker: str, company_groups: dict) -> list | None:
        """Return the candidate list whose group contains *ticker*, or None."""
        for group in company_groups.values():
            if any(c["ticker"].upper() == ticker.upper() for c in group):
                return group
        return None

