"""
ExchangeDisambiguatorAgent
--------------------------
AI Agent: given candidate stock listings, resolve which one the user means
and (if multiple) ask for clarification.

This is a genuine AI agent — it uses an LLM to auto-select the most likely
listing when the query is unambiguous (e.g. "apple" → AAPL on NASDAQ) before
falling back to presenting an interactive menu.

Two distinct scenarios:

  A) SAME company on multiple exchanges (e.g. HDFC Bank on NSE + BSE + NYSE)
     → deduplicate to one row per exchange, try LLM auto-select first.

  B) DIFFERENT companies matching the query (e.g. "hindustan" → HUL, Zinc,
     Petroleum, Aeronautics …)
     → try LLM to identify clear winner; otherwise show menu.

Auto-resolves silently when there is exactly one candidate after deduplication.
"""
import re
import json
from groq import Groq
from tools.ticker_tools import search_ticker

_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_AUTOSELECT = """\
You are a stock ticker disambiguation expert.

Given a user query and a list of matching stock listings, decide whether
one listing is the OBVIOUS best match (confidence >= 80%) for the query.

If a single listing is clearly what the user means, return:
  {"auto": "TICKER", "reason": "one sentence"}

If the query is genuinely ambiguous across multiple valid companies, return:
  {"auto": null, "reason": "why it is ambiguous"}

Return ONLY valid JSON — no prose, no markdown.
"""


def _normalise_name(name: str) -> str:
    """Lower-case, strip legal suffixes and ADR/listing tags for fuzzy comparison."""
    name = name.lower()
    for suffix in (
        "limited", "ltd", "inc", "corp", "plc", "llc", "n.v.", "s.a.", "ag",
        # ADR / depositary-receipt tags that Polygon appends — must be stripped
        # so "Infosys Limited ADR" groups with "Infosys Limited" as one company.
        "adr", "adrc", "ads", "sponsored adr", "unsponsored adr",
    ):
        name = re.sub(rf"\b{re.escape(suffix)}\b\.?", "", name)
    return re.sub(r"\s+", " ", name).strip()


class ExchangeDisambiguatorAgent:
    """
    AI agent that resolves the intended stock listing from a list of candidates.

    Uses an LLM to auto-select obvious matches before falling back to an
    interactive menu, keeping the user experience smooth for common queries
    while still handling genuinely ambiguous ones gracefully.
    """

    def __init__(self, groq_client: Groq | None = None):
        self._client = groq_client

    def resolve(self, company_name: str) -> str:
        """
        Discover candidates via search_ticker, then disambiguate.
        Used as a direct fallback when no pre-fetched candidates are available.
        """
        result = search_ticker(company_name)
        if "error" in result:
            raise ValueError(result["error"])
        candidates = result.get("candidates", [])
        if not candidates:
            raise ValueError(f"No listings found for '{company_name}'")
        return self.resolve_from_candidates(company_name, candidates)

    def resolve_from_candidates(self, query: str, candidates: list) -> str:
        """
        Main entry point when candidates are already gathered externally.
        Raises ValueError if candidates list is empty or user cancels.

        Pipeline:
          1. Group by company name
          2. If single company: try LLM auto-select exchange; else ask
          3. If multiple companies: try LLM to identify obvious winner;
             else present interactive menu
        """
        if not candidates:
            raise ValueError(f"No listings found for '{query}'")

        # Group candidates by normalised company name
        company_groups = self._group_by_company(candidates)

        if len(company_groups) == 1:
            return self._resolve_single_company(
                query, list(company_groups.values())[0]
            )
        else:
            # Try LLM auto-select before showing interactive menu
            if self._client:
                auto = self._llm_auto_select(query, candidates)
                if auto:
                    print(f"  🤖 [ExchangeDisambiguatorAgent] Auto-selected → {auto}")
                    return auto
            return self._resolve_multiple_companies(query, company_groups)

    # ── grouping ──────────────────────────────────────────────────────────────

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

        # Within each group: equities first
        for key in groups:
            groups[key].sort(key=lambda c: 0 if c.get("type", "").upper() == "EQUITY" else 1)

        return groups

    # ── LLM auto-selection ────────────────────────────────────────────────────

    def _llm_auto_select(self, query: str, candidates: list) -> str | None:
        """
        Ask the LLM if one listing is the obvious intended match.

        Returns a ticker string if the LLM is confident (≥ 80%) about a single
        best match, otherwise returns None to fall through to the interactive menu.
        """
        lines = "\n".join(
            f"  {c['ticker']:<18} {c['exchange']:<22} {c['name']}"
            for c in candidates[:20]
        )
        try:
            resp = self._client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system",  "content": _SYSTEM_AUTOSELECT},
                    {"role": "user",    "content":
                     f"User query: '{query}'\n\nCandidates:\n{lines}"},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data   = json.loads(match.group())
                ticker = data.get("auto")
                reason = data.get("reason", "")
                if ticker:
                    # Verify the LLM-suggested ticker is in the candidate list
                    valid = {c["ticker"].upper() for c in candidates}
                    if ticker.upper() in valid:
                        print(f"  🤖 [ExchangeDisambiguatorAgent] "
                              f"LLM auto-selected {ticker}: {reason}")
                        return ticker.upper()
        except Exception as exc:
            print(f"  ⚠️  [ExchangeDisambiguatorAgent] LLM auto-select failed: {exc}")
        return None

    # ── single-company flow ───────────────────────────────────────────────────

    def _resolve_single_company(self, query: str, candidates: list) -> str:
        """One company, possibly on many exchanges."""
        unique_exchanges = self._best_per_exchange(candidates)

        if len(unique_exchanges) == 1:
            return self._auto_pick(unique_exchanges[0])

        return self._ask_exchange(query, unique_exchanges)

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

    # ── multi-company flow ────────────────────────────────────────────────────

    TOO_MANY = 10   # ask user to refine when company count exceeds this

    def _resolve_multiple_companies(self, query: str, groups: dict) -> str:
        """Many companies match — pick company first, then exchange if needed."""
        reps = []
        for norm_name, candidates in groups.items():
            best = candidates[0]   # equity-first after _group_by_company
            reps.append((norm_name, best, candidates))

        # If too many results, ask user to narrow down first
        if len(reps) > self.TOO_MANY:
            reps = self._ask_to_refine(query, reps)
            if reps is None:
                raise ValueError("Cancelled by user")

        print(f"\n  ⚠️  [ExchangeDisambiguatorAgent] "
              f"Multiple companies match '{query}':\n")
        for i, (_, best, _) in enumerate(reps, 1):
            adr_tag = "  [ADR]" if best.get("is_adr") else ""
            print(f"    {i}.  {best['ticker']:<16} {best['exchange']:<22} {best['name']}{adr_tag}")

        # Collect company choice
        while True:
            try:
                choice = input("\n  Choose a company number: ").strip()
            except (EOFError, KeyboardInterrupt):
                raise ValueError("Cancelled by user")

            if choice.isdigit() and 1 <= int(choice) <= len(reps):
                _, best, all_candidates = reps[int(choice) - 1]
                print(f"  ✅ Company selected: {best['name']}")
                unique_exchanges = self._best_per_exchange(all_candidates)
                if len(unique_exchanges) == 1:
                    return self._auto_pick(unique_exchanges[0])
                return self._ask_exchange(best["name"], unique_exchanges)

            print("  Invalid — enter a number from the list.")

    @staticmethod
    def _ask_to_refine(query: str, reps: list) -> list | None:
        """
        Called when more than TOO_MANY companies matched.
        Shows a summary and asks the user to type a more specific term.
        Filters the existing reps list by the refined term — no new API call needed.
        Returns the filtered reps, or None if the user cancels.
        """
        print(f"\n  ℹ️  [ExchangeDisambiguatorAgent] "
              f"'{query}' matched {len(reps)} companies — that's too many to list.\n"
              f"  A few examples:")
        for _, best, _ in reps[:5]:
            print(f"     • {best['name']}")
        if len(reps) > 5:
            print(f"     … and {len(reps) - 5} more")

        while True:
            try:
                refine = input(
                    f"\n  Please type a more specific name to narrow the results\n"
                    f"  (or press Enter to show all {len(reps)}, Ctrl+C to cancel): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return None

            # User pressed Enter — show everything
            if not refine:
                return reps

            # Filter existing reps by the refined term (case-insensitive substring)
            term = refine.lower()
            narrowed = [
                r for r in reps
                if term in r[1]["name"].lower()     # match on display name
                or term in r[1]["ticker"].lower()   # or ticker
            ]

            if not narrowed:
                print(f"  ⚠️  No companies matched '{refine}' — try a different term.")
                continue

            if len(narrowed) > ExchangeDisambiguatorAgent.TOO_MANY:
                print(f"  Still {len(narrowed)} matches — please be more specific.")
                continue

            print(f"  🔎 Narrowed to {len(narrowed)} compan"
                  f"{'y' if len(narrowed) == 1 else 'ies'} matching '{refine}'")
            return narrowed

    # ── prompts ───────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_pick(candidate: dict) -> str:
        print(f"  ✅ [ExchangeDisambiguatorAgent] "
              f"Resolved → {candidate['ticker']} "
              f"({candidate['name']}, {candidate['exchange']})")
        return candidate["ticker"]

    @staticmethod
    def _ask_exchange(label: str, exchanges: list) -> str:
        print(f"\n  ⚠️  [ExchangeDisambiguatorAgent] "
              f"'{label}' is listed on multiple exchanges:\n")
        for i, c in enumerate(exchanges, 1):
            adr_tag = "  [ADR]" if c.get("is_adr") else ""
            print(f"    {i}.  {c['ticker']:<16} {c['exchange']:<22} {c['name']}{adr_tag}")

        ticker_set = {c["ticker"].upper() for c in exchanges}
        while True:
            try:
                choice = input("\n  Choose a number or type the ticker directly: ").strip()
            except (EOFError, KeyboardInterrupt):
                raise ValueError("Cancelled by user")

            if choice.isdigit() and 1 <= int(choice) <= len(exchanges):
                pick = exchanges[int(choice) - 1]
                print(f"  ✅ Using → {pick['ticker']} ({pick['exchange']})")
                return pick["ticker"]

            if choice.upper() in ticker_set:
                print(f"  ✅ Using → {choice.upper()}")
                return choice.upper()

            print("  Invalid — enter a list number or a ticker symbol.")

