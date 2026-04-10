"""
CompanySearchAgent
------------------
Single purpose: given a partial name or abbreviation, find ALL publicly listed
companies that match, returning a structured list for downstream ticker lookup.

Uses llama-3.3-70b-versatile (reliable, no request-size issues) which has strong
knowledge of global equity markets from training data.

Falls back to a single-entry list (the raw query) if the LLM call fails,
so the rest of the pipeline always gets something to work with.
"""
import json
import re
from groq import Groq
from config import LLM_MODEL, TEMP_STRUCT, TOKENS_SEARCH

MODEL = LLM_MODEL

_SYSTEM = """\
You are a stock market expert. Given a company name, abbreviation, or industry term,
list ALL publicly listed companies whose name contains that term.

Return ONLY a JSON array, no prose:
[{"name": "Hindustan Unilever Limited", "ticker_hint": "HINDUNILVR"}, ...]

Rules:
- Use CURRENT official names (e.g. "Hindustan Unilever Limited", not "Hindustan Lever").
- Cover ALL exchanges: NSE, BSE, NYSE, NASDAQ, LSE, etc.
- For Indian terms, be exhaustive on NSE/BSE (many companies share prefixes).
- Equities only — no mutual funds, ETFs, or bonds.
- Return up to 15 companies. More is better than fewer.
- ticker_hint = current primary ticker without exchange suffix (null if unknown).
- Return ONLY the raw JSON array.
"""


class CompanySearchAgent:
    """
    Discovers publicly listed companies matching a query.
    Uses LLM training knowledge — fast and reliable on the free Groq tier.
    """

    def __init__(self, client: Groq):
        self.client = client

    def discover(self, query: str) -> list[dict]:
        """
        Return [{name, ticker_hint}, ...] for all listed companies matching query.
        Falls back to [{"name": query, "ticker_hint": None}] on any failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": f"List all listed companies matching: '{query}'"},
                ],
                max_tokens=TOKENS_SEARCH,
                temperature=TEMP_STRUCT,
            )
            content = response.choices[0].message.content or ""

            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                companies = json.loads(match.group())
                if isinstance(companies, list) and companies:
                    print(
                        f"  🤖 [CompanySearchAgent] Found {len(companies)} "
                        f"compan{'y' if len(companies)==1 else 'ies'}: "
                        + ", ".join(c.get("name", "?") for c in companies[:4])
                        + ("…" if len(companies) > 4 else "")
                    )
                    return companies

            print("  ⚠️  [CompanySearchAgent] Unparseable response — falling back")
        except Exception as exc:
            print(f"  ⚠️  [CompanySearchAgent] Failed ({exc}) — falling back")

        return [{"name": query, "ticker_hint": None}]

