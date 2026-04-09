"""
CandidateFilterAgent
--------------------
Single purpose: given a raw candidate list and the user's original query,
discard listings that are clearly unrelated.

Two-stage approach:
  Stage 1 — Fast heuristic (no LLM)
    Keep candidates whose company name contains at least one significant word
    from the query.  Significant = length > 2, not a stop-word.
    Example: query "hindustan" → drop "Halliburton", "Bank of Nova Scotia".

  Stage 2 — LLM review (only if Stage 1 leaves fewer than 2 results OR
    the query is an abbreviation/ticker-like)
    Ask the LLM to score each remaining candidate for relevance and, if the
    list still looks sparse, suggest refined search terms for a re-fetch.

Returns (filtered_candidates, refetch_hints) where refetch_hints is a list
of alternative search strings to try (empty list if none needed).
"""
import re
from groq import Groq

_STOP_WORDS = {
    "the", "of", "and", "in", "for", "to", "a", "an", "on",
    "at", "by", "is", "its", "with", "as", "or",
}

_SYSTEM_REVIEW = """\
You are a stock candidate relevance filter.

Given:
  - A user query (company name or abbreviation)
  - A list of stock candidates (ticker, name, exchange)

Your job:
  1. Mark each candidate as KEEP or DROP.
     Keep: the company is genuinely related to the query.
     Drop: the company name has no meaningful connection to the query
           (e.g. same ticker letters by coincidence, unrelated ADR, etc.)
  2. If fewer than 2 candidates survive, suggest up to 3 refined search
     strings that might find more relevant companies.

Return ONLY JSON — no prose, no markdown:
{
  "keep": ["TICKER1", "TICKER2"],
  "drop": ["TICKER3"],
  "refetch": ["refined search term 1", "refined search term 2"]
}
"""


def _significant_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _names_overlap(query_words: set, name: str) -> bool:
    """
    Return True if any query word matches any word in the candidate name.
    Matching rules:
      1. Exact:      "pharma"  == "pharma"
      2. Prefix:     "pharma"  is prefix of "pharmaceutical"
      3. Suffix:     "pharma"  is suffix of (rare, but handled)
    Handles abbreviations ("pharma"→"pharmaceutical") and partial names.
    """
    name_words = _significant_words(name)
    for qw in query_words:
        for nw in name_words:
            if qw == nw or nw.startswith(qw) or qw.startswith(nw):
                return True
    return False


class CandidateFilterAgent:
    """
    Reviews raw yfinance candidates and removes irrelevant results.
    Falls back to LLM review when heuristic is insufficient.
    """

    MIN_KEEP = 2      # below this, trigger LLM review
    MODEL    = "llama-3.3-70b-versatile"

    def __init__(self, client: Groq):
        self.client = client

    def filter(
        self,
        query: str,
        candidates: list,
    ) -> tuple[list, list]:
        """
        Returns (kept_candidates, refetch_hints).
        refetch_hints is a list of alternative search strings (may be empty).
        """
        if not candidates:
            return [], []

        query_words = _significant_words(query)
        is_abbrev   = len(query.replace(" ", "")) <= 6 and query.upper() == query

        # ── Stage 1: name-based heuristic ────────────────────────────────────
        if query_words and not is_abbrev:
            kept = [
                c for c in candidates
                if _names_overlap(query_words, c.get("name", ""))
            ]
        else:
            # Abbreviation/ticker query — heuristic can't help; keep all
            kept = list(candidates)

        dropped = [c for c in candidates if c not in kept]

        if dropped:
            print(
                f"  🔍 [CandidateFilterAgent] Dropped {len(dropped)} irrelevant listing(s): "
                + ", ".join(f"{c['ticker']} ({c['name'][:30]})" for c in dropped[:4])
                + ("…" if len(dropped) > 4 else "")
            )

        # ── Stage 2: LLM review if too few survived ───────────────────────────
        if len(kept) < self.MIN_KEEP:
            print(
                f"  🤖 [CandidateFilterAgent] Only {len(kept)} candidate(s) after heuristic "
                f"— asking LLM to review and suggest re-fetch terms…"
            )
            kept, refetch = self._llm_review(query, kept + dropped)
        else:
            refetch = []

        print(
            f"  ✅ [CandidateFilterAgent] {len(kept)} relevant listing(s) kept"
            + (f" | Re-fetch hints: {refetch}" if refetch else "")
        )
        return kept, refetch

    # ── LLM review ────────────────────────────────────────────────────────────

    def _llm_review(self, query: str, candidates: list) -> tuple[list, list]:
        import json

        candidate_lines = "\n".join(
            f"  {c['ticker']:<18} {c['exchange']:<22} {c['name']}"
            for c in candidates
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_REVIEW},
                    {
                        "role": "user",
                        "content": (
                            f"Query: '{query}'\n\nCandidates:\n{candidate_lines}"
                        ),
                    },
                ],
                max_tokens=512,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                keep_tickers  = {t.upper() for t in data.get("keep", [])}
                refetch_hints = data.get("refetch", [])
                kept = (
                    [c for c in candidates if c["ticker"].upper() in keep_tickers]
                    if keep_tickers
                    else candidates   # LLM kept nothing → safety: keep all
                )
                return kept, refetch_hints
        except Exception as exc:
            print(f"  ⚠️  [CandidateFilterAgent] LLM review failed ({exc}) — keeping all")

        return candidates, []
