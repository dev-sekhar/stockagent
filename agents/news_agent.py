"""
news_agent.py
-------------
NewsAgent — AI agent that fetches the latest company news from yfinance
and curates the most relevant items from a trusted source list.

Trusted / dependable financial news sources
-------------------------------------------
  Global:  Reuters, Bloomberg, Associated Press, Financial Times,
           Wall Street Journal, CNBC, MarketWatch, Barron's, Fortune,
           Forbes, Business Wire, PR Newswire, Benzinga, The Street,
           Investopedia, Motley Fool

  India:   Moneycontrol, Economic Times, Livemint / Mint,
           Business Standard, NDTV Profit, Hindu BusinessLine

Flow
----
  1. yfinance Ticker.news  → raw list (Yahoo Finance aggregated, ~20 items)
  2. Filter to curated publisher list
  3. If < 3 pass filter, fall back to full raw list
  4. LLM selects and summarises the 5 most investor-relevant items
  5. If LLM fails, return the top 5 raw items (no summaries)
"""

import datetime
import json
import re
from typing import Any, Dict, List

import yfinance as yf
from groq import Groq
from config import LLM_MODEL, TEMP_REASON, TOKENS_NEWS

from agents.base_agent import BaseAgent
from tools.news_rss import fetch_yahoo_rss, fetch_google_news_rss


# ── Trusted publisher registry ────────────────────────────────────────────────

_TRUSTED_SOURCES: frozenset = frozenset({
    # Global wire / broadcast
    "reuters",
    "bloomberg",
    "associated press", "ap news", "ap",
    "financial times", "ft.com",
    "wall street journal", "wsj",
    "cnbc",
    "marketwatch",
    "barron's", "barrons",
    "fortune",
    "forbes",
    "business wire",
    "pr newswire",
    "benzinga",
    "the street", "thestreet",
    "investopedia",
    "motley fool",
    # Indian financial press
    "moneycontrol",
    "economic times",
    "livemint", "mint",
    "business standard",
    "ndtv profit",
    "hindu businessline", "the hindu businessline",
    # Broad aggregator (Yahoo's own editorial)
    "yahoo finance",
})

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a financial news curator for investors.
You receive recent news articles about a company and must select the
5 most significant and relevant ones for an investor audience.

Prefer articles about:
  • Earnings, revenue guidance, analyst ratings/price targets
  • M&A, major partnerships, contract wins/losses
  • Regulatory actions, legal proceedings, government policy
  • Leadership changes, strategic pivots
  • Product launches, market expansion, technology developments

For each selected article write a concise one-sentence investor-focused summary
that captures WHY it matters (not just what happened).

Respond ONLY as a JSON array — no markdown fences, no extra text:
[
  {
    "title":     "original headline",
    "summary":   "one-sentence investor-relevant summary",
    "publisher": "source name",
    "url":       "article url",
    "date":      "YYYY-MM-DD"
  }
]
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class NewsAgent(BaseAgent):
    """
    Fetches and curates latest company news.

    Uses BaseAgent infrastructure but performs a one-shot LLM curation call
    rather than a tool-calling loop (all data is already assembled from
    yfinance before the LLM is invoked).
    """

    model = LLM_MODEL
    tools = []   # No tool loop needed for news curation

    def fetch(self, ticker: str, max_raw: int = 25) -> List[Dict[str, Any]]:
        """
        Return up to 5 curated news items for *ticker*.

        Source priority (first non-empty wins):
          1. yfinance Ticker.news         — Yahoo Finance aggregated feed
          2. Yahoo Finance RSS            — free, no API key, same underlying source
          3. Google News RSS              — free, no API key, broad coverage

        Each item has keys: title, summary, publisher, url, date.
        Returns [] if no news is available from any source.
        """
        raw = self._get_raw_news(ticker, max_raw)

        if not raw:
            print(f"  ℹ️  [NewsAgent] yfinance returned no news for {ticker} — trying RSS feeds…")
            raw = self._get_rss_news(ticker, max_raw)

        if not raw:
            print(f"  ⚠️  [NewsAgent] No news found for {ticker} from any source.")
            return []

        trusted    = self._filter_trusted(raw)
        candidates = trusted if len(trusted) >= 3 else raw[:max_raw]
        return self._llm_curate(candidates[:15], ticker)

    # ── Raw news ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_raw_news(ticker: str, max_items: int) -> List[Dict]:
        try:
            items = yf.Ticker(ticker).news
            return (items or [])[:max_items]
        except Exception:
            return []

    @staticmethod
    def _get_rss_news(ticker: str, max_items: int) -> List[Dict]:
        """
        Fallback: try Yahoo Finance RSS then Google News RSS.
        Merges results, deduplicates by title, returns up to max_items.
        """
        seen_titles: set = set()
        merged: List[Dict] = []

        for item in fetch_yahoo_rss(ticker, max_items):
            key = item["title"].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                merged.append(item)

        if len(merged) < 5:
            # Build a search query: base ticker + company name if available
            base = ticker.split(".")[0]
            try:
                info = yf.Ticker(ticker).info
                company = info.get("longName") or info.get("shortName") or base
            except Exception:
                company = base
            query = f"{base} {company} stock news"
            for item in fetch_google_news_rss(query, max_items):
                key = item["title"].lower()
                if key not in seen_titles:
                    seen_titles.add(key)
                    merged.append(item)

        return merged[:max_items]

    # ── Source filter ─────────────────────────────────────────────────────────

    @staticmethod
    def _filter_trusted(items: List[Dict]) -> List[Dict]:
        out = []
        for item in items:
            pub = (item.get("publisher") or "").lower().strip()
            if any(src in pub for src in _TRUSTED_SOURCES):
                out.append(item)
        return out

    # ── LLM curation ─────────────────────────────────────────────────────────

    def _llm_curate(self, items: List[Dict], ticker: str) -> List[Dict]:
        """Send up to 15 items to the LLM; return 5 curated results."""
        compact = []
        for item in items:
            # Support both yfinance format (providerPublishTime / link)
            # and RSS format (providerPublishTime / url already normalised)
            ts = item.get("providerPublishTime", 0)
            if item.get("date") and item["date"] != "unknown":
                date_str = item["date"]
            else:
                try:
                    date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                except Exception:
                    date_str = "unknown"
            compact.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "url":       item.get("link") or item.get("url", ""),
                "date":      date_str,
            })

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Ticker: {ticker}\n"
                            f"Articles:\n{json.dumps(compact)}"
                        ),
                    },
                ],
                max_tokens=TOKENS_NEWS,
                temperature=TEMP_REASON,
            )
            text  = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result[:5]
        except Exception:
            pass

        # Fallback: top-5 without LLM summaries
        return compact[:5]
