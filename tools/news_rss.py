"""
news_rss.py
-----------
Free RSS-based news fallback for NewsAgent.

No API key required. Uses two public feeds:

  1. Yahoo Finance RSS  — ticker-specific headlines
     https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}

  2. Google News RSS    — broad search fallback
     https://news.google.com/rss/search?q={query}

Both return items in the same dict format that NewsAgent expects:
  { title, publisher, url, date, providerPublishTime }

Uses only stdlib (xml.etree.ElementTree) + requests (already in requirements).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict

import requests

_TIMEOUT = 8  # seconds per request


def _parse_rfc2822(date_str: str) -> int:
    """Parse RSS pubDate (RFC 2822) → Unix timestamp. Returns 0 on failure."""
    try:
        return int(parsedate_to_datetime(date_str).timestamp())
    except Exception:
        return 0


def _items_from_xml(xml_text: str) -> List[Dict]:
    """Parse an RSS 2.0 feed and return normalised item dicts."""
    items = []
    try:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return items

        for item in channel.findall("item"):
            title     = (item.findtext("title") or "").strip()
            link      = (item.findtext("link")  or "").strip()
            pub_date  = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            publisher = (
                source_el.text.strip()
                if source_el is not None and source_el.text
                else "Yahoo Finance"
            )

            ts = _parse_rfc2822(pub_date)
            try:
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                date_str = "unknown"

            if title and link:
                items.append({
                    "title":               title,
                    "publisher":           publisher,
                    "link":                link,
                    "url":                 link,
                    "date":                date_str,
                    "providerPublishTime": ts,
                })
    except ET.ParseError:
        pass
    return items


def fetch_yahoo_rss(ticker: str, max_items: int = 20) -> List[Dict]:
    """
    Fetch news for *ticker* from Yahoo Finance's public RSS feed.
    Returns [] on any network or parse error (never raises).
    """
    # Strip exchange suffix for the RSS endpoint (it only knows base tickers)
    base_ticker = ticker.split(".")[0]
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={base_ticker}&region=US&lang=en-US"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return _items_from_xml(resp.text)[:max_items]
    except Exception:
        return []


def fetch_google_news_rss(query: str, max_items: int = 20) -> List[Dict]:
    """
    Search Google News RSS for *query* (e.g. 'INFY Infosys stock news').
    Returns [] on any network or parse error (never raises).

    Note: Google News RSS items carry a redirector link; the article title
    and source name are reliable but the link requires a browser redirect.
    """
    url = (
        f"https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        items = _items_from_xml(resp.text)
        # Google News items have <source> elements; fall back to "Google News"
        for item in items:
            if item["publisher"] == "Yahoo Finance":
                item["publisher"] = "Google News"
        return items[:max_items]
    except Exception:
        return []
