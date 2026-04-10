"""
sentiment_agent.py
------------------
SentimentAgent — analyses curated news articles for a stock, assigns
per-article sentiment, aggregates to an overall score, and predicts the
most likely price direction for the NEXT TRADING DAY.

Inputs
------
  ticker  : verified ticker symbol  (e.g. "INFY", "INFY.NS")
  news    : list of curated article dicts from NewsAgent.fetch()
            Each item must have at least: title, summary (optional), publisher, date

Pipeline
--------
  1. Pull last 5 trading days close prices from yfinance → price context string
  2. Single LLM call:
       • Assign sentiment (positive / negative / neutral) + score 1–10 per article
       • Derive overall sentiment score
       • Predict next-trading-day direction with confidence + expected % move
  3. Format and return a structured dict + a ready-to-display markdown string

Output
------
  {
    "articles":   [{ title, sentiment, score }],
    "overall":    "positive" | "negative" | "neutral",
    "score":      float 0-10,
    "prediction": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "move":       "+0.5–1.5%" (LLM-estimated range),
    "reasoning":  "...",
    "formatted":  "...ready-to-print markdown...",
  }
"""

import json
import re
from typing import Any, Dict, List

import yfinance as yf
from groq import Groq

from agents.base_agent import BaseAgent
from config import LLM_MODEL, TEMP_STRUCT, TOKENS_SENTIMENT

_MODEL = LLM_MODEL

_SYSTEM = """\
Role: Score news sentiment per article and predict next-trading-day price direction.
Inputs: ticker, last 5 days closing prices, list of news articles.

Scoring rules:
- sentiment: "positive" | "negative" | "neutral" per article
- score: 1 (very negative) to 10 (very positive), 5 = neutral
- overall: weighted average score across all articles
- prediction: "bullish" | "bearish" | "neutral" for next trading day
- confidence: "high" | "medium" | "low"
- move: estimated % range (e.g. "+0.5-1.5%" or "-1-2%")
- reasoning: max 2-3 sentences on key drivers

Output ONLY valid JSON (no markdown, no prose):
{
  "articles": [{"title": "...", "sentiment": "positive|negative|neutral", "score": <1-10>}],
  "overall":    "positive|negative|neutral",
  "score":      <float 1-10>,
  "prediction": "bullish|bearish|neutral",
  "confidence": "high|medium|low",
  "move":       "<range e.g. +0.5-1.5%>",
  "reasoning":  "<max 2-3 sentences>"
}
"""

# Sentiment → display label
_LABEL = {
    "positive": "🟢 Positive",
    "negative": "🔴 Negative",
    "neutral":  "⚪ Neutral",
}

_PREDICTION_LABEL = {
    "bullish": "📈 BULLISH",
    "bearish": "📉 BEARISH",
    "neutral": "➡️  NEUTRAL",
}

_CONFIDENCE_LABEL = {
    "high":   "🔵 High",
    "medium": "🟡 Medium",
    "low":    "🔴 Low",
}


class SentimentAgent(BaseAgent):
    """
    Scores news sentiment per article, aggregates to an overall signal, and
    predicts the next-trading-day price direction.

    Uses a single one-shot LLM call (no tool loop) — all data is assembled
    from yfinance before invoking the LLM.
    """

    model = _MODEL
    tools = []  # one-shot, no tool-calling loop

    def analyse(self, ticker: str, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse sentiment and predict next-day direction.

        Returns a dict with keys:
          articles, overall, score, prediction, confidence, move, reasoning, formatted
        Returns None if news list is empty.
        """
        if not news:
            return None

        price_context = self._price_context(ticker)
        result        = self._llm_analyse(ticker, news, price_context)
        result["formatted"] = self._format(ticker, result, price_context)
        return result

    # ── price context ─────────────────────────────────────────────────────────

    @staticmethod
    def _price_context(ticker: str) -> str:
        """Return a short string describing recent closing prices."""
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty:
                return "Recent price data not available."
            lines = []
            for date, row in hist.iterrows():
                lines.append(
                    f"  {date.strftime('%Y-%m-%d')}  close={row['Close']:.2f}"
                )
            return "Last 5 trading days:\n" + "\n".join(lines)
        except Exception:
            return "Recent price data not available."

    # ── LLM analysis ─────────────────────────────────────────────────────────

    def _llm_analyse(
        self, ticker: str, news: List[Dict], price_context: str
    ) -> Dict[str, Any]:
        """Ask the LLM to score each article and predict next-day direction."""
        articles_text = json.dumps(
            [
                {
                    "title":     item.get("title", ""),
                    "summary":   item.get("summary", ""),
                    "publisher": item.get("publisher", ""),
                    "date":      item.get("date", ""),
                }
                for item in news
            ],
            indent=2,
        )

        user_msg = (
            f"Ticker: {ticker}\n\n"
            f"Recent Price Context:\n{price_context}\n\n"
            f"News Articles:\n{articles_text}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=TOKENS_SENTIMENT,
                temperature=TEMP_STRUCT,
            )
            text  = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Guarantee required keys exist
                data.setdefault("articles",   [])
                data.setdefault("overall",    "neutral")
                data.setdefault("score",      5.0)
                data.setdefault("prediction", "neutral")
                data.setdefault("confidence", "low")
                data.setdefault("move",       "unknown")
                data.setdefault("reasoning",  "")
                return data
        except Exception as exc:
            print(f"  ⚠️  [SentimentAgent] LLM analysis failed: {exc}")

        # Fallback: neutral on everything
        return {
            "articles":   [
                {"title": a.get("title", ""), "sentiment": "neutral", "score": 5}
                for a in news
            ],
            "overall":    "neutral",
            "score":      5.0,
            "prediction": "neutral",
            "confidence": "low",
            "move":       "unknown",
            "reasoning":  "Sentiment analysis unavailable.",
        }

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _format(ticker: str, result: Dict, price_context: str) -> str:
        """Render the analysis as a markdown string for display."""
        lines = [f"\n---\n\n### 📰 News Sentiment — {ticker}\n"]

        # Per-article table
        articles = result.get("articles", [])
        if articles:
            lines.append("| # | Headline | Sentiment | Score |")
            lines.append("|---|----------|-----------|-------|")
            for i, art in enumerate(articles, 1):
                sentiment = art.get("sentiment", "neutral")
                score     = art.get("score", 5)
                title     = (art.get("title") or "")[:70]
                if len(art.get("title", "")) > 70:
                    title += "…"
                lines.append(
                    f"| {i} | {title} "
                    f"| {_LABEL.get(sentiment, sentiment)} "
                    f"| {score}/10 |"
                )
            lines.append("")

        # Overall sentiment bar
        overall   = result.get("overall", "neutral")
        score     = float(result.get("score", 5.0))
        filled    = round(score)
        bar       = "█" * filled + "░" * (10 - filled)
        lines.append(
            f"**Overall Sentiment:** {_LABEL.get(overall, overall)}  "
            f"`[{bar}]` **{score:.1f}/10**\n"
        )

        # Next-day prediction block
        prediction = result.get("prediction", "neutral")
        confidence = result.get("confidence", "low")
        move       = result.get("move", "unknown")
        reasoning  = result.get("reasoning", "")

        lines.append("#### 🔭 Next Trading Day Outlook")
        lines.append(
            f"| | |"
            f"\n|---|---|"
            f"\n| **Direction** | {_PREDICTION_LABEL.get(prediction, prediction)} |"
            f"\n| **Confidence** | {_CONFIDENCE_LABEL.get(confidence, confidence)} |"
            f"\n| **Est. Move** | `{move}` |"
        )
        if reasoning:
            lines.append(f"\n> {reasoning}")

        lines.append(
            "\n> ⚠️  *This is a sentiment-based signal, not financial advice. "
            "Past sentiment does not guarantee future price movements.*"
        )

        return "\n".join(lines)
