"""
company_profile.py
------------------
CompanyProfileAgent — AI agent that synthesises company data, financial
metrics, and curated news into a well-formatted investor-facing snapshot.

Inputs are pre-assembled by the caller (CompanyDataTool + NewsAgent outputs).
A single one-shot LLM call narrates the combined data; no tool-calling loop
is needed since all raw data arrives in the user message.

Falls back to a structured plain-text format if the LLM is unavailable.
"""

import json
import math
from typing import Any, Dict, List, Optional

from groq import Groq
from config import LLM_MODEL, TEMP_REASON, TOKENS_PROFILE

from agents.base_agent import BaseAgent


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
Role: Synthesise financial data into an investor-facing company snapshot.

Constraints:
- Do NOT invent data not present in the input JSON — show "N/A" for missing values
- Large numbers: use B/M/T suffix with currency symbol (e.g. $2.4B, ₹45,000Cr)
- Percentages provided as floats (e.g. 17.2) → display as 17.2%, not 0.172
- Skip any section where all values are null/missing
- Total output ≤ 700 words

Output format: GitHub-flavoured markdown with these sections:

## 🏢 {Company Name}  ({TICKER})
{1–2 sentence business description}
**Sector:** {sector}  |  **Industry:** {industry}  |  **Country:** {country}  |  **Exchange:** {exchange}

---

### 📊 Key Metrics
| Metric | Value |
|--------|-------|
| Market Cap | ... |
| P/E (TTM / Fwd) | ... |
| EPS (TTM / Fwd) | ... |
| 52-Week Range | low – high |
| Beta | ... |
| Dividend Yield | ... % |
| Avg Volume (10d) | ... |

### 💰 Financials (TTM)
| Metric | Value |
|--------|-------|
| Revenue | ... |
| Gross Profit | ... |
| Net Income | ... |
| Gross / Net Margin | ...% / ...% |
| ROE / ROA | ...% / ...% |
| Debt / Equity | ... |
| Free Cash Flow | ... |
| Revenue Growth | ...% YoY |

### 📰 Latest News
- **YYYY-MM-DD** · [Headline](url) · *Publisher*
  > One-sentence investor summary.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """JSON serialisation fallback — converts NaN/Inf to null."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


def _fmt_large(val: Any, currency: str = "") -> str:
    """Format a large number with B/M/T suffix."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        prefix = currency + " " if currency else ""
        if v >= 1e12:
            return f"{prefix}{v / 1e12:.2f}T"
        if v >= 1e9:
            return f"{prefix}{v / 1e9:.2f}B"
        if v >= 1e6:
            return f"{prefix}{v / 1e6:.2f}M"
        return f"{prefix}{round(v, 2):,}"
    except Exception:
        return str(val)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CompanyProfileAgent(BaseAgent):
    """
    Synthesises CompanyDataTool output + NewsAgent output into a readable
    investor-facing markdown report.

    Usage
    -----
      agent   = CompanyProfileAgent(groq_client)
      report  = agent.summarise("INFY.NS", company_data, news)
    """

    model = LLM_MODEL
    tools = []   # One-shot — no tool-calling loop

    def summarise(
        self,
        ticker:       str,
        company_data: Dict[str, Any],
        news:         List[Dict[str, Any]],
        price_data:   Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Return a formatted markdown company snapshot.

        Parameters
        ----------
        ticker       : stock ticker symbol
        company_data : output from CompanyDataTool.fetch()
        news         : output from NewsAgent.fetch()
        price_data   : optional dict with current price info for context
        """
        if company_data.get("error"):
            return (
                f"⚠️  Company data unavailable for **{ticker}**: "
                f"{company_data['error']}"
            )

        payload: Dict[str, Any] = {
            "ticker":       ticker,
            "profile":      company_data.get("profile",      {}),
            "fundamentals": company_data.get("fundamentals", {}),
            "financials":   company_data.get("financials",   {}),
            "news":         news[:5],
        }
        if price_data:
            payload["current_price"] = price_data

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(payload, default=_json_safe),
                    },
                ],
                max_tokens=TOKENS_PROFILE,
                temperature=TEMP_REASON,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return self._basic_format(ticker, company_data, news)

    # ── Fallback (no LLM) ─────────────────────────────────────────────────────

    @staticmethod
    def _basic_format(
        ticker:       str,
        data:         Dict[str, Any],
        news:         List[Dict[str, Any]],
    ) -> str:
        """Plain-text fallback when the LLM is unavailable."""
        p   = data.get("profile",      {})
        f   = data.get("fundamentals", {})
        fin = data.get("financials",   {})
        cur = f.get("currency", "")

        lines = [
            f"## 🏢 {p.get('name', ticker)}  ({ticker})",
            (
                f"**Sector:** {p.get('sector') or 'N/A'}  |  "
                f"**Industry:** {p.get('industry') or 'N/A'}  |  "
                f"**Country:** {p.get('country') or 'N/A'}  |  "
                f"**Exchange:** {p.get('exchange') or 'N/A'}"
            ),
            "",
        ]

        if p.get("description"):
            lines += [p["description"], ""]

        # Key metrics table
        lines += [
            "### 📊 Key Metrics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Market Cap | {_fmt_large(f.get('market_cap'), cur)} |",
            f"| P/E (TTM / Fwd) | {f.get('pe_trailing') or 'N/A'} / {f.get('pe_forward') or 'N/A'} |",
            f"| EPS (TTM / Fwd) | {f.get('eps_trailing') or 'N/A'} / {f.get('eps_forward') or 'N/A'} |",
            f"| 52-Week Range | {f.get('week_52_low') or '?'} – {f.get('week_52_high') or '?'} |",
            f"| Beta | {f.get('beta') or 'N/A'} |",
            f"| Dividend Yield | {f.get('dividend_yield_pct') or 'N/A'}% |",
            "",
        ]

        # Financials table
        lines += [
            "### 💰 Financials (TTM)",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Revenue | {_fmt_large(fin.get('revenue'), cur)} |",
            f"| Net Income | {_fmt_large(fin.get('net_income'), cur)} |",
            f"| Gross / Net Margin | {fin.get('gross_margin_pct') or 'N/A'}% / {fin.get('net_margin_pct') or 'N/A'}% |",
            f"| ROE | {fin.get('roe_pct') or 'N/A'}% |",
            f"| Debt / Equity | {fin.get('debt_to_equity') or 'N/A'} |",
            f"| Free Cash Flow | {_fmt_large(fin.get('free_cash_flow'), cur)} |",
            "",
        ]

        if news:
            lines.append("### 📰 Latest News")
            for item in news[:5]:
                title = item.get("title", "?")
                url   = item.get("url", "#")
                pub   = item.get("publisher", "")
                date  = item.get("date", "")
                summ  = item.get("summary", "")
                lines.append(f"- **{date}** · [{title}]({url}) · *{pub}*")
                if summ:
                    lines.append(f"  > {summ}")

        return "\n".join(lines)
