"""
Orchestrator
------------
Self-Healing Supervisor that routes natural-language queries to sub-agents.

Pipeline:
  1. Groq LLM classifies the query intent and extracts entities
     (company/ticker, period or start/end dates).
  2. If a company name is given, ExchangeDisambiguatorAgent resolves it.
  3. The appropriate price agent is called with:
       • RateLimiter         — token-bucket cap (30 calls/min)
       • _with_retry()       — exponential back-off (3 attempts)
       • CircuitBreaker      — opens after 3 consecutive failures; auto-heals
       • Fallback chains     — if agent A fails, agent B is tried automatically
       • Ticker-variant heal — INFY → INFY.NS → INFY.BO → … on resolution failure
       • LLM self-diagnosis  — classifies errors → chooses wait/variant/fallback

Self-healing components live in supervisor.py (SelfHealingSupervisor).
Call orchestrator.health_report() at any time to inspect circuit states.
"""
import json
import time
import threading
from groq import Groq

from supervisor import SelfHealingSupervisor, CircuitOpenError

from agents.ticker_resolver        import TickerResolverAgent
from agents.company_search         import CompanySearchAgent
from tools.polygon_search          import PolygonSearchAgent
from agents.candidate_filter       import CandidateFilterAgent
from agents.exchange_disambiguator import ExchangeDisambiguatorAgent
from agents.current_price          import CurrentPriceAgent
from agents.historical_price       import HistoricalPriceAgent
from agents.date_range             import DateRangeAgent
from agents.chart_agent            import ChartAgent
from agents.comparison_chart_agent import ComparisonChartAgent
from agents.news_agent             import NewsAgent
from agents.sentiment_agent        import SentimentAgent
from agents.company_profile        import CompanyProfileAgent
from tools.company_data            import CompanyDataTool

from config import (
    LLM_MODEL, TEMP_STRUCT, TOKENS_CLASSIFY,
    GROQ_RPM,
    CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT,
    CB_RECOVERY_POLYGON, CB_RECOVERY_PRICE,
    DEFAULT_BENCHMARK, DEFAULT_BENCHMARK_NAME,
)
from tools.access_gateway import gateway

MODEL = LLM_MODEL

# ── Orchestration tool definitions ───────────────────────────────────────────
# These are "meta-tools": the LLM uses them to declare intent + extract params.
# Each maps 1-to-1 to a downstream agent workflow.

ROUTING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_stock_price",
            "description": (
                "User wants the current / latest / last traded price of a stock. "
                "Also covers 'today\'s price', 'right now', 'what is it trading at'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_or_ticker": {
                        "type": "string",
                        "description": "Company name (e.g. 'Apple') or ticker (e.g. 'AAPL').",
                    }
                },
                "required": ["company_or_ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_stock_prices",
            "description": (
                "User wants historical price data over a relative window: "
                "'last month', '3 months', 'last year', '5 years', 'year-to-date', 'all time'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_or_ticker": {
                        "type": "string",
                        "description": "Company name or ticker symbol.",
                    },
                    "period": {
                        "type": "string",
                        "description": "yfinance period code: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Bar interval: 1d (default) 1wk 1mo",
                        "default": "1d",
                    },
                },
                "required": ["company_or_ticker", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_prices_between_dates",
            "description": (
                "User specifies an explicit start AND end date, "
                "e.g. 'from Jan 2023 to June 2023' or 'between 2022-01-01 and 2022-12-31'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_or_ticker": {
                        "type": "string",
                        "description": "Company name or ticker symbol.",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD.",
                    },
                },
                "required": ["company_or_ticker", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": (
                "User wants a company snapshot: profile, financials, key metrics, "
                "or latest news. Covers 'tell me about', 'company info', 'fundamentals', "
                "'who is', 'what does X do', 'news about', 'financials for'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_or_ticker": {
                        "type": "string",
                        "description": "Company name or ticker symbol.",
                    }
                },
                "required": ["company_or_ticker"],
            },
        },
    },
]

_SYSTEM = (
    "You are a stock query router. Analyse the user's message and call exactly one tool:\n"
    "  • get_current_stock_price         – for latest / current price requests\n"
    "  • get_historical_stock_prices     – for relative-period requests (last month, ytd …)\n"
    "  • get_stock_prices_between_dates  – for explicit start/end date requests\n"
    "  • get_company_info                – for company profile, fundamentals, news, 'tell me about'\n"
    "CRITICAL: Extract the company name or ticker EXACTLY as the user typed it — "
    "do NOT expand, complete, guess, or add words to it. "
    "If the user typed 'hindustan', pass 'hindustan'. "
    "If the user typed 'hdfc', pass 'hdfc'. Never infer the full company name. "
    "Convert natural-language dates to YYYY-MM-DD. "
    "Map natural-language periods to yfinance codes (e.g. 'last year' → '1y')."
)


def _is_ticker(s: str) -> bool:
    """Heuristic: 1–5 all-uppercase ASCII letters → probably already a ticker."""
    return bool(s) and s.upper() == s and s.replace(".", "").isalpha() and len(s) <= 6


# Words that signal the user wants price/time data (word-level match)
_PRICE_INTENT_WORDS = {
    "price", "cost", "worth", "trading", "trade", "current", "now", "today",
    "latest", "last", "close", "open", "high", "low",
    "history", "historical", "historic", "past", "previous", "trend",
    "week", "month", "year", "ytd", "all", "since", "between", "from", "to",
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max",
    "chart", "graph", "plot",
}

# Single-word company-info signals — matched at WORD boundaries (set intersection)
# so "info" won't match inside "infosys", "about" won't match inside "aboutface" etc.
_COMPANY_INTENT_WORDS = {
    "about", "info", "information", "profile", "overview",
    "news", "financials", "fundamentals", "earnings", "revenue", "describe",
}

# Multi-word phrases — must stay as substring match (can't split on spaces)
_COMPANY_INTENT_PHRASES = [
    "tell me about", "who is", "what is", "what does",
]

# Words that signal the user wants to compare multiple stocks
_COMPARE_WORDS = {"compare", "comparison", "vs", "versus", "alongside", "relative"}


def _query_has_explicit_intent(query: str) -> bool:
    """
    Return True only when the query has EXPLICIT keywords that make the desired
    data type unambiguous.  Bare company names (e.g. 'apple', 'infosys') return
    False so the clarification menu is always shown.

    Single-word signals use set intersection (word-boundary) to prevent false
    positives such as 'info' matching inside 'infosys'.
    """
    lower = query.lower()
    words = set(lower.split())
    if words & _PRICE_INTENT_WORDS:          # price/time words
        return True
    if words & _COMPANY_INTENT_WORDS:        # company-info words (word-boundary)
        return True
    return any(p in lower for p in _COMPANY_INTENT_PHRASES)  # multi-word phrases


def _query_has_price_intent(query: str) -> bool:
    """Legacy alias — kept so nothing else breaks."""
    return _query_has_explicit_intent(query)


def _is_comparison_query(query: str) -> bool:
    """Return True if the query explicitly asks to compare multiple stocks."""
    lower = query.lower()
    return any(w in lower for w in _COMPARE_WORDS)


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket rate limiter — caps outbound API calls to calls_per_minute.
    Thread-safe; shared across all agents in the process.
    """
    def __init__(self, calls_per_minute: int = 30):
        self._limit = calls_per_minute
        self._timestamps: list = []
        self._lock = threading.Lock()

    def acquire(self):
        """Block if the rate limit is reached, then record this call."""
        with self._lock:
            now = time.time()
            # Drop timestamps older than 60 s
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self._limit:
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
                print(f"  ⏳ [RateLimiter] Limit reached — waiting {wait:.1f}s…")
                time.sleep(wait)
                self._timestamps = [t for t in self._timestamps if time.time() - t < 60]
            self._timestamps.append(time.time())


_rate_limiter = RateLimiter(calls_per_minute=GROQ_RPM)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, *args, max_attempts: int = 3, supervisor=None, agent_name: str = "", **kwargs):
    """
    Call fn(*args, **kwargs) with exponential back-off on failure.

    If a SelfHealingSupervisor is provided, the error is classified before
    each retry:
      • wait_retry  → longer initial wait for rate-limit errors (60 s)
      • use_fallback / abort → stop retrying immediately
    Raises the last exception if all attempts are exhausted.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            _rate_limiter.acquire()
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break

            # Classify to pick a smarter wait time
            strategy = (
                supervisor.classify_error(str(exc))
                if supervisor
                else "try_variant"
            )

            if strategy in ("use_fallback", "abort"):
                break   # Don't retry — caller should use its fallback chain

            # Rate-limit errors need a longer back-off
            wait = 60 if strategy == "wait_retry" else 2 ** attempt
            print(
                f"  ⚠️  [Retry] Attempt {attempt}/{max_attempts} failed "
                f"({strategy}): {exc}. Retrying in {wait}s…"
            )
            time.sleep(wait)
    raise last_exc


class Orchestrator:
    # ── Exchange-suffix variants tried when base ticker resolution fails ───────
    _SUFFIX_VARIANTS = [".NS", ".BO", ".L", ".AX", ".TO", ".HK", ".T", ""]

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

        # ── Sub-agents ────────────────────────────────────────────────────────
        self.ticker_resolver        = TickerResolverAgent(self.client)
        self.company_search         = CompanySearchAgent(self.client)
        self.polygon_search         = PolygonSearchAgent.from_env()
        self.candidate_filter       = CandidateFilterAgent(self.client)
        self.disambiguator          = ExchangeDisambiguatorAgent(self.client)
        self.current_agent          = CurrentPriceAgent(self.client)
        self.historical_agent       = HistoricalPriceAgent(self.client)
        self.date_range_agent       = DateRangeAgent(self.client)
        self.chart_agent            = ChartAgent(self.client)
        self.comparison_chart_agent = ComparisonChartAgent()
        self.company_data_tool      = CompanyDataTool()
        self.news_agent             = NewsAgent(self.client)
        self.sentiment_agent        = SentimentAgent(self.client)
        self.company_profile_agent  = CompanyProfileAgent(self.client)

        # ── Self-Healing Supervisor ───────────────────────────────────────────
        self.supervisor = (
            SelfHealingSupervisor(self.client)
            .register("Polygon",        failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_POLYGON)
            .register("CompanySearch",  failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_TIMEOUT)
            .register("CurrentPrice",   failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_PRICE)
            .register("HistoricalPrice",failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_PRICE)
            .register("DateRange",      failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_PRICE)
            .register("ChartAgent",     failure_threshold=max(1, CB_FAILURE_THRESHOLD - 1), recovery_timeout=CB_RECOVERY_TIMEOUT)
            .register("CompanyData",    failure_threshold=max(1, CB_FAILURE_THRESHOLD - 1), recovery_timeout=CB_RECOVERY_TIMEOUT)
            .register("NewsAgent",      failure_threshold=max(1, CB_FAILURE_THRESHOLD - 1), recovery_timeout=CB_RECOVERY_TIMEOUT)
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _ticker_variants(ticker: str) -> list:
        """
        Generate exchange-suffix variants to try when a ticker fails.

        E.g. INFY → [INFY.NS, INFY.BO, INFY.L, INFY.AX, …]
             INFY.NS → [INFY, INFY.BO, …]
        """
        base           = ticker.split(".")[0]
        current_suffix = ticker[len(base):]
        return [
            base + s
            for s in Orchestrator._SUFFIX_VARIANTS
            if s != current_suffix
        ]

    def _resolve(self, company_or_ticker: str) -> str:
        """
        Return a verified ticker symbol.

        Pipeline  (self-healing fallback chain)
        ----------------------------------------
        Fast path  — looks like a ticker AND yfinance confirms it.

        Slow path  (each step guarded by a circuit breaker):
          A. Polygon.io search          [circuit: Polygon]
          B. CompanySearchAgent         [circuit: CompanySearch]
             → per company: yfinance name search
          C. Direct yfinance search on raw query (safety net — no circuit)
          D. CandidateFilterAgent filter + optional re-fetch
          E. ExchangeDisambiguatorAgent → final ticker

        Healing:
          • If CompanySearch circuit opens  → Step B is skipped (Step C still runs)
          • If Polygon circuit opens        → Step A is skipped silently
          • If resolution still fails       → try exchange-suffix variants via
            _ticker_variants() before giving up
        """
        from tools.ticker_tools import verify_ticker, search_ticker

        if _is_ticker(company_or_ticker):
            print(f"  🔎 [TickerResolverAgent] Verifying '{company_or_ticker}' as a ticker…")
            if verify_ticker(company_or_ticker):
                print(f"  ✅ Ticker confirmed: {company_or_ticker.upper()}")
                return company_or_ticker.upper()
            print(f"  ↩️  '{company_or_ticker}' not found as ticker — searching by name…")

        all_candidates: list = []
        seen_tickers:   set  = set()

        def _add_raw(candidates: list):
            for c in candidates:
                if c["ticker"] not in seen_tickers:
                    seen_tickers.add(c["ticker"])
                    all_candidates.append(c)

        def _add_candidates(result: dict):
            _add_raw(result.get("candidates", []))

        # ── Step A: Polygon.io (circuit-protected) ────────────────────────────
        if self.polygon_search:
            try:
                polygon_hits = self.supervisor.call(
                    "Polygon", self.polygon_search.search,
                    company_or_ticker, limit=20,
                )
                _add_raw(polygon_hits)
            except CircuitOpenError as e:
                print(f"  ⚠️  {e} — skipping Polygon")
            except Exception as exc:
                diag = self.supervisor.diagnose(
                    "Polygon", str(exc), {"query": company_or_ticker}
                )
                print(f"  ⚠️  [Polygon] {exc} → {diag['action']}: {diag['suggestion']}")
        else:
            print("  ℹ️  [Orchestrator] POLYGON_API_KEY not set — skipping Polygon search")

        # ── Step B: CompanySearchAgent (circuit-protected, graceful skip) ─────
        companies: list = []
        if self.supervisor.is_healthy("CompanySearch"):
            print(f"  🌐 [CompanySearchAgent] Searching for companies matching '{company_or_ticker}'…")
            try:
                companies = self.supervisor.call(
                    "CompanySearch", self.company_search.discover,
                    company_or_ticker,
                )
            except CircuitOpenError as e:
                print(f"  ⚠️  {e} — using direct yfinance only")
            except Exception as exc:
                diag = self.supervisor.diagnose(
                    "CompanySearch", str(exc), {"query": company_or_ticker}
                )
                print(f"  ⚠️  [CompanySearch] {exc} → {diag['action']}: {diag['suggestion']}")
        else:
            print("  ⚠️  [CompanySearch] circuit OPEN — skipping web search, using yfinance only")

        for company in companies:
            hint = company.get("ticker_hint")
            name = company.get("name") or company_or_ticker
            before = len(seen_tickers)
            _add_candidates(search_ticker(name))
            name_added = len(seen_tickers) > before
            if not name_added and hint:
                _add_candidates(search_ticker(hint))

        # ── Step C: Direct yfinance search (no circuit — always runs) ─────────
        _add_candidates(search_ticker(company_or_ticker, max_candidates=15, max_results=50))

        if not all_candidates:
            raise ValueError(f"No listed companies found for '{company_or_ticker}'")

        # ── Step C.5: probe Indian exchange variants for any ADR found ─────────
        # yfinance search often returns the US ADR but omits .NS/.BO listings.
        # We verify those variants directly so the exchange menu can appear.
        adr_candidates = [c for c in all_candidates if c.get("is_adr")]
        for adr in adr_candidates:
            base = adr["ticker"].split(".")[0]
            for suffix, exchange_label, raw_exchange in [
                (".NS", "NSE (India)", "NSI"),
                (".BO", "BSE (India)", "BSE"),
            ]:
                variant = base + suffix
                if variant not in seen_tickers and verify_ticker(variant):
                    seen_tickers.add(variant)
                    all_candidates.append({
                        "ticker":       variant,
                        "name":         adr["name"],
                        "exchange":     exchange_label,
                        "raw_exchange": raw_exchange,
                        "type":         "EQUITY",
                        "is_adr":       False,
                    })
                    print(f"  🔍 [Orchestrator] Found exchange variant: {variant}")

        # ── Step D: filter + optional re-fetch ────────────────────────────────
        filtered, refetch_hints = self.candidate_filter.filter(
            company_or_ticker, all_candidates
        )

        if refetch_hints:
            for hint in refetch_hints[:3]:
                print(f"  🔄 [Orchestrator] Re-fetching with refined term: '{hint}'…")
                _add_candidates(search_ticker(hint, max_candidates=10))
            filtered, _ = self.candidate_filter.filter(company_or_ticker, all_candidates)

        if not filtered:
            raise ValueError(f"No relevant listed companies found for '{company_or_ticker}'")

        print(
            f"  📋 [TickerResolverAgent] {len(filtered)} relevant listing(s) from "
            f"{len(companies)} compan{'y' if len(companies) == 1 else 'ies'} found"
        )

        # ── Step E: disambiguate exchange ──────────────────────────────────────
        return self.disambiguator.resolve_from_candidates(company_or_ticker, filtered)

    # ── comparison flow ──────────────────────────────────────────────────────

    def _run_comparison(self, initial_tickers: list[str] | None = None) -> str:
        """
        Interactively collect 2+ tickers from the user and render a comparison chart.

        initial_tickers: pre-resolved ticker(s) already collected before entering
                         comparison mode (e.g. from option 4 in _clarify_intent).
        """
        print("\n  📊 [ComparisonChartAgent] Stock Comparison Mode")
        print("  ─────────────────────────────────────────────")
        tickers: list[str] = list(initial_tickers or [])
        if tickers:
            print(f"  Already added: {', '.join(tickers)}")

        n = len(tickers) + 1
        while True:
            suffix = "" if n <= 2 else "  (or press Enter to proceed)"
            prompt = f"  Enter stock {n}{suffix}: "
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                if len(tickers) >= 2:
                    break
                print("  ⚠️  Please enter at least 2 stocks to compare.")
                continue

            try:
                ticker = self._resolve(raw)
                tickers.append(ticker)
                print(f"  ✅ Added: {ticker}")
                n += 1
            except ValueError as exc:
                print(f"  ❌ {exc} — try again")

        if len(tickers) < 2:
            return "Comparison cancelled — need at least 2 stocks."

        # ── Detect cross-exchange: same base symbol on different exchanges ─────
        arbitrage = False
        base_map: dict[str, str] = {}
        cross_pairs: list[tuple[str, str]] = []
        for t in tickers:
            base = t.split(".")[0]
            if base in base_map:
                cross_pairs.append((base_map[base], t))
            else:
                base_map[base] = t

        if cross_pairs:
            pairs_str = ", ".join(f"{a} vs {b}" for a, b in cross_pairs)
            print(f"\n  💱 Cross-exchange listing(s) detected: {pairs_str}")
            print("  This may indicate arbitrage potential.")
            try:
                resp = input("  Show arbitrage spread view? (y/n) [n]: ").strip().lower()
                arbitrage = (resp == "y")
            except (EOFError, KeyboardInterrupt):
                pass

        # ── Period / date range ───────────────────────────────────────────────
        print("\n  ⏱️  Time period:")
        print("  Options: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max")
        try:
            period   = input("  Period [default: 1mo]: ").strip() or "1mo"
        except (EOFError, KeyboardInterrupt):
            period = "1mo"

        print(f"\n  📊 Comparing {' vs '.join(tickers)} ({period})…")
        return self.comparison_chart_agent.plot(
            tickers, period=period, arbitrage=arbitrage
        )

    # ── intent clarification ─────────────────────────────────────────────────

    @staticmethod
    def _ask_output_format() -> str:
        """Ask whether the user wants raw data (LLM narration) or a chart."""
        print("\n  📊 How would you like the data?\n")
        print("    1.  Summary / raw data  (text)")
        print("    2.  Chart  (interactive HTML — opens in browser)")
        while True:
            try:
                choice = input("\n  Enter 1 or 2: ").strip()
            except (EOFError, KeyboardInterrupt):
                return "raw"
            if choice == "1":
                return "raw"
            if choice == "2":
                return "chart"
            print("  Please enter 1 or 2.")

    @staticmethod
    def _clarify_intent(ticker: str, intent: str, args: dict) -> list[tuple[str, dict]]:
        """
        Called when the user's query had no explicit price/time keywords.

        Presents a multi-select menu.  Users may enter a single choice (e.g. "1")
        or a comma-separated combination (e.g. "1,4" or "3,4,5").

        Rules
        -----
        - Options 1/2/3/4 can be freely combined.
        - Option 5 (compare) must be selected alone.
        - If only 1, 2, or 3 are selected, company info / news / financials
          are NOT fetched.
        - Options 2 and 3 collect their required parameters (period/dates/format)
          before the list is returned.

        Returns a list of (intent, args) pairs to execute in order.
        """
        print(f"\n  ❓ [Orchestrator] What would you like for {ticker}?\n")
        print(f"    1.  Current / latest price")
        print(f"    2.  Historical data  (relative period — e.g. last 1 month)")
        print(f"    3.  Price between two dates")
        print(f"    4.  Company information & news  (profile, financials, headlines)")
        print(f"    5.  Compare with another stock  (must be selected alone)")
        print(f"\n  Tip: enter a single number or combine with commas, e.g. 1,4 or 3,4,5")

        while True:
            try:
                raw = input("\n  Enter choice(s): ").strip()
            except (EOFError, KeyboardInterrupt):
                raise ValueError("Cancelled by user")

            # Parse comma-separated numbers
            try:
                choices = sorted({int(c.strip()) for c in raw.split(",") if c.strip()})
            except ValueError:
                print("  ⚠️  Please enter numbers separated by commas, e.g. 1 or 1,4.")
                continue

            if not choices or any(c not in (1, 2, 3, 4, 5) for c in choices):
                print("  ⚠️  Valid options are 1, 2, 3, 4, 5.")
                continue

            if 5 in choices and len(choices) > 1:
                print("  ⚠️  Option 5 (Compare) must be selected on its own.")
                continue

            # Option 5 — handled separately, return early
            if choices == [5]:
                return [("compare", args)]

            # Collect params for option 2 (historical) if selected
            hist_intent = None
            hist_args: dict = {}
            if 2 in choices:
                period = input(
                    "  Period (1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max) [default: 1mo]: "
                ).strip() or "1mo"
                fmt = Orchestrator._ask_output_format()
                hist_args = {**args, "period": period, "interval": "1d", "output_format": fmt}
                hist_intent = "chart_historical" if fmt == "chart" else "get_historical_stock_prices"

            # Collect params for option 3 (date range) if selected
            range_intent = None
            range_args: dict = {}
            if 3 in choices:
                start = input("  Start date (YYYY-MM-DD): ").strip()
                end   = input("  End date   (YYYY-MM-DD): ").strip()
                fmt   = Orchestrator._ask_output_format()
                range_args = {**args, "start": start, "end": end, "output_format": fmt}
                range_intent = "chart_date_range" if fmt == "chart" else "get_stock_prices_between_dates"

            # Build ordered action list
            actions: list[tuple[str, dict]] = []
            for c in choices:
                if c == 1:
                    actions.append(("get_current_stock_price", args))
                elif c == 2 and hist_intent:
                    actions.append((hist_intent, hist_args))
                elif c == 3 and range_intent:
                    actions.append((range_intent, range_args))
                elif c == 4:
                    actions.append(("get_company_info", args))

            return actions

    # ── public entry point ───────────────────────────────────────────────────

    def health_report(self) -> str:
        """Return a markdown health table for all registered agents."""
        return self.supervisor.health_report()

    def _fetch_company_snapshot(self, ticker: str) -> tuple:
        """
        Fetch company data + news in parallel (best-effort).

        Returns (company_data_dict, news_list).
        Either may be {} / [] if the fetch fails or the circuit is open.
        """
        company_data: dict = {}
        news: list = []

        # Company data (circuit-protected)
        try:
            company_data = self.supervisor.call(
                "CompanyData", self.company_data_tool.fetch, ticker
            )
        except CircuitOpenError as e:
            print(f"  ⚠️  {e}")
        except Exception as exc:
            print(f"  ⚠️  [CompanyData] {exc}")

        # News (circuit-protected)
        try:
            news = self.supervisor.call(
                "NewsAgent", self.news_agent.fetch, ticker
            )
        except CircuitOpenError as e:
            print(f"  ⚠️  {e}")
        except Exception as exc:
            print(f"  ⚠️  [NewsAgent] {exc}")

        return company_data, news

    def _fetch_sentiment(self, ticker: str, news: list) -> str:
        """
        Run SentimentAgent on curated news items.
        Returns the formatted markdown string, or "" on failure / empty news.
        """
        if not news:
            return ""
        try:
            print(f"  🧠 [SentimentAgent] Analysing {len(news)} article(s) for {ticker}…")
            result = self.sentiment_agent.analyse(ticker, news)
            if result:
                return result.get("formatted", "")
        except Exception as exc:
            print(f"  ⚠️  [SentimentAgent] {exc}")
        return ""

    def handle(self, user_query: str) -> str:
        """
        Classify the query, resolve the ticker, delegate to the right agent.

        Every price / chart call is:
          1. Wrapped by the circuit breaker (supervisor.call)
          2. Retried with error-type-aware back-off (_with_retry)
          3. Healed with ticker-suffix variants if the base ticker fails
          4. Fallen back to an alternative agent if the primary circuit opens
        """

        # Step 0 – comparison shortcut
        if _is_comparison_query(user_query):
            print("  🔀 [Orchestrator] Comparison mode detected")
            return self._run_comparison()

        # Step 1 – intent classification + entity extraction
        gateway.check("LLM")
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_query},
            ],
            tools=ROUTING_TOOLS,
            tool_choice="required",
            max_tokens=TOKENS_CLASSIFY,
            temperature=TEMP_STRUCT,
        )

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return "Sorry, I couldn't understand your query. Try asking for a stock price by name or ticker."

        tc     = msg.tool_calls[0]
        intent = tc.function.name
        args   = json.loads(tc.function.arguments)
        print(f"  🧭 [Orchestrator] Intent: {intent} | Args: {args}")

        # Step 2 – resolve ticker (with exchange-suffix healing on failure)
        ticker = self._resolve_with_healing(args["company_or_ticker"])

        # Step 3 – clarify if query has no explicit intent signal
        # Both price/time words AND company-info phrases are checked.
        # A bare name like "apple" has neither → always shows the menu.
        if not _query_has_explicit_intent(user_query):
            actions = self._clarify_intent(ticker, intent, args)
        else:
            actions = [(intent, args)]

        # Handle compare (must be alone)
        if len(actions) == 1 and actions[0][0] == "compare":
            return self._run_comparison(initial_tickers=[ticker])

        # Step 4 – pre-fetch company snapshot once if any action needs it.
        # Only options that explicitly request company info trigger the fetch.
        _COMPANY_INTENTS = {"get_company_info", "chart_historical", "chart_date_range"}
        needs_snapshot   = any(a[0] in _COMPANY_INTENTS for a in actions)

        company_data: dict = {}
        news: list         = []
        if needs_snapshot:
            company_data, news = self._fetch_company_snapshot(ticker)

        # Step 5 – execute each action and collect results
        results: list[str] = []
        for action_intent, action_args in actions:
            result = self._execute_intent(action_intent, action_args, ticker, company_data, news)
            if result:
                results.append(result)

        return "\n\n---\n\n".join(results) if results else f"No data returned for {ticker}."

    def _execute_intent(
        self,
        intent: str,
        args: dict,
        ticker: str,
        company_data: dict,
        news: list,
    ) -> str:
        """Execute a single resolved intent and return its result string."""
        sup = self.supervisor

        if intent == "get_current_stock_price":
            print(f"  📊 [CurrentPriceAgent] Fetching latest price for {ticker}…")
            try:
                return _with_retry(
                    sup.call, "CurrentPrice", self.current_agent.fetch, ticker,
                    supervisor=sup, agent_name="CurrentPrice",
                )
            except CircuitOpenError:
                print("  🔄 [Supervisor] CurrentPrice circuit OPEN → falling back to last close")
                return _with_retry(
                    sup.call, "HistoricalPrice", self.historical_agent.fetch, ticker,
                    period="5d",
                    supervisor=sup, agent_name="HistoricalPrice",
                )
            except Exception as exc:
                return self._heal_and_retry(intent, ticker, exc, args)

        if intent == "get_company_info":
            print(f"  🏢 [CompanyProfileAgent] Fetching full company snapshot for {ticker}…")
            profile   = self.company_profile_agent.summarise(ticker, company_data, news)
            sentiment = self._fetch_sentiment(ticker, news)
            return f"{profile}{sentiment}"

        if intent == "get_historical_stock_prices":
            period   = args.get("period", "1mo")
            interval = args.get("interval", "1d")
            print(f"  📈 [HistoricalPriceAgent] Fetching {ticker} ({period}/{interval})…")
            try:
                return _with_retry(
                    sup.call, "HistoricalPrice", self.historical_agent.fetch,
                    ticker, period=period, interval=interval,
                    supervisor=sup, agent_name="HistoricalPrice",
                )
            except CircuitOpenError:
                print("  🔴 [Supervisor] HistoricalPrice circuit OPEN — cannot recover")
                return f"Historical data for {ticker} is temporarily unavailable. Please try again later."
            except Exception as exc:
                return self._heal_and_retry(intent, ticker, exc, args)

        if intent == "get_stock_prices_between_dates":
            start = args["start"]
            end   = args["end"]
            print(f"  📅 [DateRangeAgent] Fetching {ticker} from {start} to {end}…")
            try:
                return _with_retry(
                    sup.call, "DateRange", self.date_range_agent.fetch,
                    ticker, start=start, end=end,
                    supervisor=sup, agent_name="DateRange",
                )
            except CircuitOpenError:
                print("  🔄 [Supervisor] DateRange circuit OPEN → falling back to HistoricalPrice")
                return _with_retry(
                    sup.call, "HistoricalPrice", self.historical_agent.fetch,
                    ticker, period="1mo",
                    supervisor=sup, agent_name="HistoricalPrice",
                )
            except Exception as exc:
                return self._heal_and_retry(intent, ticker, exc, args)

        if intent == "chart_historical":
            period   = args.get("period", "1mo")
            interval = args.get("interval", "1d")
            print(f"  📊 [ChartAgent] Rendering chart for {ticker} ({period})…")
            try:
                return sup.call(
                    "ChartAgent", self.chart_agent.plot,
                    ticker, period=period, interval=interval,
                    company_data=company_data, news=news,
                )
            except CircuitOpenError:
                print("  🔄 [Supervisor] ChartAgent circuit OPEN → falling back to text summary")
                return _with_retry(
                    sup.call, "HistoricalPrice", self.historical_agent.fetch,
                    ticker, period=period, interval=interval,
                    supervisor=sup, agent_name="HistoricalPrice",
                )
            except Exception as exc:
                diag = sup.diagnose("ChartAgent", str(exc), {"ticker": ticker, "period": period})
                print(f"  ⚠️  [ChartAgent] {exc} → {diag['action']}: {diag['suggestion']}")
                if diag["action"] != "abort":
                    return _with_retry(
                        sup.call, "HistoricalPrice", self.historical_agent.fetch,
                        ticker, period=period, interval=interval,
                        supervisor=sup, agent_name="HistoricalPrice",
                    )
                raise

        if intent == "chart_date_range":
            start = args["start"]
            end   = args["end"]
            print(f"  📊 [ChartAgent] Rendering chart for {ticker} ({start} → {end})…")
            try:
                return sup.call(
                    "ChartAgent", self.chart_agent.plot,
                    ticker, start=start, end=end,
                    company_data=company_data, news=news,
                )
            except CircuitOpenError:
                print("  🔄 [Supervisor] ChartAgent circuit OPEN → falling back to text summary")
                return _with_retry(
                    sup.call, "DateRange", self.date_range_agent.fetch,
                    ticker, start=start, end=end,
                    supervisor=sup, agent_name="DateRange",
                )
            except Exception as exc:
                diag = sup.diagnose("ChartAgent", str(exc), {"ticker": ticker})
                print(f"  ⚠️  [ChartAgent] {exc} → {diag['action']}: {diag['suggestion']}")
                raise

        return f"Unrecognised intent: {intent}"

    # ── Healing helpers ───────────────────────────────────────────────────────

    def _resolve_with_healing(self, company_or_ticker: str) -> str:
        """
        Resolve a ticker; if resolution fails for a bare symbol, try
        exchange-suffix variants before giving up.
        """
        try:
            return self._resolve(company_or_ticker)
        except ValueError as exc:
            # Only attempt suffix healing for bare symbols (no period → no suffix yet)
            if "." not in company_or_ticker and _is_ticker(company_or_ticker):
                diag = self.supervisor.diagnose(
                    "TickerResolver", str(exc), {"query": company_or_ticker}
                )
                if diag["action"] == "try_variant":
                    from tools.ticker_tools import verify_ticker
                    for variant in self._ticker_variants(company_or_ticker):
                        print(f"  🔄 [Supervisor] Trying ticker variant: {variant}")
                        if verify_ticker(variant):
                            print(f"  ✅ Healed → {variant}")
                            return variant
            raise

    def _heal_and_retry(self, intent: str, ticker: str, exc: Exception, args: dict) -> str:
        """
        Called when a price agent raises an unexpected exception.

        Diagnoses the failure and either:
          • tries exchange-suffix ticker variants (try_variant)
          • tells the user to wait (wait_retry)
          • surfaces the error (abort / use_fallback)
        """
        diag = self.supervisor.diagnose(intent, str(exc), {"ticker": ticker, **args})
        action = diag["action"]
        print(f"  🩺 [Supervisor] Diagnosis → {action}: {diag['suggestion']}")

        if action == "try_variant":
            from tools.ticker_tools import verify_ticker
            for variant in self._ticker_variants(ticker):
                if not verify_ticker(variant):
                    continue
                print(f"  🔄 [Supervisor] Retrying with variant: {variant}")
                try:
                    if intent == "get_current_stock_price":
                        return _with_retry(
                            self.supervisor.call, "CurrentPrice",
                            self.current_agent.fetch, variant,
                            supervisor=self.supervisor,
                        )
                    if intent == "get_historical_stock_prices":
                        return _with_retry(
                            self.supervisor.call, "HistoricalPrice",
                            self.historical_agent.fetch, variant,
                            period=args.get("period", "1mo"),
                            supervisor=self.supervisor,
                        )
                except Exception:
                    continue

        if action == "wait_retry":
            return (
                f"⏳ The data provider is rate-limiting requests for {ticker}. "
                f"Please wait ~60 s and try again."
            )

        raise exc
