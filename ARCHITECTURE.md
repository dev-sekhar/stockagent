# Stock Price Agent — Architecture Document

> **Version:** 3.0  
> **Date:** April 2026  
> **Stack:** Python · Groq (llama-3.3-70b-versatile) · yfinance · Plotly · Polygon.io

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architectural Principles](#2-architectural-principles)
3. [Component Taxonomy](#3-component-taxonomy)
4. [Directory Structure](#4-directory-structure)
5. [Component Catalogue](#5-component-catalogue)
   - 5.1 [Infrastructure Layer](#51-infrastructure-layer)
   - 5.2 [True AI Agents](#52-true-ai-agents)
   - 5.3 [Orchestrator Agents](#53-orchestrator-agents)
   - 5.4 [Service Components](#54-service-components)
   - 5.5 [Integration Components](#55-integration-components)
   - 5.6 [Tools Layer](#56-tools-layer)
   - 5.7 [Code Review Pipeline](#57-code-review-pipeline)
6. [Data Flow Diagrams](#6-data-flow-diagrams)
   - 6.1 [Single-Stock Query Flow](#61-single-stock-query-flow)
   - 6.2 [Chart Rendering Flow](#62-chart-rendering-flow)
   - 6.3 [Multi-Stock Comparison Flow](#63-multi-stock-comparison-flow)
   - 6.4 [Company Information Flow](#64-company-information-flow)
   - 6.5 [Ticker Resolution Flow](#65-ticker-resolution-flow)
7. [Self-Healing Supervisor](#7-self-healing-supervisor)
8. [CurveSpec Contract](#8-curvespec-contract)
9. [LLM Usage Patterns](#9-llm-usage-patterns)
10. [Resilience & Error Handling](#10-resilience--error-handling)
11. [Code Review Pipeline](#11-code-review-pipeline)
12. [Governance & Observability](#12-governance--observability)
13. [Configuration & Environment](#13-configuration--environment)
14. [Design Decisions & Trade-offs](#14-design-decisions--trade-offs)

---

## 1. System Overview

The Stock Price Agent is a **multi-agent AI system** that answers natural-language stock queries. Users can ask for current prices, historical data, interactive charts, multi-stock comparisons, and full company profiles — all from a single conversational CLI.

```
User ──► main.py ──► Orchestrator ──► [Resolution Pipeline]
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                     Price Agents    Chart Agents    Company Agents
                          │                │                │
                          └────────────────┼────────────────┘
                                           ▼
                                    SelfHealingSupervisor
                                    (circuit breakers, retry, diagnosis)
```

**Key design goals:**

| Goal | How achieved |
|------|-------------|
| Single responsibility | Every class/file has exactly one job |
| True AI agents | LLM-powered reasoning for any decision requiring interpretation |
| Deterministic tools | All data fetching is pure Python — no LLM in the hot path |
| Resilience | Circuit breakers + fallback chains + LLM self-diagnosis |
| Observability | Per-agent health tracking, `health` CLI command |
| Extensibility | New data sources = new Tool; new query types = new Agent + routing tool |

---

## 2. Architectural Principles

### 2.1 Task Decomposition & Specialisation
Each component has a **single, named responsibility**. A component that fetches data does not render; a component that renders does not reason; a component that reasons does not store state.

### 2.2 Strict Layer Separation

```
  CLI (main.py)
    │
  Orchestrator  ←── routes intent, owns resilience policy
    │
  AI Agents     ←── LLM-powered reasoning and decisions
    │
  Tools         ←── pure data fetch / compute, no LLM
    │
  External APIs ←── yfinance, Groq, Polygon.io
```

### 2.3 Communication Protocol — CurveSpec
All data flowing between tools and the rendering layer travels as **CurveSpec** objects — a typed contract that decouples producers from consumers. See §8.

### 2.4 AI Agent Definition
A class qualifies as a **True AI Agent** only if it:
- Uses the Groq LLM for at least one **decision** (not just formatting)
- Has a goal it pursues autonomously (the caller does not direct individual steps)
- Can handle novel inputs gracefully using LLM reasoning

Pure data fetchers, renderers, and format converters are **Tools** or **Services**, not agents.

### 2.5 Naming Conventions

| Suffix / Location | Meaning |
|---|---|
| `*Agent` in `agents/` | True AI agent (MUST use Groq) |
| `*Tool` in `tools/` | Pure data fetch or compute (NO LLM) |
| `ChartCompiler`, `DataValidator` | Service — pure logic, no LLM, lives in `agents/` |
| `orchestrator.py`, `supervisor.py` | Infrastructure — project root |

### 2.6 Scalability & Robustness
- **RateLimiter**: token-bucket, capped at 30 Groq calls/minute
- **Retry**: error-type-aware back-off (60 s for rate limits, 2× for others)
- **Circuit breaker**: opens after 3 consecutive failures; probes after recovery timeout
- **Fallback chains**: every agent has a deterministic fallback if it is unavailable

---

## 3. Component Taxonomy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE                                                               │
│  orchestrator.py          supervisor.py                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  TRUE AI AGENTS  (agents/) — use Groq LLM                                    │
│  CompanySearchAgent        CandidateFilterAgent    TickerResolverAgent        │
│  ExchangeDisambiguatorAgent  PolygonSearchAgent*   CurrentPriceAgent          │
│  HistoricalPriceAgent      DateRangeAgent          SectorIndexAgent           │
│  NewsAgent                 CompanyProfileAgent                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATOR AGENTS  (agents/) — coordinate, NO LLM                         │
│  ChartAgent                ComparisonChartAgent                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  SERVICES  (agents/) — pure logic, NO LLM                                    │
│  ChartCompiler             DataValidator                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  TOOLS  (tools/) — pure data fetch / compute, NO LLM                         │
│  StockCurveTool            IndexCurveTool          SpreadCurveTool            │
│  CompanyDataTool           price_tools             ticker_tools               │
│  history_tools             CurveSpec                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  CODE REVIEW  (review/)                                                       │
│  BaselineChecker (MCP)     ProjectReviewAgent      HybridReviewOrchestrator   │
└──────────────────────────────────────────────────────────────────────────────┘

* PolygonSearchAgent: legacy HTTP wrapper — named Agent for historical reasons
```

---

## 4. Directory Structure

```
stock-agent/
│
├── main.py                     # CLI entry point
├── orchestrator.py             # Central query router + self-healing supervisor
├── supervisor.py               # CircuitBreaker + AgentHealth + SelfHealingSupervisor
├── review.py                   # Code review CLI entry point
│
├── agents/                     # All agent/service classes
│   ├── base_agent.py           # Groq tool-calling loop — all AI agents inherit this
│   ├── curve_spec.py           # Backward-compat shim → tools/curve_spec.py
│   │
│   │── Resolution pipeline
│   ├── company_search.py       # [AI] Web-search for company info
│   ├── polygon_search.py       # [Integration] Polygon.io ticker search
│   ├── candidate_filter.py     # [AI] Filter/rank ticker candidates
│   ├── ticker_resolver.py      # [AI] Resolve ambiguous ticker to canonical form
│   ├── exchange_disambiguator.py # [AI] Choose correct exchange when multiple match
│   │
│   │── Price agents
│   ├── current_price.py        # [AI] Fetch + format latest traded price
│   ├── historical_price.py     # [AI] Fetch + format OHLCV over a period
│   ├── date_range.py           # [AI] Fetch + format OHLCV between two dates
│   │
│   │── Chart pipeline
│   ├── sector_index.py         # [AI] Map stock → sector index (e.g. Nifty Pharma)
│   ├── chart_agent.py          # [Orchestrator] Single-stock chart coordinator
│   ├── comparison_chart_agent.py # [Orchestrator] Multi-stock comparison coordinator
│   ├── chart_compiler.py       # [Service] Plotly renderer + company panel injection
│   │
│   │── Company enrichment
│   ├── news_agent.py           # [AI] Curate trusted-source news via LLM
│   ├── company_profile.py      # [AI] One-shot LLM narration of company snapshot
│   │
│   │── Utilities
│   ├── data_validator.py       # [Service] Validate + clean OHLCV data
│   ├── industry_index.py       # [Helper] Industry → index lookup table
│   ├── _setup_curves.py        # [Internal] Shared curve-setup helpers
│   └── spread_curve.py         # [Shim] Backward-compat → tools/spread_curve.py
│
├── tools/                      # Pure data-fetch and compute
│   ├── curve_spec.py           # CurveSpec dataclass — universal data contract
│   ├── stock_curve.py          # StockCurveTool — yfinance OHLCV → CurveSpec
│   ├── index_curve.py          # IndexCurveTool — yfinance index → CurveSpec
│   ├── spread_curve.py         # SpreadCurveTool — compute price spread (arbitrage)
│   ├── company_data.py         # CompanyDataTool — yfinance profile + financials
│   ├── price_tools.py          # Low-level price fetch helpers
│   ├── history_tools.py        # Low-level historical data helpers
│   └── ticker_tools.py         # Ticker search + validation helpers
│
├── review/                     # Hybrid code review pipeline
│   ├── __init__.py
│   ├── rules.py                # Single source of truth — project architectural rules
│   ├── mcp_baseline.py         # MCP layer — pylint/flake8/bandit via subprocess
│   ├── project_reviewer.py     # [AI] ProjectReviewAgent — architectural compliance
│   └── review_orchestrator.py  # HybridReviewOrchestrator — merge baseline + AI
│
├── requirements.txt            # groq yfinance plotly pandas tabulate requests etc.
├── .env.example                # GROQ_API_KEY template
└── ARCHITECTURE.md             # This document
```

---

## 5. Component Catalogue

### 5.1 Infrastructure Layer

#### `orchestrator.py` — Orchestrator (Central Query Router)

**Role:** The single entry point for all user queries. Routes to sub-agents, manages the complete resilience policy.

**Responsibilities:**
- LLM-based intent classification via `ROUTING_TOOLS` (meta-tool pattern)
- Entity extraction: company name / ticker, period, dates
- Ticker resolution with exchange-suffix healing (`INFY` → `INFY.NS` → `INFY.BO` → …)
- Intent clarification loop (asks user when ambiguous)
- Comparison mode detection and delegation
- Company snapshot enrichment for price and chart responses
- Health reporting

**Key methods:**

| Method | Description |
|--------|-------------|
| `handle(query)` | Main entry point — full pipeline |
| `_resolve_with_healing(raw)` | Ticker resolution with suffix variants |
| `_clarify_intent(ticker, intent, args)` | Interactive disambiguation |
| `_run_comparison(tickers)` | Multi-stock comparison pipeline |
| `_fetch_company_snapshot(ticker)` | Best-effort company data + news (non-blocking) |
| `_heal_and_retry(intent, ticker, exc, args)` | LLM-diagnosed recovery |
| `health_report()` | Markdown table of all agent circuit states |

**Routing tools (meta-tools):**

| Tool name | Trigger |
|-----------|---------|
| `get_current_stock_price` | "current price", "right now", "trading at" |
| `get_historical_stock_prices` | "last year", "3 months", "ytd", period codes |
| `get_stock_prices_between_dates` | "from Jan to June", explicit date ranges |
| `get_company_info` | "tell me about", "company profile", "financials" |

---

#### `supervisor.py` — SelfHealingSupervisor (Infrastructure Service)

**Role:** Circuit-breaker fabric that wraps every agent call. **Not an AI agent** — it is a deterministic infrastructure service with a single LLM touch-point (`diagnose()`).

**Components:**

| Class | Description |
|-------|-------------|
| `CircuitState` | Enum: `CLOSED` / `OPEN` / `HALF_OPEN` |
| `AgentHealth` | Per-agent state: consecutive failures, last error, circuit timestamps |
| `CircuitBreaker` | Callable wrapper — checks health, records outcome |
| `SelfHealingSupervisor` | Registry of breakers; `call()`, `diagnose()`, `health_report()` |

**Circuit-breaker state machine:**
```
CLOSED ──[3 consecutive failures]──► OPEN
  ▲                                    │
  │                              [recovery_timeout]
  │                                    ▼
  └──────[probe succeeds]────── HALF_OPEN
                  (probe fails → back to OPEN)
```

**Error classification → recovery strategy:**

| Pattern | Strategy |
|---------|----------|
| 413, rate_limit, TPM | `wait_retry` (60 s) |
| timeout, connection error | `wait_retry` (2× backoff) |
| 401, 403, api_key | `use_fallback` |
| no data, not found, 404 | `try_variant` |
| other | `try_variant` |

---

### 5.2 True AI Agents

All inherit `BaseAgent` and use Groq's tool-calling loop.

#### `BaseAgent` (`agents/base_agent.py`)
Foundation class. Drives the Groq `chat.completions` tool-calling loop:
1. Send `system` + `user` messages
2. If response contains tool calls → execute each → append results → loop
3. If response is plain text → return it

Subclasses set: `system_prompt`, `tools` (Groq format), `tool_fn_map`.

---

#### Resolution Pipeline

| Agent | File | LLM role |
|-------|------|----------|
| `CompanySearchAgent` | `company_search.py` | Decides search strategy; parses web results to extract company identity |
| `PolygonSearchAgent` | `polygon_search.py` | Calls Polygon.io API; legacy HTTP wrapper (named Agent historically) |
| `CandidateFilterAgent` | `candidate_filter.py` | Ranks candidate tickers; eliminates duplicates/ADRs/wrong markets |
| `TickerResolverAgent` | `ticker_resolver.py` | Final arbitration: picks canonical ticker from filtered candidates |
| `ExchangeDisambiguatorAgent` | `exchange_disambiguator.py` | Resolves same-company on multiple exchanges vs. different companies matching same query; tries LLM auto-select before presenting menu |

**Resolution flow:**
```
raw input
    │
    ├─[already a ticker]──► suffix-heal if needed ──► canonical ticker
    │
    └─[company name]──► CompanySearch + PolygonSearch
                                │
                         CandidateFilter
                                │
                         ExchangeDisambiguator
                                │
                        canonical ticker (e.g. INFY.NS)
```

---

#### Price Agents

| Agent | File | LLM role | Fallback |
|-------|------|----------|----------|
| `CurrentPriceAgent` | `current_price.py` | Formats price with currency, exchange, market status | HistoricalPriceAgent (5d) |
| `HistoricalPriceAgent` | `historical_price.py` | Narrates OHLCV summary; decides which stats are worth highlighting | DateRangeAgent |
| `DateRangeAgent` | `date_range.py` | Same as Historical but for explicit date windows | HistoricalPriceAgent (1mo) |

---

#### Chart Intelligence

| Agent | File | LLM role |
|-------|------|----------|
| `SectorIndexAgent` | `sector_index.py` | Maps stock ticker → industry sector → appropriate sector index. Fast path: keyword lookup. LLM fallback: free-text sector string → known sector key. Cross-exchange fallback if same-exchange index unavailable. |

---

#### Company Enrichment

| Agent | File | LLM role |
|-------|------|----------|
| `NewsAgent` | `news_agent.py` | Filters yfinance news to trusted publishers; LLM selects + summarises 5 most investor-relevant items |
| `CompanyProfileAgent` | `company_profile.py` | One-shot LLM narration: turns raw `CompanyDataTool` dict + news list into formatted investor-facing markdown snapshot |

---

### 5.3 Orchestrator Agents

These coordinate tools — no LLM involved.

#### `ChartAgent` (`agents/chart_agent.py`)
**Decides what to fetch, delegates everything.**

Pipeline:
1. `StockCurveTool` → OHLCV CurveSpec for the stock
2. `_detect_index()` → broad-market benchmark for the stock's exchange
3. `IndexCurveTool` → benchmark CurveSpec
4. `SectorIndexAgent` → sector index (if available and different from benchmark)
5. `IndexCurveTool` → sector index CurveSpec (if found)
6. `ChartCompiler.compile([stock, benchmark, sector], company_data, news)`

**Exchange → index mapping (built-in):**

| Suffix | Index | Name |
|--------|-------|------|
| `.NS` | `^NSEI` | Nifty 50 |
| `.BO` | `^BSESN` | Sensex |
| `.L` | `^FTSE` | FTSE 100 |
| `.T` | `^N225` | Nikkei 225 |
| `.HK` | `^HSI` | Hang Seng |
| `.AX` | `^AXJO` | ASX 200 |
| *(default)* | `^GSPC` | S&P 500 |

---

#### `ComparisonChartAgent` (`agents/comparison_chart_agent.py`)
**Multi-stock comparison mode.**

Pipeline:
1. For each ticker: `StockCurveTool` → CurveSpec
2. For each unique exchange: `IndexCurveTool` → benchmark CurveSpec
3. If arbitrage mode (same base ticker on ≥2 exchanges): `SpreadCurveTool` → spread CurveSpec
4. `ChartCompiler.compile(all_curves)` — sector indices excluded (too cluttered)

**Arbitrage detection:** same ticker root (e.g. `INFY`) on 2+ exchange suffixes → spreads are computed and plotted.

---

### 5.4 Service Components

#### `ChartCompiler` (`agents/chart_compiler.py`)
**Sole Plotly rendering engine.**

Responsibilities:
- Receive list of `CurveSpec` objects
- Align date axes (inner join across all series)
- Rebase all series to 100 (indexed comparison) or use raw prices (arbitrage/spread mode)
- Assign colours from palettes (stock, benchmark, sector, spread)
- Determine chart colour (green/red/amber) based on start vs end price
- Build Plotly subplots: top panel (prices, indexed) + bottom panel (volume or spread)
- Add crosshair cursor with unified hover (shows all values at cursor position)
- Inject company data panel as dark-themed HTML below the chart
- Save HTML, open in browser

**Colour logic:**

| Condition | Curve colour |
|-----------|-------------|
| End price > start price | Green (`#00c851`) |
| End price < start price | Red (`#e74c3c`) |
| No change | Amber (`#f0a500`) |

---

#### `DataValidator` (`agents/data_validator.py`)
Validates and cleans OHLCV data before it reaches price agents:
- Checks for required columns
- Removes rows with null prices
- Detects and flags suspicious zero-volume days
- Returns cleaned DataFrame + validation report

---

### 5.5 Integration Components

#### `PolygonSearchAgent` (`agents/polygon_search.py`)
HTTP wrapper for the Polygon.io `/v3/reference/tickers` endpoint.
- Searches by name keyword or ticker prefix
- Returns structured candidate list (ticker, name, exchange, type)
- Rate-limit aware; registered with supervisor circuit breaker
- Note: named `Agent` for historical reasons; does not use Groq

---

### 5.6 Tools Layer

All tools are pure Python — no LLM, no side effects beyond network I/O.

#### `CurveSpec` (`tools/curve_spec.py`)
Universal data contract between tools and `ChartCompiler`.

```python
@dataclass
class CurveSpec:
    ticker:    str           # e.g. "INFY.NS"
    label:     str           # Display name e.g. "Infosys"
    role:      str           # "stock" | "benchmark" | "sector" | "spread"
    dates:     list[str]     # ISO date strings
    closes:    list[float]   # Adjusted close prices
    volumes:   list[int]     # Daily volumes (0 for indices)
    currency:  str           # e.g. "INR"
    error:     str | None    # Non-None if fetch failed
```

---

#### `StockCurveTool` (`tools/stock_curve.py`)
- Fetches OHLCV via yfinance
- Returns `CurveSpec(role='stock')`
- Handles period/date-range modes
- Cleans split-adjusted prices

#### `IndexCurveTool` (`tools/index_curve.py`)
- Fetches index OHLCV via yfinance
- Returns `CurveSpec(role='benchmark')` or `CurveSpec(role='sector')`
- Always sets `volumes=[]` (indices have no volume)

#### `SpreadCurveTool` (`tools/spread_curve.py`)
- Takes two `CurveSpec` objects (same stock, different exchanges)
- Computes percentage price spread: `(price_A / price_B - 1) × 100`
- Returns `CurveSpec(role='spread')`
- Used in arbitrage mode by `ComparisonChartAgent`

#### `CompanyDataTool` (`tools/company_data.py`)
- Pure yfinance `Ticker.info` fetch
- Returns structured dict with three sections:

```
{
  "profile":      { name, sector, industry, country, exchange, description }
  "fundamentals": { market_cap, pe_trailing, pe_forward, eps_ttm, 52w_high/low,
                    beta, dividend_yield, avg_volume_10d }
  "financials":   { revenue_ttm, gross_margin, operating_margin, net_margin,
                    total_debt, free_cash_flow, revenue_growth, earnings_growth }
  "error":        null | "error message"
}
```

- NaN/Inf → `None` (JSON-safe), percentages converted from 0-1 fractions

#### `price_tools.py`, `history_tools.py`, `ticker_tools.py`
Low-level helpers used by price agents and the resolution pipeline.

---

### 5.7 Code Review Pipeline

See §11 for full detail.

| Component | File | Role |
|-----------|------|------|
| `rules.py` | `review/rules.py` | Single source of truth — all architectural rules + LLM system prompt |
| `BaselineChecker` | `review/mcp_baseline.py` | MCP layer: runs pylint, flake8, bandit via subprocess |
| `ProjectReviewAgent` | `review/project_reviewer.py` | AI layer: architectural compliance check |
| `HybridReviewOrchestrator` | `review/review_orchestrator.py` | Merges both layers → unified markdown report |

---

## 6. Data Flow Diagrams

### 6.1 Single-Stock Query Flow

```
User: "Infosys current price"
        │
        ▼
main.py → orchestrator.handle()
        │
        ├─ Step 1: Groq LLM (ROUTING_TOOLS)
        │    → intent: get_current_stock_price
        │    → company_or_ticker: "Infosys"
        │
        ├─ Step 2: _resolve_with_healing("Infosys")
        │    ├─ not a bare ticker → ExchangeDisambiguatorAgent
        │    │    ├─ CompanySearchAgent (web search)
        │    │    ├─ PolygonSearchAgent (Polygon.io)
        │    │    ├─ CandidateFilterAgent (rank candidates)
        │    │    └─ LLM auto-select → "INFY.NS" ✓
        │    └─ returns "INFY.NS"
        │
        ├─ Step 3: skip clarification (intent clear)
        │
        ├─ Step 4: supervisor.call("CurrentPrice", current_agent.fetch, "INFY.NS")
        │    └─ CurrentPriceAgent → "INFY.NS: ₹1,847.30 (+1.2%)"
        │
        ├─ Step 5: _fetch_company_snapshot("INFY.NS")
        │    ├─ supervisor.call("CompanyData", company_data_tool.fetch, "INFY.NS")
        │    └─ supervisor.call("NewsAgent", news_agent.fetch, "INFY.NS")
        │
        ├─ Step 6: company_profile_agent.summarise("INFY.NS", company_data, news)
        │    └─ One-shot LLM → formatted markdown snapshot
        │
        └─ Return: price text + "---" + company snapshot
```

---

### 6.2 Chart Rendering Flow

```
User: "INFY.NS 1 year chart"
        │
        ▼
orchestrator → intent: chart_historical, period: 1y
        │
        ├─ _fetch_company_snapshot("INFY.NS")   ← parallel best-effort
        │
        ├─ supervisor.call("ChartAgent", chart_agent.plot, "INFY.NS",
        │                  period="1y", company_data=..., news=...)
        │
        └─ ChartAgent.plot()
             │
             ├─ StockCurveTool("INFY.NS", period="1y")
             │    └─ CurveSpec(role="stock", ticker="INFY.NS", ...)
             │
             ├─ _detect_index("INFY.NS") → (^NSEI, "Nifty 50")
             │
             ├─ IndexCurveTool("^NSEI", period="1y")
             │    └─ CurveSpec(role="benchmark", ticker="^NSEI", ...)
             │
             ├─ SectorIndexAgent.find("INFY.NS")
             │    ├─ yfinance sector → "Technology"
             │    ├─ keyword map → ("^CNXIT", "Nifty IT")   [.NS suffix match]
             │    └─ IndexCurveTool("^CNXIT") → CurveSpec(role="sector", ...)
             │
             └─ ChartCompiler.compile(
                    curves=[stock_spec, benchmark_spec, sector_spec],
                    company_data={...}, news=[...])
                  │
                  ├─ Align dates (inner join)
                  ├─ Rebase all to 100
                  ├─ Build Plotly traces
                  ├─ Add crosshair + unified hover
                  ├─ Colour stock curve (green/red/amber)
                  ├─ Build dark HTML company panel
                  ├─ fig.to_html() + inject panel before </body>
                  └─ Save INFY_NS_1y.html, open browser
```

---

### 6.3 Multi-Stock Comparison Flow

```
User: "compare Infosys vs Wipro vs HCL"
        │
        ▼
orchestrator → _is_comparison_query() → True
        │
        └─ _run_comparison(initial_tickers=["INFY.NS"])
             │
             ├─ Prompt user for all tickers (up to 5)
             ├─ Resolve each via ExchangeDisambiguatorAgent
             │
             └─ ComparisonChartAgent.plot(tickers, period)
                  │
                  ├─ For each ticker:
                  │    StockCurveTool → CurveSpec(role="stock")
                  │
                  ├─ Deduplicate exchanges → unique benchmarks
                  │    IndexCurveTool → CurveSpec(role="benchmark") per exchange
                  │
                  ├─ Arbitrage check (same base ticker, ≥2 exchanges)?
                  │    Yes → SpreadCurveTool → CurveSpec(role="spread")
                  │
                  └─ ChartCompiler.compile(all_curves)
                       # sector indices NOT included in comparison mode
```

---

### 6.4 Company Information Flow

```
User: "tell me about Reliance Industries"
        │
        ▼
orchestrator → intent: get_company_info
        │
        ├─ _resolve_with_healing("Reliance Industries")
        │    └─ returns "RELIANCE.NS"
        │
        ├─ skip clarification gate (get_company_info always bypasses)
        │
        └─ _fetch_company_snapshot("RELIANCE.NS")
             │
             ├─ CompanyDataTool.fetch("RELIANCE.NS")
             │    └─ yfinance Ticker.info → { profile, fundamentals, financials }
             │
             ├─ NewsAgent.fetch("RELIANCE.NS")
             │    ├─ yfinance Ticker.news → raw list
             │    ├─ filter to _TRUSTED_SOURCES
             │    └─ LLM → top 5 investor-relevant items
             │
             └─ CompanyProfileAgent.summarise("RELIANCE.NS", data, news)
                  └─ One-shot LLM → full investor snapshot markdown
```

---

### 6.5 Ticker Resolution Flow

```
Input: "hindustan"
        │
        ├─ _is_ticker("hindustan") → False (lowercase, >6 chars)
        │
        └─ ExchangeDisambiguatorAgent.resolve("hindustan")
             │
             ├─ CompanySearchAgent.search("hindustan")  [web search]
             ├─ PolygonSearchAgent.search("hindustan")  [Polygon.io API]
             │
             ├─ Merge candidates: [HUL.NS, HINDUNILVR.NS, HINDZINC.NS,
             │                     HINDPETRO.NS, HAL.NS, ...]
             │
             ├─ CandidateFilterAgent.filter(candidates, "hindustan")
             │    → removes duplicates, ADRs, foreign listings
             │
             ├─ LLM auto-select: ambiguous (multiple valid companies) → null
             │
             └─ Present numbered menu to user:
                  1. Hindustan Unilever (HINDUNILVR.NS)
                  2. Hindustan Zinc (HINDZINC.NS)
                  3. Hindustan Petroleum (HINDPETRO.NS)
                  ...
                  User picks → canonical ticker
```

---

## 7. Self-Healing Supervisor

The `SelfHealingSupervisor` in `supervisor.py` protects every agent call.

### Registered agents and their thresholds

| Agent name | Failure threshold | Recovery timeout |
|------------|:-----------------:|:----------------:|
| Polygon | 3 | 120 s |
| CompanySearch | 3 | 60 s |
| CurrentPrice | 3 | 120 s |
| HistoricalPrice | 3 | 120 s |
| DateRange | 3 | 120 s |
| ChartAgent | 3 | 60 s |
| CompanyData | 3 | 120 s |
| NewsAgent | 3 | 60 s |

### Fallback chains

| Primary | Fallback |
|---------|---------|
| `CurrentPrice` OPEN | `HistoricalPrice` (5d period) |
| `HistoricalPrice` OPEN | `DateRange` (1mo window) |
| `DateRange` OPEN | `HistoricalPrice` (1mo period) |
| `ChartAgent` OPEN | Text summary via `HistoricalPrice` |

### Exchange-suffix healing variants

When bare ticker resolution fails, the orchestrator tries these suffixes in order:
```
INFY → INFY.NS → INFY.BO → INFY.L → INFY.AX → INFY.TO → INFY.HK → INFY.T
```

### LLM self-diagnosis

On unclassified errors, `SelfHealingSupervisor.diagnose()` makes a zero-shot LLM call to classify the error and suggest a recovery action. Falls back to rule-based classification if the LLM is unavailable.

---

## 8. CurveSpec Contract

`CurveSpec` (defined in `tools/curve_spec.py`) is the **only** data format that flows between tools and `ChartCompiler`. This decoupling means:
- Tools can be swapped without touching `ChartCompiler`
- `ChartCompiler` can render any data source that produces `CurveSpec`

### Role values and their chart mapping

| `role` | Subplot | Default colour | Description |
|--------|---------|---------------|-------------|
| `"stock"` | Top (price) | Palette (`#00b4d8`, …) | Primary asset price |
| `"benchmark"` | Top (price) | `#a0a0c8` (muted blue-grey) | Exchange-level index |
| `"sector"` | Top (price) | `#f0a500` (amber) | Sector/industry index |
| `"spread"` | Bottom | `#e74c3c` (red) | Price spread % (arbitrage) |

### Volume vs spread in bottom panel

- **Normal / comparison mode:** volume bars (bar chart, semi-transparent)
- **Arbitrage mode** (spread CurveSpec present): spread % bars instead of volume

---

## 9. LLM Usage Patterns

The system uses three distinct LLM patterns, each in the appropriate place:

### Pattern 1: Tool-Calling Loop (BaseAgent)
Used by: `CompanySearchAgent`, `CandidateFilterAgent`, `ExchangeDisambiguatorAgent`, `NewsAgent`, price agents

```
System prompt (role + rules)
    → User message (query or data)
    → LLM → tool call
    → Execute tool → feed result back
    → LLM → ... (loop until text response)
    → Return final text
```

**When to use:** Agent needs to decide WHAT action to take, then act on the result, potentially across multiple steps.

### Pattern 2: One-Shot LLM (direct Groq call)
Used by: `CompanyProfileAgent`, `SectorIndexAgent` (fallback), `SelfHealingSupervisor.diagnose()`

```
System prompt + rich user message (pre-assembled data)
    → Single LLM call
    → Parse structured response or markdown
```

**When to use:** All data is already available; LLM's only job is narration, classification, or selection.

### Pattern 3: Meta-Tool Routing (Orchestrator)
Used by: `Orchestrator.handle()`

```
System prompt (routing instructions)
    → User message (raw query)
    → LLM → exactly one tool call (intent declaration)
    → Parse intent + args → dispatch to sub-agents
```

**When to use:** Routing/intent classification where the LLM acts as a deterministic classifier.

### Token management

| Agent | Strategy |
|-------|----------|
| `ProjectReviewAgent` | Files ≤250 lines sent in full; larger files → skeleton (imports + signatures). `read_file_section` capped at 100 lines; `read_related_file` truncated at 120 lines |
| `NewsAgent` | Sends at most 15 items to LLM for curation |
| `CompanyProfileAgent` | Sends at most 5 news items; `max_tokens=1600` |
| `SelfHealingSupervisor.diagnose()` | Error truncated to 300 chars; context to 400 chars; `max_tokens=200` |

---

## 10. Resilience & Error Handling

### Retry policy (`_with_retry` in orchestrator.py)

| Error type | Wait time | Max attempts |
|-----------|-----------|:------------:|
| Rate limit (413, TPM) | 60 s | 3 |
| Other transient | 2× exponential | 3 |
| Unrecoverable (auth) | N/A | 1 (abort) |

### Token limit (rate limiter)
Token-bucket algorithm: 30 calls/minute, enforced per-`Orchestrator` instance via `threading.Lock`.

### Graceful degradation

| Failure | Degraded response |
|---------|------------------|
| `CompanyDataTool` fails | Price response without company panel |
| `NewsAgent` fails | Company profile without news section |
| `SectorIndexAgent` returns None | Chart with exchange benchmark only |
| `ChartAgent` circuit OPEN | Plain text historical price summary |
| All resolution fails | User-visible error with suggestion |
| LLM self-diagnosis fails | Rule-based classification fallback |

---

## 11. Code Review Pipeline

The `review/` package provides a **hybrid code review** combining automated static analysis (MCP layer) with AI architectural review.

```
review.py <path> [--baseline-only] [--agent-only] [--output report.md]
        │
        └─ HybridReviewOrchestrator
             │
             ├─ BaselineChecker (MCP layer)
             │    ├─ pylint  → style, errors, dead code
             │    ├─ flake8  → PEP-8 compliance
             │    └─ bandit  → security vulnerabilities
             │
             └─ ProjectReviewAgent (AI layer)
                  ├─ reads rules.py (single source of truth)
                  ├─ reads file skeleton (or full file if ≤250 lines)
                  ├─ tools: read_file_section, read_related_file
                  └─ checks architectural compliance:
                       • correct agent/tool/service taxonomy
                       • CurveSpec contract adherence
                       • Groq injection pattern
                       • import conventions
                       • forbidden patterns (print in agents, hardcoded keys)
```

### `rules.py` — Architectural rule registry
Single source of truth for all project rules. Sections:
- `ARCHITECTURE_TAXONOMY` — authoritative class → file mapping
- `NAMING_CONVENTIONS` — Agent/Tool/Service naming rules
- `CURVESPEC_CONTRACT` — data exchange rules
- `GROQ_INJECTION_PATTERN` — how Groq client must be passed
- `IMPORT_CONVENTIONS` — allowed cross-layer imports
- `FORBIDDEN_PATTERNS` — what must never appear in each layer
- `GROQ_TOKEN_LIMITS` — per-agent token budgets
- `SYSTEM_PROMPT` — combined prompt used by `ProjectReviewAgent`

---

## 12. Governance & Observability

### Health monitoring
```
$ python main.py
You: health

### 🏥 Agent Health
| Agent         | State          | Calls | Failures | Last Error |
|---------------|----------------|------:|---------:|------------|
| Polygon       | 🟢 closed      |    12 |        0 |            |
| CompanySearch | 🟢 closed      |     8 |        1 | `413 ...`  |
| CurrentPrice  | 🟢 closed      |    15 |        0 |            |
| ChartAgent    | 🔴 open (45s)  |     3 |        3 | `no data`  |
```

### Logging
All agents print structured emoji-prefixed status lines:
```
  🧭 [Orchestrator]   Intent: chart_historical | Args: {...}
  🔍 [CompanySearch]  Searching for "hindustan"...
  📊 [ChartAgent]     Rendering chart for INFY.NS (1y)...
  🏢 [CompanyProfile] Fetching company snapshot for INFY.NS...
  ⚠️  [CompanyData]   yfinance error: no data found for XYZ
  🔴 [CircuitBreaker] ChartAgent → OPEN (3 consecutive failures)
  🟡 [CircuitBreaker] ChartAgent → HALF_OPEN (recovery probe)
  ✅ [CircuitBreaker] ChartAgent → CLOSED (recovered)
  🔄 [Supervisor]     CurrentPrice circuit OPEN → falling back to last close
```

### Architectural governance via review.py
```bash
# Full review (baseline + AI)
python review.py agents/chart_agent.py

# Baseline only (linting/security)
python review.py agents/ --baseline-only

# AI architectural review only
python review.py agents/sector_index.py --agent-only

# Save report
python review.py agents/ --output review_report.md
```

---

## 13. Configuration & Environment

### Environment variables (`.env`)

| Variable | Required | Description |
|----------|:--------:|-------------|
| `GROQ_API_KEY` | ✅ | Groq Cloud API key (llama-3.3-70b-versatile) |
| `POLYGON_API_KEY` | ⬜ | Polygon.io key (ticker search fallback; free tier sufficient) |

### LLM model
All agents use `llama-3.3-70b-versatile` via Groq. The model is set in `BaseAgent.model` and `orchestrator.py:MODEL`. Changing both constants switches the whole system.

### Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `groq` | LLM API client |
| `yfinance` | Market data (prices, fundamentals, news) |
| `plotly` | Interactive HTML charts |
| `pandas` | Date alignment, OHLCV manipulation |
| `python-dotenv` | `.env` loading |
| `tabulate` | Text table formatting |
| `requests` | HTTP for Polygon.io + web search |
| `matplotlib` | Static PNG fallback (legacy) |

---

## 14. Design Decisions & Trade-offs

### Why Groq / llama-3.3-70b-versatile?
Fast inference (sub-second on simple calls), generous free tier, strong tool-calling support. The 12,000 TPM free-tier limit is managed by the RateLimiter + 60 s back-off for 413 errors.

### Why yfinance as primary data source?
Zero cost, covers global markets, returns fundamentals + news alongside price data. Limitations: unofficial Yahoo Finance scraper (can break), no real-time quotes (15 min delay for most markets). Polygon.io is the fallback for ticker resolution.

### Why CurveSpec instead of raw DataFrames?
DataFrames passed between layers create implicit coupling (column name assumptions, index type assumptions). CurveSpec is a typed, immutable contract — adding a new tool never breaks the renderer.

### Why ChartCompiler is a Service, not an Agent?
Rendering is deterministic: given the same CurveSpecs, the output is always the same HTML. There is no decision to be made, so no LLM is needed. Calling it an Agent would violate the taxonomy and inflate token usage.

### Why one-shot LLM for CompanyProfileAgent vs tool loop?
All data arrives pre-assembled from `CompanyDataTool` + `NewsAgent`. The LLM's only job is narration — no tool calls are needed, so the loop overhead would waste tokens and latency.

### Why supervisor.py is not in agents/?
`SelfHealingSupervisor` is infrastructure, not an AI agent. It does not pursue a goal, cannot handle novel queries, and its core behaviour (circuit tracking) is deterministic. The single LLM call in `diagnose()` is a classification utility, not autonomous reasoning.

### Why sector indices are excluded from comparison mode?
With 3+ stocks from different sectors, adding sector indices makes the chart unreadable (too many curves, different denominators). Exchange-level benchmarks are still included as they provide meaningful context for all listed stocks.

### Comparison vs arbitrage mode
If the user requests the same base ticker (e.g. `INFY`) across multiple exchanges (`.NS` and `.BO`), the system detects this automatically and adds a spread curve to show the price differential — enabling direct arbitrage analysis without user configuration.

---

*This document reflects the system as of v3.0. Update §5 (Component Catalogue) and `review/rules.py` whenever new agents or tools are added.*
