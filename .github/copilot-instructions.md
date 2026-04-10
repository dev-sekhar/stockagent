# GitHub Copilot Instructions — Stock Price Agent

## Project Overview

A multi-agent CLI system that answers natural-language stock queries.
Natural-language queries are routed by the **Orchestrator** to small,
single-purpose agents powered by **Groq LLaMA 3.3 70B** (`llama-3.3-70b-versatile`)
and **yfinance** for market data.

## Architecture

```
User query
    │
    ▼
Orchestrator          ← LLM classifies intent via ROUTING_TOOLS, extracts entities
    │
    ├─ ExchangeDisambiguatorAgent  ← company name → verified ticker (LLM + ask_user tool)
    ├─ TickerResolverAgent         ← simpler ticker lookup via search_ticker tool
    │
    ├─ CurrentPriceAgent           ← latest traded price
    ├─ HistoricalPriceAgent        ← OHLCV for relative periods (1mo, 6mo, 1y …)
    ├─ DateRangeAgent              ← OHLCV between specific start/end dates
    │
    ├─ ChartAgent                  ← single-stock interactive HTML chart
    │       ├─ StockCurveTool      ← fetches OHLCV into CurveSpec
    │       ├─ IndexCurveTool      ← fetches benchmark/sector index into CurveSpec
    │       ├─ SectorIndexAgent    ← AI decides which sector index to overlay
    │       └─ ChartCompiler       ← rebases to 100, renders Plotly HTML
    ├─ ComparisonChartAgent        ← multi-ticker side-by-side chart
    │
    ├─ CompanyProfileAgent         ← company snapshot (sector, P/E, market cap …)
    ├─ NewsAgent                   ← recent headlines
    └─ SelfHealingSupervisor       ← circuit-breaker + retry + fallback (supervisor.py)
```

## Agent vs Tool — Where Does New Code Go?

### The rule

A file belongs in `agents/` **only** if its class satisfies all three criteria:

| Criterion | What it means |
|-----------|--------------|
| **Task-oriented capability** | It performs a distinct, named job (resolve a ticker, curate news, render a profile) — not a utility shared by many callers |
| **Defined inputs and outputs** | It has a clear public method signature: typed arguments in, typed result out |
| **Autonomy** | It makes decisions — via an LLM call, a heuristic that chooses between paths, or logic that adapts to what it finds |

If a class fails any criterion it belongs in `tools/`:

| Fails | Example | Goes in |
|-------|---------|---------|
| No autonomy (pure function / lookup) | `DataValidator`, `IndustryIndexAgent` | `tools/` |
| No decisions (thin API wrapper) | `PolygonSearchAgent` | `tools/` |
| No task ownership (renderer/compiler) | `ChartCompiler` | `tools/` |
| Backward-compat shim with no logic | `curve_spec.py` (was agents shim) | delete or `tools/` |

### Quick checklist before creating a new file

```
[ ] Does it call an LLM, apply a multi-step heuristic, or choose between strategies?
    → YES → agents/
    → NO  → tools/

[ ] Does it own a single named responsibility end-to-end?
    → YES, with decisions → agents/
    → YES, pure transform → tools/

[ ] Is it a dataclass, renderer, validator, or HTTP wrapper?
    → tools/ (always)
```

---

## Key Patterns

### 1. Adding a New Agent

Only add to `agents/` when the checklist above passes. Subclass `BaseAgent`
and set three class attributes:

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    system_prompt = "You are a specialist in …"
    tools = [MY_TOOL_DEFINITION]        # Groq-format tool schema(s)
    tool_fn_map = {"my_tool": my_fn}    # name → callable

    def my_action(self, query: str) -> str:
        return self.run(query)
```

`BaseAgent.run()` drives the full tool-calling loop automatically.

### 2. Adding a New Tool

Place tool files in `tools/`. Each tool exposes:
- A Python function that does the actual work and returns a `dict`.
- A `TOOL_DEFINITION` dict in Groq function-calling schema format.

```python
# tools/my_tool.py
def my_tool(param: str) -> dict:
    ...
    return {"result": ...}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "…",
        "parameters": {
            "type": "object",
            "properties": {"param": {"type": "string", "description": "…"}},
            "required": ["param"],
        },
    },
}
```

### 3. Adding a New Route to the Orchestrator

Add an entry to `ROUTING_TOOLS` in `orchestrator.py` and handle the new
function name in `Orchestrator.handle()`.

### 4. Chart Data Pipeline (CurveSpec)

Chart agents do NOT own data-fetching or rendering logic. They:
1. Call `StockCurveTool` / `IndexCurveTool` to get `CurveSpec` objects.
2. Pass the list of `CurveSpec`s to `ChartCompiler.compile()`.

`CurveSpec` carries: `ticker`, `label`, `role` (`"stock"` | `"benchmark"` | `"sector"`), `df` (OHLCV DataFrame), `meta` dict.

### 5. Exchange Suffix → Index Mapping

Defined in `agents/chart_agent.py` (`_EXCHANGE_INDEX`). Add new exchange
suffixes there when supporting new markets.

### 6. Self-Healing Supervisor

Wrap unreliable agent calls with:

```python
result = supervisor.call("AgentName", agent.method, *args)
```

Register agents at startup:
```python
supervisor.register("AgentName", failure_threshold=3, recovery_timeout=60)
```

Circuit states: `CLOSED` → `OPEN` (after N failures) → `HALF_OPEN` (probe) → `CLOSED`.
Error actions: `wait_retry` | `try_variant` | `use_fallback` | `abort`.

### 7. Agent Clarification Pattern (ask_user tool)

**Rule: agents must never use hardcoded Python menus or `if len > 1` branching to ask the user questions.  All interactive prompts must go through the `ask_user` tool so the LLM decides *when* and *what* to ask.**

When an agent might need user input to resolve ambiguity (e.g. multiple matching companies, multiple exchange listings), give it the `ask_user` tool and instruct the LLM in the system prompt to call it.

```python
from tools.ask_user_tool import ask_user, TOOL_DEFINITION as ASK_USER_TOOL

class MyAgent(BaseAgent):
    system_prompt = """
    ...your task description...

    When you detect genuine ambiguity that cannot be resolved automatically,
    call ask_user with a clear question and a list of options.
    When fully resolved, return only the final answer — no prose.
    """
    tools       = [ASK_USER_TOOL]
    tool_fn_map = {"ask_user": ask_user}
```

The `ask_user` tool:
- Accepts `question` (required) and `options` (optional list of labelled choices)
- Displays the question, collects the user's numbered selection or free-form text
- Returns `{"answer": <chosen_label>, "index": <0-based int | null>}`
- The LLM receives the answer as a tool result and continues reasoning

**When to call ask_user (LLM decides, via system prompt instructions):**
- Multiple distinct companies match a query
- One company is listed on multiple exchanges
- Any other case where two valid options exist and the user's preference is unknown

**When NOT to call ask_user:**
- Only one option exists — auto-select silently
- One option is overwhelmingly the standard choice for the query (e.g. a well-known US company with a single US listing)

**The Orchestrator must not replicate this logic.**  Pass all candidates to the agent and let the agent's LLM tool loop handle all ambiguity.

### 8. Multi-Intent Routing

The orchestrator supports routing a single query to **multiple intents** in two ways:

**A. Explicit multi-intent query** (user types a full sentence):
> "AAPL current price and company profile"

The LLM may return multiple `tool_calls` from `ROUTING_TOOLS`. `handle()` processes ALL of them and executes each via `_execute_intent()`, joining results with `---`.

**B. Clarification flow** (bare company name, no explicit intent):
> You: `infosys`

`_clarify_intent()` asks the user one free-form question:
```
  ❓ What would you like for INFY?
     (e.g. 'current price', 'price and news', '3mo chart', '1mo data and company info', 'compare')

  >
```
The user's reply (any phrasing) is sent to the LLM with `ROUTING_TOOLS`. The LLM converts it into one or more tool calls:
- `"1,4"` → get_current_stock_price + get_company_info
- `"price and chart"` → get_current_stock_price + chart_historical
- `"3mo chart and news"` → chart_historical(period=3mo) + get_company_info

**No hardcoded number-to-intent mapping exists.** The LLM handles all phrasing.

**Company snapshot is only fetched** when `get_company_info`, `chart_historical`, or `chart_date_range` is in the action list (controlled by `_COMPANY_INTENTS` in `handle()`).

**ROUTING_TOOLS** contains all 7 routable intents:
`get_current_stock_price`, `get_historical_stock_prices`, `get_stock_prices_between_dates`, `get_company_info`, `chart_historical`, `chart_date_range`, `compare`

When adding a new intent type, add it to ROUTING_TOOLS **and** add it to `_COMPANY_INTENTS` in `handle()` if it requires the company snapshot.

### 9. Access Gateway — External Call Gatekeeper

Every outbound network call **must** go through `tools/access_gateway.py`.

```python
from tools.access_gateway import gateway

# Guard only (raises AccessDeniedError if disabled)
gateway.check("MARKET_DATA")

# Guard + execute + log
result = gateway.call("MARKET_DATA", yf.Ticker, "AAPL")
```

**Categories:**

| Category | Covers |
|----------|--------|
| `LLM` | Groq `chat.completions.create` in base_agent, orchestrator, supervisor |
| `MARKET_DATA` | yfinance calls in price_tools, history_tools |
| `POLYGON` | Polygon.io HTTP requests |
| `NEWS_RSS` | Yahoo Finance & Google News RSS fetches |

**Kill-switches** (env vars — default all enabled):
```
GATEWAY_LLM_ENABLED=false          # disables all Groq calls
GATEWAY_MARKET_DATA_ENABLED=false  # disables yfinance
GATEWAY_POLYGON_ENABLED=false      # disables Polygon search
GATEWAY_NEWS_RSS_ENABLED=false     # disables RSS news feeds
```

**Rules:**
- Import the module-level `gateway` singleton — do NOT instantiate a new `AccessGateway`.
- Add a `gateway.check(category)` call at the top of every new function that makes an outbound network request.
- `AccessDeniedError` is a `RuntimeError` subclass — callers may catch it to produce a graceful degradation message.
- The gateway status is printed at startup: `🛡️  [AccessGateway] ✅ Groq LLM API | ✅ yfinance / market data | …`


```
stock-agent/
├── .env                     # GROQ_API_KEY (never commit)
├── .env.example
├── main.py                  # CLI entry point
├── orchestrator.py          # Intent routing + self-healing pipeline
├── supervisor.py            # SelfHealingSupervisor, CircuitBreaker, RateLimiter
├── requirements.txt
├── agents/                  # ← only files that pass the Agent checklist
│   ├── base_agent.py        # BaseAgent (shared tool-calling loop)
│   ├── ticker_resolver.py   # LLM: best ticker for ambiguous name
│   ├── exchange_disambiguator.py  # LLM auto-select + interactive menu
│   ├── company_search.py    # LLM: discovers all matching listed companies
│   ├── candidate_filter.py  # Heuristic + LLM: filters irrelevant candidates
│   ├── current_price.py     # LLM tool-calling loop → latest price
│   ├── historical_price.py  # LLM tool-calling loop → OHLCV (relative period)
│   ├── date_range.py        # LLM tool-calling loop → OHLCV (date range)
│   ├── chart_agent.py       # Orchestrates curves + AI sector decision → HTML chart
│   ├── comparison_chart_agent.py  # Multi-ticker + arbitrage chart orchestration
│   ├── sector_index.py      # Keyword + LLM fallback → sector index ticker
│   ├── company_profile.py   # One-shot LLM: synthesises data + news → markdown
│   └── news_agent.py        # Trusted-source filter + LLM curation → headlines
└── tools/                   # ← pure functions, renderers, validators, API wrappers
    ├── curve_spec.py        # CurveSpec dataclass (data contract)
    ├── stock_curve.py       # StockCurveTool → CurveSpec(role='stock')
    ├── index_curve.py       # IndexCurveTool → CurveSpec(role='benchmark'/'sector')
    ├── spread_curve.py      # SpreadCurveTool → CurveSpec(role='spread')
    ├── chart_compiler.py    # Plotly HTML renderer (no LLM, no decisions)
    ├── ticker_tools.py      # search_ticker(), verify_ticker()
    ├── price_tools.py       # get_current_price()
    ├── history_tools.py     # get_historical_prices(), get_price_range_between_dates()
    ├── company_data.py      # CompanyDataTool — yfinance profile + fundamentals
    ├── data_validator.py    # DataValidator — deterministic OHLC validation
    ├── polygon_search.py    # PolygonSearchAgent — Polygon.io HTTP wrapper
    ├── industry_index.py    # Sector lookup table (no LLM; use sector_index.py instead)
    ├── news_rss.py          # Free RSS fallback for news (Yahoo Finance + Google News)
    ├── ask_user_tool.py     # ask_user() + TOOL_DEFINITION — LLM-driven user prompts
    └── access_gateway.py    # AccessGateway singleton — gatekeeper for all external calls
```

## Tech Stack

| Component | Library/Service |
|-----------|----------------|
| LLM inference | Groq (`groq` SDK) — `llama-3.3-70b-versatile` |
| Market data | `yfinance` |
| Charts | `plotly` (interactive HTML) + `matplotlib` |
| Data processing | `pandas` |
| Table formatting | `tabulate` |
| HTTP requests | `requests` |
| Env config | `python-dotenv` |

## Conventions

- **Temperature**: always `0.1` for reasoning agents; `0.0` for structured-JSON-only responses.
- **Max tokens**: `4096` for agents, `128` for classification/JSON calls.
- **Tool definitions**: defined alongside their implementing function in `tools/`, imported into agents.
- **System prompts**: defined as class-level strings on each `BaseAgent` subclass.
- **Groq client**: constructed once in `main.py` / `Orchestrator.__init__()` and passed by dependency injection.
- **Interactive prompts**: use the `ask_user` tool — never raw `input()` or hardcoded menus inside agents.  The only permitted raw `input()` call is in `main.py` (the top-level REPL) and `orchestrator.py` (the clarify-intent menu).
- **External calls**: all outbound network requests must call `gateway.check(category)` first (see Pattern 9).
- **Output files**: interactive HTML charts are written to the project root and opened in the browser automatically.
- **No hardcoded API keys**: always load from environment via `python-dotenv`.

## Environment Setup

```bash
pip install -r requirements.txt
copy .env.example .env   # then set GROQ_API_KEY=<your key>
python main.py
```

Obtain a free API key at <https://console.groq.com>.
