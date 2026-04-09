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
    ├─ ExchangeDisambiguatorAgent  ← company name → verified ticker (LLM + interactive menu)
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

## File Structure

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
    └── industry_index.py    # Sector lookup table (no LLM; use sector_index.py instead)
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

- **Temperature**: always `0.1` for reasoning agents; `0.0` for structured-JSON-only responses (e.g., `ExchangeDisambiguatorAgent`).
- **Max tokens**: `4096` for agents, `128` for classification/JSON calls.
- **Tool definitions**: defined alongside their implementing function in `tools/`, imported into agents.
- **System prompts**: defined as class-level strings on each `BaseAgent` subclass.
- **Groq client**: constructed once in `main.py` / `Orchestrator.__init__()` and passed by dependency injection.
- **Interactive prompts**: only in `ExchangeDisambiguatorAgent` and `main.py`; all other agents are headless.
- **Output files**: interactive HTML charts are written to the project root and opened in the browser automatically.
- **No hardcoded API keys**: always load from environment via `python-dotenv`.

## Environment Setup

```bash
pip install -r requirements.txt
copy .env.example .env   # then set GROQ_API_KEY=<your key>
python main.py
```

Obtain a free API key at <https://console.groq.com>.
