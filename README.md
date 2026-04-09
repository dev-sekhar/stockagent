# 📈 Stock Price Agent

A multi-agent system powered by **Groq LLaMA 3.3 70B** + **yfinance**.
Natural-language queries are routed by an **Orchestrator** to small,
single-purpose agents.

## Architecture

```
User query
    │
    ▼
Orchestrator          ← classifies intent, extracts entities
    │
    ├─ TickerResolverAgent   ← company name → ticker symbol
    │
    ├─ CurrentPriceAgent     ← latest traded price
    ├─ HistoricalPriceAgent  ← OHLCV for relative periods
    └─ DateRangeAgent        ← OHLCV between specific dates
```

Each agent has a single responsibility and its own tools and system prompt.

## Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env        # then add your GROQ_API_KEY
python main.py
```

## Example queries

| Query | Agent called |
|-------|-------------|
| `What is Apple's stock price?` | Orchestrator → TickerResolver → CurrentPriceAgent |
| `TSLA current price` | Orchestrator → CurrentPriceAgent |
| `Show MSFT last 3 months` | Orchestrator → HistoricalPriceAgent |
| `How did Amazon do in 2023?` | Orchestrator → TickerResolver → HistoricalPriceAgent |
| `NVDA from Jan 2024 to June 2024` | Orchestrator → DateRangeAgent |

## File structure

```
stock-agent/
├── tools/
│   ├── ticker_tools.py       # search_ticker()
│   ├── price_tools.py        # get_current_price()
│   └── history_tools.py      # get_historical_prices(), get_price_range_between_dates()
├── agents/
│   ├── base_agent.py         # BaseAgent (shared tool-calling loop)
│   ├── ticker_resolver.py    # TickerResolverAgent
│   ├── current_price.py      # CurrentPriceAgent
│   ├── historical_price.py   # HistoricalPriceAgent
│   └── date_range.py         # DateRangeAgent
├── orchestrator.py           # Orchestrator
├── main.py                   # CLI entry point
├── requirements.txt
└── .env.example
```
