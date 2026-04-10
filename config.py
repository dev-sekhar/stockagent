"""
config.py — centralised runtime configuration
----------------------------------------------
All tuneable values are read from environment variables with sensible
defaults so the app works out-of-the-box without a .env file.

To override any value add it to your .env file, e.g.:
    LLM_MODEL=llama-3.3-70b-versatile
    LLM_TEMP_REASON=0.1
    DEFAULT_BENCHMARK=^NSEI
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
# Model used by all agents.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Temperature for reasoning / tool-calling agents (allow some creativity).
TEMP_REASON = float(os.getenv("LLM_TEMP_REASON", "0.1"))

# Temperature for structured-output-only calls (deterministic JSON).
TEMP_STRUCT = float(os.getenv("LLM_TEMP_STRUCT", "0.0"))

# ── Token limits per use-case ─────────────────────────────────────────────────
TOKENS_AGENT     = int(os.getenv("TOKENS_AGENT",     "4096"))  # general tool-calling
TOKENS_CLASSIFY  = int(os.getenv("TOKENS_CLASSIFY",   "256"))  # intent routing
TOKENS_SENTIMENT = int(os.getenv("TOKENS_SENTIMENT", "1024"))  # news sentiment
TOKENS_NEWS      = int(os.getenv("TOKENS_NEWS",      "1200"))  # news curation
TOKENS_PROFILE   = int(os.getenv("TOKENS_PROFILE",   "1600"))  # company profile
TOKENS_SEARCH    = int(os.getenv("TOKENS_SEARCH",     "512"))  # company search / filter
TOKENS_SECTOR    = int(os.getenv("TOKENS_SECTOR",      "32"))  # sector key (short answer)
TOKENS_DIAGNOSE  = int(os.getenv("TOKENS_DIAGNOSE",   "200"))  # supervisor diagnosis

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Groq free tier allows 30 requests/minute on most models.
GROQ_RPM = int(os.getenv("GROQ_RPM", "30"))

# ── Circuit-breaker defaults ──────────────────────────────────────────────────
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD",   "3"))
CB_RECOVERY_TIMEOUT  = float(os.getenv("CB_RECOVERY_TIMEOUT", "60"))

# Per-agent overrides (seconds).  Set to empty string to use the default above.
CB_RECOVERY_POLYGON  = float(os.getenv("CB_RECOVERY_POLYGON",  "120"))
CB_RECOVERY_PRICE    = float(os.getenv("CB_RECOVERY_PRICE",     "90"))

# ── Market defaults ───────────────────────────────────────────────────────────
# Default broad-market benchmark overlay when no exchange-specific index is found.
# Change to ^NSEI for India-first deployments, ^FTSE for UK, etc.
DEFAULT_BENCHMARK      = os.getenv("DEFAULT_BENCHMARK",       "^GSPC")
DEFAULT_BENCHMARK_NAME = os.getenv("DEFAULT_BENCHMARK_NAME",  "S&P 500")

# ── News RSS feeds ────────────────────────────────────────────────────────────
YAHOO_RSS_URL   = os.getenv(
    "YAHOO_RSS_URL",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
)
GOOGLE_NEWS_URL = os.getenv(
    "GOOGLE_NEWS_URL",
    "https://news.google.com/rss/search",
)
RSS_TIMEOUT = int(os.getenv("RSS_TIMEOUT", "8"))  # seconds per request

# ── Access Gateway — external service kill-switches ───────────────────────────
# Set any of these to "false" (case-insensitive) to block that category of
# outbound call.  Useful for offline testing or compliance restrictions.
def _bool_env(key: str, default: bool = True) -> bool:
    val = os.getenv(key, "")
    if not val:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")

GATEWAY_LLM_ENABLED         = _bool_env("GATEWAY_LLM_ENABLED",         True)
GATEWAY_MARKET_DATA_ENABLED = _bool_env("GATEWAY_MARKET_DATA_ENABLED",  True)
GATEWAY_POLYGON_ENABLED     = _bool_env("GATEWAY_POLYGON_ENABLED",      True)
GATEWAY_NEWS_RSS_ENABLED    = _bool_env("GATEWAY_NEWS_RSS_ENABLED",      True)
