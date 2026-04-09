"""
supervisor.py
-------------
Self-Healing Supervisor for the stock-agent Orchestrator.

Provides circuit-breaker protection, per-agent health tracking, intelligent
error classification, LLM-powered self-diagnosis, and a health-report view.

Circuit-breaker states
----------------------
  CLOSED    — normal operation
  OPEN      — tripped after N consecutive failures; calls are blocked
  HALF_OPEN — recovery probe: one call allowed to test if the agent is healthy

Error classification → recovery strategies
------------------------------------------
  wait_retry   — transient (rate limit, timeout); back off and retry
  try_variant  — data issue; try a different ticker / search term
  use_fallback — agent unavailable; skip to the next fallback in the chain
  abort        — unrecoverable (auth failure, bad config); surface to user

Usage
-----
  sup = SelfHealingSupervisor(groq_client)
  sup.register("Polygon",       failure_threshold=3, recovery_timeout=120)
  sup.register("CompanySearch", failure_threshold=3, recovery_timeout=60)

  # Wraps the call with circuit-breaker + health tracking
  result = sup.call("Polygon", polygon_agent.search, query)

  # Diagnose a failure
  diag = sup.diagnose("Polygon", str(exc), {"query": query})
  # → {"action": "use_fallback", "reason": "...", "suggestion": "..."}

  # Check health
  print(sup.health_report())
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from groq import Groq


# ── Circuit-breaker state ─────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED    = "closed"     # Normal — all calls pass through
    OPEN      = "open"       # Tripped — calls blocked until recovery timeout
    HALF_OPEN = "half_open"  # Recovery probe — one call allowed through


class CircuitOpenError(RuntimeError):
    """Raised when a call is blocked because the agent's circuit is open."""


# ── Agent health record ───────────────────────────────────────────────────────

@dataclass
class AgentHealth:
    """Tracks the health state of a single registered agent."""

    name:                 str
    failure_threshold:    int   = 3      # consecutive failures before OPEN
    recovery_timeout:     float = 120.0  # seconds before HALF_OPEN probe

    # Runtime state (updated by CircuitBreaker on every call)
    state:                CircuitState = CircuitState.CLOSED
    consecutive_failures: int          = 0
    total_calls:          int          = 0
    total_failures:       int          = 0
    last_error:           str          = ""
    last_success_time:    float        = field(default_factory=time.time)
    circuit_opened_at:    float        = 0.0

    # ── State transitions ──────────────────────────────────────────────────────

    def is_callable(self) -> bool:
        """
        Return True if a call may proceed.
        Automatically transitions OPEN → HALF_OPEN when recovery_timeout expires.
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.circuit_opened_at >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                print(f"  🟡 [CircuitBreaker] {self.name} → HALF_OPEN (recovery probe)")
                return True
            return False
        # HALF_OPEN: allow the probe call through
        return True

    def record_success(self) -> None:
        was_degraded = self.state != CircuitState.CLOSED
        self.consecutive_failures = 0
        self.total_calls         += 1
        self.last_success_time    = time.time()
        self.state                = CircuitState.CLOSED
        if was_degraded:
            print(f"  ✅ [CircuitBreaker] {self.name} → CLOSED (recovered)")

    def record_failure(self, error: str) -> None:
        self.consecutive_failures += 1
        self.total_calls          += 1
        self.total_failures       += 1
        self.last_error            = error

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — re-open immediately
            self.state             = CircuitState.OPEN
            self.circuit_opened_at = time.time()
            print(f"  🔴 [CircuitBreaker] {self.name} probe failed → re-OPEN")
        elif (
            self.state != CircuitState.OPEN
            and self.consecutive_failures >= self.failure_threshold
        ):
            self.state             = CircuitState.OPEN
            self.circuit_opened_at = time.time()
            print(
                f"  🔴 [CircuitBreaker] {self.name} → OPEN "
                f"({self.consecutive_failures} consecutive failures)"
            )

    def status_emoji(self) -> str:
        return {
            CircuitState.CLOSED:    "🟢",
            CircuitState.OPEN:      "🔴",
            CircuitState.HALF_OPEN: "🟡",
        }[self.state]

    def seconds_until_probe(self) -> float:
        """Seconds remaining before HALF_OPEN probe (0 if not OPEN)."""
        if self.state != CircuitState.OPEN:
            return 0.0
        return max(0.0, self.recovery_timeout - (time.time() - self.circuit_opened_at))


# ── Circuit breaker (callable wrapper) ───────────────────────────────────────

class CircuitBreaker:
    """
    Wraps a callable with circuit-breaker protection backed by an AgentHealth
    record.  Instantiate via SelfHealingSupervisor.register().
    """

    def __init__(self, health: AgentHealth):
        self.health = health

    def __call__(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self.health.is_callable():
            remaining = self.health.seconds_until_probe()
            raise CircuitOpenError(
                f"[{self.health.name}] circuit is OPEN — "
                f"retry in {remaining:.0f}s"
            )
        try:
            result = fn(*args, **kwargs)
            self.health.record_success()
            return result
        except Exception as exc:
            self.health.record_failure(str(exc))
            raise


# ── Error-pattern tables ──────────────────────────────────────────────────────

_RATE_LIMIT_PATTERNS: List[str] = [
    "413", "rate_limit", "tokens per minute", "request too large",
    "request entity too large", "tpm", "ratelimit",
]
_NO_DATA_PATTERNS: List[str] = [
    "no data", "no price", "not found", "empty dataframe",
    "delisted", "invalid ticker", "no listings", "404",
]
_AUTH_PATTERNS: List[str] = [
    "401", "403", "unauthorized", "forbidden", "api key",
    "authentication", "invalid_api_key",
]
_TIMEOUT_PATTERNS: List[str] = [
    "timeout", "timed out", "connection error", "connection reset",
    "read timeout", "connect timeout",
]


# ── Self-Healing Supervisor ───────────────────────────────────────────────────

class SelfHealingSupervisor:
    """
    Monitors agent health, manages circuit breakers, classifies errors,
    and performs LLM-powered self-diagnosis to suggest recovery actions.

    All public methods are safe to call even when groq_client is None;
    the LLM layer degrades gracefully to rule-based classification.
    """

    def __init__(self, groq_client: Optional[Groq] = None):
        self.client:   Optional[Groq]                    = groq_client
        self._health:  Dict[str, AgentHealth]            = {}
        self._breakers: Dict[str, CircuitBreaker]        = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name:              str,
        failure_threshold: int   = 3,
        recovery_timeout:  float = 120.0,
    ) -> "SelfHealingSupervisor":
        """
        Register an agent with the supervisor.  Returns self for chaining.

        Can be called multiple times with the same name to update thresholds
        (will reset health state for that agent).
        """
        health               = AgentHealth(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self._health[name]   = health
        self._breakers[name] = CircuitBreaker(health)
        return self

    # ── Protected call ────────────────────────────────────────────────────────

    def call(self, agent_name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke fn(*args, **kwargs) through the circuit breaker for agent_name.

        Auto-registers the agent with default thresholds if not yet registered.
        Raises CircuitOpenError if the circuit is OPEN.
        Propagates the original exception (after recording the failure) if the
        call itself fails.
        """
        if agent_name not in self._breakers:
            self.register(agent_name)
        return self._breakers[agent_name](fn, *args, **kwargs)

    # ── Health queries ────────────────────────────────────────────────────────

    def is_healthy(self, agent_name: str) -> bool:
        """Return True if the agent's circuit is CLOSED or HALF_OPEN."""
        h = self._health.get(agent_name)
        return h is None or h.is_callable()

    # ── Error classification ──────────────────────────────────────────────────

    @staticmethod
    def classify_error(error: str) -> str:
        """
        Classify an error string into a recovery strategy without an LLM call.

        Returns one of:
          'wait_retry'   — transient; back off and retry
          'try_variant'  — data issue; try a different ticker or search term
          'use_fallback' — agent unavailable; use the next fallback
          'abort'        — unrecoverable; surface to the user
        """
        err = error.lower()
        if any(p in err for p in _RATE_LIMIT_PATTERNS):
            return "wait_retry"
        if any(p in err for p in _TIMEOUT_PATTERNS):
            return "wait_retry"
        if any(p in err for p in _AUTH_PATTERNS):
            return "use_fallback"
        if any(p in err for p in _NO_DATA_PATTERNS):
            return "try_variant"
        return "try_variant"

    # ── LLM self-diagnosis ────────────────────────────────────────────────────

    def diagnose(
        self,
        agent_name: str,
        error:      str,
        context:    Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Diagnose why *agent_name* failed and suggest the best recovery action.

        Tries the LLM first; falls back to rule-based classification if the
        LLM is unavailable or returns an invalid response.

        Returns
        -------
        dict with keys:
          action     : 'wait_retry' | 'try_variant' | 'use_fallback' | 'abort'
          reason     : human-readable explanation
          suggestion : concrete next step
        """
        _SUGGESTIONS = {
            "wait_retry":   "Wait ~60 s for the rate limit to clear, then retry",
            "try_variant":  "Try a different ticker suffix (e.g. .NS, .BO, .L) or search term",
            "use_fallback": "Skip this agent and use the next source in the fallback chain",
            "abort":        "Cannot recover automatically — surface the error to the user",
        }

        # ── Fast path: rule-based ──────────────────────────────────────────────
        strategy = self.classify_error(error)
        fallback = {
            "action":     strategy,
            "reason":     f"Rule-based classification of error: {error[:120]}",
            "suggestion": _SUGGESTIONS[strategy],
        }

        if not self.client:
            return fallback

        # ── LLM path ──────────────────────────────────────────────────────────
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a diagnostic assistant for a stock-price multi-agent system. "
                            "Analyse the failed agent call and suggest the best recovery action.\n"
                            "Respond ONLY with valid JSON (no markdown):\n"
                            '{"action":"...","reason":"...","suggestion":"..."}\n'
                            "action MUST be one of: wait_retry | try_variant | use_fallback | abort"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Agent: {agent_name}\n"
                            f"Error: {error[:300]}\n"
                            f"Context: {json.dumps(context, default=str)[:400]}"
                        ),
                    },
                ],
                max_tokens=200,
                temperature=0.0,
            )
            text  = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if result.get("action") in {"wait_retry", "try_variant", "use_fallback", "abort"}:
                    return result
        except Exception:
            pass

        return fallback

    # ── Observability ─────────────────────────────────────────────────────────

    def health_report(self) -> str:
        """Return a markdown-formatted health table for all registered agents."""
        if not self._health:
            return "_No agents registered._"

        rows = [
            "### 🏥 Agent Health\n",
            "| Agent | State | Calls | Failures | Last Error |",
            "|-------|-------|------:|---------:|------------|",
        ]
        for name, h in self._health.items():
            probe_info = (
                f" ({h.seconds_until_probe():.0f}s)"
                if h.state == CircuitState.OPEN
                else ""
            )
            state_str = f"{h.status_emoji()} {h.state.value}{probe_info}"
            err_short = (h.last_error[:45] + "…") if len(h.last_error) > 45 else h.last_error
            rows.append(
                f"| {name} | {state_str} | {h.total_calls} | {h.total_failures} | `{err_short}` |"
            )
        return "\n".join(rows)
