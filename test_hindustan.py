# Quick smoke-test for the 'hindustan' query.
# Patches input() so the agent auto-selects option 1 (first company, first exchange)
# without needing a real terminal.
# Run from the stock-agent directory: python test_hindustan.py
import os, sys
from dotenv import load_dotenv

load_dotenv()

# ── Auto-answer any input() prompts with "1" ─────────────────────────────────
_answer_iter = iter(["1", "1", "1", "1", "1"])

def _auto_input(prompt=""):
    ans = next(_answer_iter, "1")
    print(f"{prompt}{ans}")
    return ans

import builtins
builtins.input = _auto_input

# ── Run the orchestrator ──────────────────────────────────────────────────────
from orchestrator import Orchestrator

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌  GROQ_API_KEY not set in .env")
    sys.exit(1)

print("\n" + "="*60)
print("TEST: 'hindustan' → current price")
print("="*60 + "\n")

orch = Orchestrator(api_key=api_key)
try:
    result = orch.handle("hindustan")
    print(f"\n✅ Agent response:\n{result}")
except Exception as e:
    print(f"\n❌ Error: {e}")
