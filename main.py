"""
main.py – Interactive CLI entry point for the Stock Price Agent.
"""
import os
import sys
from dotenv import load_dotenv
from orchestrator import Orchestrator
from tools.access_gateway import gateway

load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n❌  GROQ_API_KEY not found in environment / .env file.")
        print("   1. Copy  .env.example  →  .env")
        print("   2. Paste your key from https://console.groq.com")
        sys.exit(1)

    orchestrator = Orchestrator(api_key=api_key)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        📈  Stock Price Agent (v3)  📉               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Just type a company name or ticker to get started. ║")
    print("║                                                      ║")
    print("║  Examples:                                           ║")
    print("║    apple                 → choose from a menu        ║")
    print("║    INFY.NS current price → latest price              ║")
    print("║    AAPL 1y chart         → interactive chart         ║")
    print("║    tell me about Wipro   → company profile & news    ║")
    print("║    compare INFY vs WIPRO → side-by-side chart        ║")
    print("║                                                      ║")
    print("║  Type  health  to see agent health status.           ║")
    print("║  Type  exit    to quit.                              ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(gateway.report())

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q", "bye"}:
            print("Goodbye! 👋")
            break
        if query.lower() == "health":
            print(f"\n{orchestrator.health_report()}\n")
            continue

        try:
            answer = orchestrator.handle(query)
            print(f"\nAgent:\n{answer}\n")
        except Exception as exc:
            print(f"\n⚠️  Error: {exc}\n")


if __name__ == "__main__":
    main()
