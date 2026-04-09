"""
review.py — Hybrid Code Review CLI
------------------------------------
Combines MCP baseline checks (pylint / flake8 / bandit) with a
ProjectReviewAgent that knows the stock-agent architectural rules.

Usage
-----
  # Review a single file (both layers)
  python review.py agents/chart_agent.py

  # Review a whole directory
  python review.py agents/

  # Baseline only (no LLM — works without GROQ_API_KEY)
  python review.py --baseline-only tools/

  # AI architectural review only (skip static analysis)
  python review.py --agent-only orchestrator.py

  # Save report to a file
  python review.py agents/ --output report.md
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from review.review_orchestrator import HybridReviewOrchestrator

load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review.py",
        description="Hybrid code review: MCP baseline + AI architectural analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Python file or directory to review",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only static analysis (pylint / flake8 / bandit) — no LLM",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run only the AI architectural review — skip static analysis",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write the report to FILE instead of stdout",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.baseline_only and args.agent_only:
        parser.error("--baseline-only and --agent-only are mutually exclusive.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key and not args.baseline_only:
        print(
            "⚠️   GROQ_API_KEY is not set — the AI layer will be skipped.\n"
            "     Use --baseline-only to suppress this warning, or set the key."
        )

    client = Groq(api_key=api_key) if api_key else None

    project_root = Path(__file__).parent
    orchestrator = HybridReviewOrchestrator(
        groq_client   = client,
        project_root  = str(project_root),
        baseline_only = args.baseline_only,
        agent_only    = args.agent_only,
    )

    target = Path(args.path)
    if not target.exists():
        print(f"❌  Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if target.is_dir():
        report = orchestrator.review_dir(str(target))
    else:
        report = orchestrator.review_file(str(target))

    if args.output:
        out = Path(args.output)
        out.write_text(report, encoding="utf-8")
        print(f"\n✅  Report saved → {out}")
    else:
        print("\n" + "=" * 72)
        print(report)


if __name__ == "__main__":
    main()
