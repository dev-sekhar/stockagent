"""
review_orchestrator.py
-----------------------
HybridReviewOrchestrator — combines the MCP baseline layer and the
ProjectReviewAgent into a single, unified review pipeline.

Flow
----
  1. BaselineChecker  →  pylint / flake8 / bandit  [fast, no LLM]
  2. ProjectReviewAgent → architectural review      [LLM, project-aware]
  3. _format_report()  → merged markdown output

Either layer can be skipped via baseline_only / agent_only flags.
"""

from pathlib import Path
from typing import List, Optional

from groq import Groq

from review.mcp_baseline import BaselineChecker, BaselineResult
from review.project_reviewer import ProjectReviewAgent


class HybridReviewOrchestrator:
    """
    Entry point for the hybrid review pipeline.

    Parameters
    ----------
    groq_client    : Groq   — shared client forwarded to ProjectReviewAgent
    project_root   : str    — root directory of the stock-agent project
    baseline_only  : bool   — skip the AI layer; run static analysis only
    agent_only     : bool   — skip static analysis; run AI review only
    """

    def __init__(
        self,
        groq_client:   Groq,
        project_root:  str,
        *,
        baseline_only: bool = False,
        agent_only:    bool = False,
    ):
        self.groq_client   = groq_client
        self.project_root  = project_root
        self.baseline_only = baseline_only
        self.agent_only    = agent_only
        self._checker      = BaselineChecker()
        self._agent        = ProjectReviewAgent(groq_client, project_root)

    # ── Public API ─────────────────────────────────────────────────────────────

    def review_file(self, file_path: str) -> str:
        """Review a single Python file; return a markdown report string."""
        path = Path(file_path).resolve()
        print(f"\n🔍  Reviewing  {path.name} …")

        baseline:     Optional[BaselineResult] = None
        agent_report: Optional[str]            = None

        # ── Layer 1: MCP baseline ──────────────────────────────────────────────
        if not self.agent_only:
            print("  ⚙️   [Baseline]  Running static analysis …", end="", flush=True)
            baseline = self._checker.check(str(path))
            print(f"  {baseline.summary()}")

            if baseline.tools_missing:
                missing = ", ".join(baseline.tools_missing)
                print(f"  ⚠️   Missing tools: {missing}  "
                      f"(run `pip install {missing}` to enable)")

        # ── Layer 2: Project agent ─────────────────────────────────────────────
        if not self.baseline_only:
            if self.groq_client is None:
                print("  ⚠️   [ProjectReviewAgent]  GROQ_API_KEY not set — skipping AI layer.")
            else:
                print("  🤖  [ProjectReviewAgent]  Analysing architecture …")
                agent_report = self._agent.review(str(path), baseline)

        return self._format_report(str(path), baseline, agent_report)

    def review_dir(
        self,
        dir_path: str,
        *,
        exclude: Optional[List[str]] = None,
    ) -> str:
        """
        Review all .py files in *dir_path* (recursive) and aggregate results.

        Files in *exclude* and any __pycache__ directories are skipped.
        Deprecated/dead files are skipped by default.
        """
        _default_exclude = {"__init__.py", "_setup_curves.py", "industry_index.py"}
        skip = set(exclude or []) | _default_exclude
        root = Path(dir_path).resolve()

        files = sorted(
            p for p in root.rglob("*.py")
            if p.name not in skip and "__pycache__" not in str(p)
        )

        if not files:
            return f"_No reviewable Python files found in `{dir_path}`._"

        reports: List[str] = []
        for f in files:
            reports.append(self.review_file(str(f)))

        header = (
            f"# 📋 Hybrid Code Review — `{root.name}/`\n"
            f"_{len(files)} file(s) reviewed_\n\n"
            f"---\n\n"
        )
        return header + "\n\n---\n\n".join(reports)

    # ── Formatting ─────────────────────────────────────────────────────────────

    def _format_report(
        self,
        file_path:    str,
        baseline:     Optional[BaselineResult],
        agent_report: Optional[str],
    ) -> str:
        name  = Path(file_path).name
        lines = [f"## 📄 `{name}`\n"]

        # ── Baseline section ───────────────────────────────────────────────────
        if baseline is not None:
            lines.append("### ⚙️  MCP Baseline\n")

            if baseline.tools_missing:
                missing = ", ".join(f"`{t}`" for t in baseline.tools_missing)
                lines.append(
                    f"> ⚠️  Tools not installed: {missing}.  "
                    f"Run `pip install {' '.join(baseline.tools_missing)}` to enable.\n"
                )

            if not baseline.issues:
                lines.append("✅ No issues found.\n")
            else:
                for sev, icon in [("error", "❌"), ("warning", "⚠️ "), ("info", "ℹ️ ")]:
                    group = [i for i in baseline.issues if i.severity == sev]
                    if not group:
                        continue
                    lines.append(f"\n**{icon} {sev.upper()}S ({len(group)})**\n")
                    for issue in group:
                        lines.append(
                            f"- `{issue.code}` L{issue.line}: {issue.message}  "
                            f"*{issue.tool}*\n"
                        )

        # ── Agent section ──────────────────────────────────────────────────────
        if agent_report is not None:
            lines.append("\n### 🤖 Architectural Review\n")
            lines.append(agent_report + "\n")
        elif not self.baseline_only:
            lines.append("\n### 🤖 Architectural Review\n")
            lines.append("_Skipped (no GROQ_API_KEY)._\n")

        return "".join(lines)
