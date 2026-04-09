"""
project_reviewer.py
--------------------
ProjectReviewAgent — AI agent that reviews Python files for project-specific
architectural compliance.

Extends BaseAgent with:
  • A system prompt built from review/rules.py (architecture taxonomy, naming
    conventions, CurveSpec contract, Groq injection pattern, import rules,
    forbidden patterns)
  • Two LLM tools:
      read_file_section  — read a line range of the file under review
      read_related_file  — read another project file for cross-reference
  • Smart content sizing: sends full file for short files (≤ MAX_DIRECT_LINES),
    structure skeleton only for larger files (avoids Groq TPM limit)

The agent deliberately skips issues already caught by the MCP baseline layer
(linting, style, generic security) — it focuses only on architecture.
"""

import re
from pathlib import Path
from typing import List, Optional

from groq import Groq

from agents.base_agent import BaseAgent
from review.rules import SYSTEM_PROMPT

# Files ≤ this many lines are sent in full; larger files get a skeleton only.
_MAX_DIRECT_LINES = 250
# Max lines returned per read_related_file call (token-budget guard).
_MAX_RELATED_LINES = 120
# Max skeleton lines sent for large files.
_MAX_SKELETON_LINES = 200


class ProjectReviewAgent(BaseAgent):
    """
    Reviews a single Python file for project-specific architectural issues.

    The agent receives:
      1. A summary of baseline issues (so it knows what NOT to re-flag)
      2. File content (full if small; structure skeleton if large)
      3. Access to read_file_section and read_related_file tools for
         on-demand deep dives without blowing the token budget

    Usage
    -----
      agent = ProjectReviewAgent(groq_client, project_root="/path/to/stock-agent")
      report = agent.review("agents/chart_agent.py")
    """

    model = "llama-3.3-70b-versatile"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file_section",
                "description": (
                    "Read a specific range of lines from the file currently being reviewed. "
                    "Use this to inspect a method body or class implementation before "
                    "making an architectural judgement."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_line": {
                            "type": "integer",
                            "description": "1-based start line (inclusive)",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based end line (inclusive)",
                        },
                    },
                    "required": ["start_line", "end_line"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_related_file",
                "description": (
                    "Read another file in the project to verify how a class is defined "
                    "or used elsewhere — useful for cross-reference checks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": (
                                "Path relative to the project root, "
                                "e.g. 'tools/curve_spec.py' or 'agents/base_agent.py'"
                            ),
                        },
                    },
                    "required": ["relative_path"],
                },
            },
        },
    ]

    def __init__(self, client: Groq, project_root: str):
        super().__init__(client)
        self.system_prompt         = SYSTEM_PROMPT
        self.project_root          = Path(project_root).resolve()
        self._current_file_lines: List[str] = []
        self.tool_fn_map = {
            "read_file_section": self._tool_read_section,
            "read_related_file": self._tool_read_related,
        }

    # ── Tool implementations ───────────────────────────────────────────────────

    def _tool_read_section(self, start_line: int, end_line: int) -> dict:
        """Return lines [start_line, end_line] (1-based) from the current file."""
        # Cap range to avoid accidentally sending thousands of tokens
        end_line = min(end_line, start_line + 99)
        start    = max(0, start_line - 1)
        end      = min(len(self._current_file_lines), end_line)
        snippet  = self._current_file_lines[start:end]
        return {
            "range": f"{start_line}–{end_line}",
            "content": "".join(f"{start + i + 1:4d}: {ln}" for i, ln in enumerate(snippet)),
        }

    def _tool_read_related(self, relative_path: str) -> dict:
        """Return the first _MAX_RELATED_LINES of a related project file."""
        target = (self.project_root / relative_path).resolve()
        # Path-traversal guard
        if not str(target).startswith(str(self.project_root)):
            return {"error": "Access denied — path is outside the project root."}
        if not target.exists():
            return {"error": f"File not found: {relative_path}"}
        try:
            lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
            truncated = len(lines) > _MAX_RELATED_LINES
            content   = "".join(lines[:_MAX_RELATED_LINES])
            if truncated:
                content += f"\n… (truncated — {len(lines)} total lines)"
            return {"path": relative_path, "content": content}
        except Exception as exc:
            return {"error": str(exc)}

    # ── Public API ─────────────────────────────────────────────────────────────

    def review(
        self,
        file_path: str,
        baseline=None,   # Optional[BaselineResult] — avoid circular import
    ) -> str:
        """
        Review *file_path* and return a markdown-formatted architectural report.

        Parameters
        ----------
        file_path : str
            Absolute or project-relative path to the Python file to review.
        baseline  : BaselineResult | None
            Results from BaselineChecker so the agent skips re-reporting
            issues already caught by static analysis.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            return f"❌ File not found: {file_path}"

        self._current_file_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        total_lines = len(self._current_file_lines)

        baseline_summary = self._format_baseline_summary(baseline)
        file_content     = self._build_file_content(total_lines)

        user_message = (
            f"Please review `{path.name}` (full path: `{file_path}`) "
            f"for project-specific architectural compliance.\n\n"
            f"## Baseline Issues (already reported — do NOT re-flag)\n"
            f"{baseline_summary}\n\n"
            f"## File to Review\n"
            f"Total lines: {total_lines}\n\n"
            f"{file_content}\n\n"
            f"Use `read_file_section` to inspect any method body before "
            f"drawing a conclusion.  Use `read_related_file` to cross-check "
            f"related modules if needed."
        )

        return self.run(user_message)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_baseline_summary(baseline) -> str:
        if not baseline or not baseline.issues:
            return "_No baseline issues._"
        # Cap at 15 lines to stay within token budget
        lines = [f"- `{i.code}` L{i.line}: {i.message}  _{i.tool}_" for i in baseline.issues[:15]]
        if len(baseline.issues) > 15:
            lines.append(f"- … and {len(baseline.issues) - 15} more")
        return "\n".join(lines)

    def _build_file_content(self, total_lines: int) -> str:
        if total_lines <= _MAX_DIRECT_LINES:
            numbered = "".join(
                f"{i + 1:4d}: {ln}" for i, ln in enumerate(self._current_file_lines)
            )
            return f"```python\n{numbered}```"

        skeleton = self._extract_skeleton()
        return (
            f"_File has {total_lines} lines — showing structure only. "
            f"Call `read_file_section` to read any specific section._\n\n"
            f"```python\n{skeleton}```"
        )

    def _extract_skeleton(self) -> str:
        """
        Extract the structural skeleton of a large file:
        imports, class / function definitions, top-level docstrings.
        """
        keep_prefixes = (
            "import ",
            "from ",
            "class ",
            "def ",
            "async def ",
            '"""',
            "# ",
            "@",      # decorators
        )
        skeleton: List[str] = []
        for i, line in enumerate(self._current_file_lines, start=1):
            if line.lstrip().startswith(keep_prefixes):
                skeleton.append(f"{i:4d}: {line}")
            if len(skeleton) >= _MAX_SKELETON_LINES:
                skeleton.append(f"   … (skeleton truncated at {_MAX_SKELETON_LINES} lines)\n")
                break
        return "".join(skeleton)
