"""
mcp_baseline.py
---------------
MCP-layer baseline checker.

Wraps three standard static-analysis CLI tools — pylint, flake8, bandit —
and exposes their results as structured Python objects.  These tools
represent the "mechanical baseline" checks that an MCP tool-server would
expose as callable functions.

Results are consumed by HybridReviewOrchestrator, which passes them to
ProjectReviewAgent so the AI layer can skip issues already reported here.

Graceful degradation: missing tools are noted in BaselineResult.tools_missing
but do not cause failure.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class BaselineIssue:
    tool:     str   # 'pylint' | 'flake8' | 'bandit'
    severity: str   # 'error'  | 'warning' | 'info'
    file:     str
    line:     int
    col:      int
    code:     str
    message:  str

    def __str__(self) -> str:
        return (
            f"[{self.tool}] {self.severity.upper():7s} "
            f"{self.code} L{self.line}: {self.message}"
        )


@dataclass
class BaselineResult:
    path:          str
    issues:        List[BaselineIssue] = field(default_factory=list)
    tools_run:     List[str]           = field(default_factory=list)
    tools_missing: List[str]           = field(default_factory=list)

    @property
    def errors(self) -> List[BaselineIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[BaselineIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> List[BaselineIssue]:
        return [i for i in self.issues if i.severity == "info"]

    def summary(self) -> str:
        counts = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] += 1
        parts = [f"{v} {k}(s)" for k, v in counts.items() if v]
        tools = ", ".join(self.tools_run) if self.tools_run else "none"
        return (", ".join(parts) or "no issues") + f"  [tools: {tools}]"


# ── Checker ───────────────────────────────────────────────────────────────────

class BaselineChecker:
    """
    MCP baseline: runs pylint, flake8, and bandit on a Python file or
    directory and returns a BaselineResult.

    Each runner is isolated — a crash in one tool does not block the others.
    """

    # ── Tool availability ──────────────────────────────────────────────────────

    @staticmethod
    def _tool_available(module_name: str) -> bool:
        """Return True if `python -m <module_name> --version` succeeds."""
        try:
            subprocess.run(
                [sys.executable, "-m", module_name, "--version"],
                capture_output=True,
                timeout=10,
            )
            return True
        except Exception:
            return False

    # ── pylint ─────────────────────────────────────────────────────────────────

    def _run_pylint(self, path: str) -> List[BaselineIssue]:
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pylint",
                    "--output-format=json",
                    # Skip convention/refactor noise; surface errors + warnings only
                    "--disable=C,R",
                    "--score=no",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            raw = result.stdout.strip()
            if not raw:
                return []
            data = json.loads(raw)
            issues = []
            for item in data:
                category = (item.get("type") or "W")[0].upper()
                severity = "error" if category in ("E", "F") else "warning"
                issues.append(BaselineIssue(
                    tool="pylint",
                    severity=severity,
                    file=item.get("path", path),
                    line=item.get("line", 0),
                    col=item.get("column", 0),
                    code=item.get("message-id", ""),
                    message=item.get("message", ""),
                ))
            return issues
        except json.JSONDecodeError:
            return []
        except Exception as exc:
            return [BaselineIssue("pylint", "info", path, 0, 0, "RUN_ERROR", str(exc))]

    # ── flake8 ─────────────────────────────────────────────────────────────────

    def _run_flake8(self, path: str) -> List[BaselineIssue]:
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "flake8",
                    "--max-line-length=120",
                    # Custom format: fields separated by :: for easy splitting
                    "--format=%(path)s::%(row)d::%(col)d::%(code)s::%(text)s",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            issues = []
            for line in result.stdout.splitlines():
                parts = line.split("::")
                if len(parts) < 5:
                    continue
                _, row, col, code, msg = parts[0], parts[1], parts[2], parts[3], "::".join(parts[4:])
                code = code.strip()
                if code.startswith(("E9", "F8", "F4", "F82")):
                    sev = "error"
                elif code.startswith("W"):
                    sev = "warning"
                else:
                    sev = "info"
                issues.append(BaselineIssue(
                    tool="flake8",
                    severity=sev,
                    file=path,
                    line=int(row),
                    col=int(col),
                    code=code,
                    message=msg.strip(),
                ))
            return issues
        except Exception as exc:
            return [BaselineIssue("flake8", "info", path, 0, 0, "RUN_ERROR", str(exc))]

    # ── bandit ─────────────────────────────────────────────────────────────────

    def _run_bandit(self, path: str) -> List[BaselineIssue]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bandit", "-r", "-f", "json", "-q", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            raw = result.stdout.strip()
            if not raw:
                return []
            data = json.loads(raw)
            sev_map = {"HIGH": "error", "MEDIUM": "warning", "LOW": "info"}
            issues = []
            for item in data.get("results", []):
                # Skip low-confidence findings to reduce noise
                if item.get("issue_confidence", "LOW") == "LOW":
                    continue
                issues.append(BaselineIssue(
                    tool="bandit",
                    severity=sev_map.get(item.get("issue_severity", "LOW"), "info"),
                    file=item.get("filename", path),
                    line=item.get("line_number", 0),
                    col=0,
                    code=item.get("test_id", ""),
                    message=item.get("issue_text", ""),
                ))
            return issues
        except json.JSONDecodeError:
            return []
        except Exception as exc:
            return [BaselineIssue("bandit", "info", path, 0, 0, "RUN_ERROR", str(exc))]

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, path: str) -> BaselineResult:
        """
        Run all available tools against *path* (file or directory).

        Returns a BaselineResult with all issues found and metadata about
        which tools ran and which were missing.
        """
        result = BaselineResult(path=path)

        for tool_module, run_fn in [
            ("pylint", self._run_pylint),
            ("flake8", self._run_flake8),
            ("bandit", self._run_bandit),
        ]:
            if self._tool_available(tool_module):
                result.tools_run.append(tool_module)
                result.issues.extend(run_fn(path))
            else:
                result.tools_missing.append(tool_module)

        return result
