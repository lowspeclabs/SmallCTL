#!/usr/bin/env python3
"""symbolmap — map symbols, events, and tool names back to source files.

SmallCTL spreads related concepts across packages. This tool indexes:
  - Python functions and classes (AST)
  - string literals that look like harness event names
  - tool registration sites (calls to @tool, ToolSpec, register_*)

Examples:
  python Agent-Tools/symbolmap.py dispatch_tools
  python Agent-Tools/symbolmap.py --event action_stall
  python Agent-Tools/symbolmap.py --tool file_patch
  python Agent-Tools/symbolmap.py --class LoopState
  python Agent-Tools/symbolmap.py --rebuild
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_tools_lib import SRC_DIR


INDEX_PATH = Path(__file__).resolve().parent / ".symbolmap.json"


class SourceIndex:
    """In-memory index of symbols and references in src/smallctl."""

    def __init__(self) -> None:
        self.functions: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.decorators: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": self.functions,
            "classes": self.classes,
            "events": self.events,
            "tools": self.tools,
            "decorators": self.decorators,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceIndex":
        idx = cls()
        idx.functions = data.get("functions", [])
        idx.classes = data.get("classes", [])
        idx.events = data.get("events", [])
        idx.tools = data.get("tools", [])
        idx.decorators = data.get("decorators", [])
        idx.metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        return idx


def _event_like_strings() -> set[str]:
    """Heuristic set of event names likely to appear in logs."""
    return {
        "model_call_start",
        "model_call_end",
        "model_output",
        "model_thinking",
        "model_token",
        "dispatch_start",
        "dispatch_complete",
        "dispatch_spec",
        "action_stall",
        "no_tool_recovery",
        "recovery_failure_event_recorded",
        "fama_mitigation_activated",
        "fama_tool_call_blocked",
        "context_invalidated",
        "tool_blocked_not_exposed",
        "task_complete",
        "task_finalize",
    }


def _build_ast_index(src_dir: Path) -> SourceIndex:
    idx = SourceIndex()
    event_pattern = re.compile(r"^[a-z][a-z0-9_]*_[a-z][a-z0-9_]*$")
    known_events = _event_like_strings()

    for py_path in src_dir.rglob("*.py"):
        rel = py_path.relative_to(src_dir)
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError as exc:
            logging.warning("Syntax error in %s: %s", py_path, exc)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                idx.functions.append({
                    "name": node.name,
                    "file": str(rel),
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                })
            elif isinstance(node, ast.ClassDef):
                idx.classes.append({
                    "name": node.name,
                    "file": str(rel),
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                })
            elif isinstance(node, ast.Call):
                # Detect @tool(...) and ToolSpec(...) and register_* calls
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in {"tool", "ToolSpec"}:
                    # Try to extract the first string argument as tool name
                    tool_name = None
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        tool_name = node.args[0].value
                    if not tool_name and node.keywords:
                        for kw in node.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                tool_name = kw.value.value
                    if tool_name:
                        idx.tools.append({
                            "name": tool_name,
                            "file": str(rel),
                            "line": node.lineno,
                            "kind": name,
                        })
                elif name and name.startswith("register"):
                    idx.decorators.append({
                        "name": name,
                        "file": str(rel),
                        "line": node.lineno,
                        "kind": "register_call",
                    })
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value in known_events or (event_pattern.match(value) and "_" in value and len(value) > 6):
                    idx.events.append({
                        "name": value,
                        "file": str(rel),
                        "line": node.lineno,
                    })

    return idx


def _source_fingerprint(src_dir: Path) -> dict[str, Any]:
    py_files = [path for path in src_dir.rglob("*.py") if path.is_file()]
    newest_mtime_ns = 0
    for path in py_files:
        try:
            newest_mtime_ns = max(newest_mtime_ns, path.stat().st_mtime_ns)
        except OSError:
            continue
    return {
        "cache_version": 2,
        "src_dir": str(src_dir.resolve()),
        "file_count": len(py_files),
        "newest_mtime_ns": newest_mtime_ns,
    }


def _cache_is_current(data: dict[str, Any], src_dir: Path) -> bool:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return False
    current = _source_fingerprint(src_dir)
    return (
        metadata.get("cache_version") == current["cache_version"]
        and metadata.get("src_dir") == current["src_dir"]
        and int(metadata.get("file_count") or -1) == current["file_count"]
        and int(metadata.get("newest_mtime_ns") or 0) >= current["newest_mtime_ns"]
    )


def _rg_search(src_dir: Path, pattern: str, extra_args: list[str] | None = None) -> list[dict[str, Any]]:
    """Run ripgrep and return matches."""
    args = ["rg", "--json", "--no-heading"] + (extra_args or []) + [pattern, str(src_dir)]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    matches: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "match":
            data = obj.get("data", {})
            path = data.get("path", {}).get("text", "")
            line_no = data.get("line_number", 0)
            text = "".join(p.get("text", "") for p in data.get("submatches", [])) or data.get("lines", {}).get("text", "").strip()
            matches.append({"file": path, "line": line_no, "text": text})
    return matches


def build_index(src_dir: Path, use_cache: bool = True) -> SourceIndex:
    if use_cache and INDEX_PATH.exists():
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if _cache_is_current(data, src_dir):
                return SourceIndex.from_dict(data)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not load cached index: %s", exc)

    idx = _build_ast_index(src_dir)
    idx.metadata = _source_fingerprint(src_dir)
    try:
        INDEX_PATH.write_text(json.dumps(idx.to_dict(), indent=2), encoding="utf-8")
    except OSError as exc:
        logging.warning("Could not write index cache: %s", exc)
    return idx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map SmallCTL symbols/events/tools to source files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("query", nargs="?", help="Symbol or substring to search")
    group.add_argument("--event", help="Search for an event string literal")
    group.add_argument("--tool", help="Search for a tool name")
    group.add_argument("--class", dest="class_name", help="Search for a class definition")
    group.add_argument("--rebuild", action="store_true", help="Rebuild the source index")
    parser.add_argument("--src-dir", help="Custom source directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser.parse_args()


def _print_section(title: str, items: list[dict[str, Any]]) -> None:
    from agent_tools_lib import Colors, colorize
    print(colorize(f"{title} ({len(items)})", Colors.BOLD + Colors.BLUE))
    for item in items[:30]:
        file = item.get("file", "")
        line = item.get("line", 0)
        name = item.get("name", item.get("text", ""))
        kind = item.get("kind", "")
        suffix = f" [{kind}]" if kind else ""
        print(f"  {file}:{line}  {name}{suffix}")
    if len(items) > 30:
        print(f"  ... and {len(items) - 30} more")


def main() -> int:
    args = _parse_args()
    src_dir = Path(args.src_dir) if args.src_dir else SRC_DIR
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    if args.rebuild:
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        build_index(src_dir, use_cache=False)
        print(f"Rebuilt index: {INDEX_PATH}")
        return 0

    idx = build_index(src_dir, use_cache=True)

    query = args.query or args.event or args.tool or args.class_name
    if not query:
        print("Provide a query or one of --event/--tool/--class", file=sys.stderr)
        return 1

    qlower = query.lower()

    results: dict[str, list[dict[str, Any]]] = {
        "functions": [f for f in idx.functions if qlower in f["name"].lower()],
        "classes": [c for c in idx.classes if qlower in c["name"].lower()],
        "events": [e for e in idx.events if qlower in e["name"].lower()],
        "tools": [t for t in idx.tools if qlower in t["name"].lower()],
        "decorators": [d for d in idx.decorators if qlower in d["name"].lower()],
    }

    # Also do a ripgrep text search for extra context (event strings, comments, etc.)
    rg_matches = _rg_search(src_dir, query)
    results["rg_matches"] = rg_matches

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"Query: {query}")
    print(f"Source: {src_dir}")
    for section, items in results.items():
        if items:
            print()
            _print_section(section.replace("_", " ").title(), items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
