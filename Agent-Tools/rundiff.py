#!/usr/bin/env python3
"""rundiff — compare two SmallCTL run directories.

Useful for before/after regression checks.

Examples:
  python Agent-Tools/rundiff.py before_dir after_dir
  python Agent-Tools/rundiff.py 6d6c87f1 57e619aa
  python Agent-Tools/rundiff.py latest-1 latest
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agent_tools_lib import (
    Colors,
    colorize,
    discover_runs,
    error_records,
    event_counter,
    get_run_objective,
    iter_records,
    load_summaries,
    load_task_summaries,
    resolve_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two SmallCTL runs.")
    parser.add_argument("left", help="Left run dir, run id, or 'latest-N'")
    parser.add_argument("right", help="Right run dir, run id, or 'latest-N'")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--events", nargs="+", help="Specific events to compare")
    parser.add_argument("--same-objective", dest="same_objective", help="Only compare runs whose objective contains TEXT")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser.parse_args()


def _resolve_with_offset(spec: str, runs: list[Path]) -> Path:
    if spec.startswith("latest-"):
        try:
            offset = int(spec.split("-", 1)[1])
        except ValueError:
            raise ValueError(f"Invalid latest-N spec: {spec}")
        if offset < 0 or offset >= len(runs):
            raise ValueError(f"latest-{offset} out of range (only {len(runs)} runs)")
        return runs[offset]
    return resolve_run_dir(spec)


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    summaries = load_summaries(run_dir)
    session = summaries.get("session_summary", {})
    task_summary = summaries.get("task_summary", {})
    task_summaries = load_task_summaries(run_dir)

    harness_records = list(iter_records(run_dir, "harness"))
    tools_records = list(iter_records(run_dir, "tools"))
    events = event_counter(harness_records)
    errors = error_records(harness_records)

    dispatch_starts = [r for r in tools_records if r.get("event") == "dispatch_start"]
    failed_dispatches = [r for r in tools_records if r.get("event") == "dispatch_complete" and (r.get("data") or {}).get("success") is False]

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "objective": get_run_objective(session, task_summary, task_summaries),
        "session_summary": session,
        "task_summary": task_summary,
        "task_count": len(task_summaries),
        "incomplete_task_count": len(session.get("incomplete_task_ids", [])),
        "deliverable_verified": session.get("deliverable_verified"),
        "overall_objective_status": session.get("overall_objective_status"),
        "total_tool_calls": task_summary.get("total_tool_calls", 0),
        "error_record_count": len(errors),
        "event_counts": dict(events),
        "dispatch_count": len(dispatch_starts),
        "failed_dispatch_count": len(failed_dispatches),
    }


def _diff_number(left: Any, right: Any, label: str) -> str:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = right - left
        sign = "+" if delta >= 0 else ""
        return f"  {label}: {left} -> {right} ({sign}{delta})"
    return f"  {label}: {left} -> {right}"


def _render_text(left: dict[str, Any], right: dict[str, Any], left_events: Counter[str], right_events: Counter[str], same_objective: str | None = None) -> str:
    lines: list[str] = []
    lines.append(colorize("Run diff", Colors.BOLD + Colors.CYAN))
    lines.append(f"  left:  {left['run_name']}  ({left['run_dir']})")
    lines.append(f"  right: {right['run_name']}  ({right['run_dir']})")
    if same_objective:
        lines.append(colorize(f"  objective filter: {same_objective}", Colors.YELLOW))
    lines.append("")
    lines.append(colorize("Outcomes", Colors.BOLD + Colors.BLUE))
    lines.append(_diff_number(left["overall_objective_status"], right["overall_objective_status"], "overall status"))
    lines.append(_diff_number(left["deliverable_verified"], right["deliverable_verified"], "deliverable verified"))
    lines.append(_diff_number(left["task_count"], right["task_count"], "tasks"))
    lines.append(_diff_number(left["incomplete_task_count"], right["incomplete_task_count"], "incomplete tasks"))
    lines.append("")
    lines.append(colorize("Volume", Colors.BOLD + Colors.BLUE))
    lines.append(_diff_number(left["total_tool_calls"], right["total_tool_calls"], "tool calls"))
    lines.append(_diff_number(left["dispatch_count"], right["dispatch_count"], "dispatch records"))
    lines.append(_diff_number(left["error_record_count"], right["error_record_count"], "error records"))
    lines.append(_diff_number(left["failed_dispatch_count"], right["failed_dispatch_count"], "failed dispatches"))

    lines.append("")
    lines.append(colorize("Event count deltas", Colors.BOLD + Colors.BLUE))
    all_events = set(left_events) | set(right_events)
    deltas = []
    for ev in sorted(all_events):
        lc = left_events.get(ev, 0)
        rc = right_events.get(ev, 0)
        if lc != rc:
            deltas.append((ev, lc, rc))
    if deltas:
        for ev, lc, rc in deltas:
            sign = "+" if rc - lc >= 0 else ""
            lines.append(f"  {ev}: {lc} -> {rc} ({sign}{rc - lc})")
    else:
        lines.append("  no event count differences")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    base_runs = discover_runs(logs_dir) if logs_dir else discover_runs()
    try:
        left_dir = _resolve_with_offset(args.left, base_runs)
        right_dir = _resolve_with_offset(args.right, base_runs)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    left = _summarize_run(left_dir)
    right = _summarize_run(right_dir)

    if args.same_objective:
        query = args.same_objective.lower()
        if query not in (left.get("objective") or "").lower() or query not in (right.get("objective") or "").lower():
            print(
                colorize(
                    f"Both runs must have an objective containing '{args.same_objective}'.\n"
                    f"  left:  {left.get('objective') or 'n/a'}\n"
                    f"  right: {right.get('objective') or 'n/a'}",
                    Colors.RED,
                ),
                file=sys.stderr,
            )
            return 1

    left_events = Counter(left["event_counts"])
    right_events = Counter(right["event_counts"])

    if args.events:
        left_events = Counter({ev: left_events.get(ev, 0) for ev in args.events})
        right_events = Counter({ev: right_events.get(ev, 0) for ev in args.events})

    if args.json:
        print(json.dumps({"left": left, "right": right}, indent=2, default=str))
        return 0

    print(_render_text(left, right, left_events, right_events, same_objective=args.same_objective))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
