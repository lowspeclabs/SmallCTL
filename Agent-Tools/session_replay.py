#!/usr/bin/env python3
"""session_replay — replay a run's tool-call sequence with policy/phase simulation.

Reads tools.jsonl + harness.jsonl and replays each dispatch in order,
annotating what the phase, run mode, and tool profiles were at that call.
Optionally checks whether the same call would be allowed today.

Examples:
  python3 Agent-Tools/session_replay.py latest
  python3 Agent-Tools/session_replay.py 4b54c65e --failures-only
  python3 Agent-Tools/session_replay.py latest --json
  python3 Agent-Tools/session_replay.py latest --policy-check
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
    extract_trace_id,
    is_background_state_changing_shell_dispatch,
    iter_records,
    resolve_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a SmallCTL run's tool-call sequence with policy simulation.")
    parser.add_argument("run", nargs="?", default="latest", help="Run dir, run id, 'latest', or 'latest-N'")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--failures-only", action="store_true", help="Only show failed dispatches")
    parser.add_argument("--policy-check", action="store_true", help="Annotate with phase/profile context per call")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _phase_at_trace(harness_records: list[dict[str, Any]], trace_id: str) -> str:
    for rec in reversed(harness_records):
        rec_tid = extract_trace_id(rec) or ""
        if rec_tid != trace_id:
            continue
        data = rec.get("data") or {}
        phase = data.get("current_phase") or data.get("active_phase") or ""
        if phase:
            return phase
    return "unknown"


def _mode_at_trace(harness_records: list[dict[str, Any]], trace_id: str) -> str:
    for rec in reversed(harness_records):
        rec_tid = extract_trace_id(rec) or ""
        if rec_tid != trace_id:
            continue
        data = rec.get("data") or {}
        mode = data.get("run_mode") or data.get("mode") or ""
        if mode:
            return mode
    return "unknown"


def _profiles_at_trace(harness_records: list[dict[str, Any]], trace_id: str) -> list[str]:
    for rec in reversed(harness_records):
        rec_tid = extract_trace_id(rec) or ""
        if rec_tid != trace_id:
            continue
        data = rec.get("data") or {}
        profiles = data.get("tool_profiles") or data.get("active_tool_profiles") or []
        if isinstance(profiles, list):
            return [str(p) for p in profiles]
    return []


def _parse_dispatches(
    tools_records: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair dispatch_start/dispatch_complete by trace_id, preserving order.

    A single model call can trigger more than one dispatch (for example an
    auto-triggered read after a primary tool call). Each pair gets its own
    ``sub_trace_id`` so the replay can distinguish them.
    """
    starts: dict[str, list[dict[str, Any]]] = {}
    dispatches: list[dict[str, Any]] = []
    subindex_by_trace: dict[str, int] = {}

    # Surface UI event kinds (tool_call, tool_result, system, etc.) when they
    # were logged at debug level.
    ui_event_kinds: dict[str, list[str]] = {}
    for rec in harness_records:
        if rec.get("event") != "ui_event":
            continue
        tid = extract_trace_id(rec) or ""
        if not tid:
            continue
        event_type = (rec.get("data") or {}).get("event_type") or ""
        if event_type:
            ui_event_kinds.setdefault(tid, []).append(str(event_type))

    for rec in tools_records:
        event = rec.get("event", "")
        tid = extract_trace_id(rec) or ""
        if event == "dispatch_start":
            starts.setdefault(tid, []).append(rec)
        elif event == "dispatch_complete":
            start = starts.get(tid, []).pop(0) if starts.get(tid) else None
            if start and not starts[tid]:
                starts.pop(tid, None)
            start_data = (start.get("data") or {}) if start else {}
            complete_data = rec.get("data") or {}
            subindex_by_trace[tid] = subindex_by_trace.get(tid, 0) + 1
            subindex = subindex_by_trace[tid]
            sub_trace_id = f"{tid}:dispatch-{subindex}"
            dispatch = {
                "trace_id": tid,
                "sub_trace_id": sub_trace_id,
                "timestamp": rec.get("timestamp"),
                "index": len(dispatches) + 1,
                "tool_name": complete_data.get("tool_name") or start_data.get("tool_name", "?"),
                "arguments": start_data.get("arguments", {}),
                "success": complete_data.get("success"),
                "error": complete_data.get("error"),
                "output": complete_data.get("output"),
                "phase": _phase_at_trace(harness_records, tid),
                "mode": _mode_at_trace(harness_records, tid),
                "profiles": _profiles_at_trace(harness_records, tid),
                "ui_event_kinds": ui_event_kinds.get(tid, []),
            }
            dispatch["background_state_changing"] = is_background_state_changing_shell_dispatch(dispatch)
            dispatches.append(dispatch)

    return dispatches


def _render_text(
    run_dir: Path,
    dispatches: list[dict[str, Any]],
    failures_only: bool,
    show_policy: bool,
) -> str:
    lines: list[str] = []
    lines.append(colorize(f"Session replay for {run_dir.name}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run directory: {run_dir}")
    lines.append("")

    total = len(dispatches)
    failed = [d for d in dispatches if d.get("success") is False]
    lines.append(f"Total dispatches: {total}  failed: {len(failed)}")

    tool_counter = Counter(d["tool_name"] for d in dispatches)
    if tool_counter:
        lines.append("")
        lines.append(colorize("Tool call distribution", Colors.BOLD + Colors.BLUE))
        for name, count in tool_counter.most_common():
            lines.append(f"  {count:>3}  {name}")

    failures_only = failures_only or False
    display = [d for d in dispatches if not failures_only or d.get("success") is False]

    lines.append("")
    lines.append(colorize(f"Dispatch sequence ({len(display)} shown)", Colors.BOLD + Colors.BLUE))

    for d in display:
        idx = d["index"]
        tool = d["tool_name"]
        success = d.get("success")
        status = colorize("OK", Colors.GREEN) if success else colorize("FAIL", Colors.RED) if success is False else "?"
        sub_tid = d.get("sub_trace_id") or d.get("trace_id", "")
        error = d.get("error", "")
        phase = d.get("phase", "?")
        mode = d.get("mode", "?")
        ui_kinds = d.get("ui_event_kinds", [])
        ui_kind_text = ",".join(ui_kinds[:3]) if ui_kinds else ""

        tail = ":".join(sub_tid.split(":")[-2:]) if ":" in sub_tid else sub_tid
        line = f"  #{idx:>3}  {status}  {tool:<20}  phase={phase:<10}  mode={mode:<10}  [{tail}]"

        lines.append(line)
        if ui_kind_text:
            lines.append(f"         ui_events: {ui_kind_text}")
        if d.get("background_state_changing"):
            lines.append(
                colorize(
                    "         warning: background state-changing command; OK only means the process launched, not that the mutation completed",
                    Colors.YELLOW,
                )
            )
        if success is False and error:
            lines.append(f"         error: {error[:200]}")
        if show_policy:
            args = d.get("arguments", {})
            if args:
                snippet = json.dumps(args, default=str, ensure_ascii=False)
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"         args: {snippet}")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    tools_records = list(iter_records(run_dir, "tools"))
    harness_records = list(iter_records(run_dir, "harness"))
    dispatches = _parse_dispatches(tools_records, harness_records)

    if not dispatches:
        print(colorize("No dispatch records found in this run.", Colors.YELLOW), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "run_dir": str(run_dir),
            "dispatches": dispatches,
        }, indent=2, default=str))
        return 0

    print(_render_text(run_dir, dispatches, args.failures_only, args.policy_check))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
