#!/usr/bin/env python3
"""trace_call — follow a single trace_id across all SmallCTL log channels.

A trace_id follows the pattern <session_id>:<task_id>:step-<n>:call-<m>.
You can also pass just the step/call suffix (e.g. step-3:call-1) if the run
contains only one session/task.

Examples:
  python Agent-Tools/trace_call.py 6cf0f870:task-0001:step-1:call-1
  python Agent-Tools/trace_call.py --run 6cf0f870 step-1:call-1
  python Agent-Tools/trace_call.py --last-error  # trace the most recent error-ish harness record
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_tools_lib import (
    Colors,
    colorize,
    error_records,
    extract_trace_id,
    find_records_by_trace_id,
    format_record_summary,
    has_error_indicator,
    iter_records,
    resolve_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace a single call across SmallCTL log channels.")
    parser.add_argument("trace", nargs="?", help="trace_id or step-call suffix")
    parser.add_argument("--run", default="latest", help="Run dir, run id, 'latest', or 'latest-N' (default: latest)")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--last-error", action="store_true", help="Trace the most recent error-ish harness record")
    parser.add_argument("--compact", "-c", action="store_true", help="Collapse model_token / chunk records")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    return parser.parse_args()


def _resolve_trace_id(run_dir: Path, trace_arg: str | None, last_error: bool) -> str:
    if last_error:
        records = list(iter_records(run_dir, "harness"))
        errs = error_records(records)
        if not errs:
            raise ValueError("No error-ish harness records found in this run")
        err = errs[-1]
        tid = extract_trace_id(err)
        if not tid:
            raise ValueError("Most recent error record has no trace_id")
        return tid

    if not trace_arg:
        raise ValueError("Provide a trace_id or use --last-error")

    trace_arg = trace_arg.strip()
    if ":" in trace_arg and not trace_arg.startswith("step-"):
        return trace_arg

    # Suffix only: prepend session_id:task_id from the first record that has them
    for rec in iter_records(run_dir, "harness"):
        tid = extract_trace_id(rec)
        if tid:
            base = ":".join(tid.split(":")[:2])
            return f"{base}:{trace_arg}"
    raise ValueError("Could not infer session/task prefix for trace suffix")


def _render_text(run_dir: Path, trace_id: str, grouped: dict[str, list[dict[str, Any]]], compact: bool = False) -> str:
    lines: list[str] = []
    lines.append(colorize(f"Trace: {trace_id}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run: {run_dir}")

    for channel in ("harness", "model_output", "chat", "tools"):
        recs = grouped.get(channel, [])
        if not recs:
            continue

        # In compact mode, collapse high-frequency token/chunk records
        display_recs = recs
        if compact and channel in {"harness", "model_output", "chat"}:
            collapsed = []
            token_run: list[dict[str, Any]] = []
            for rec in recs:
                if rec.get("event") in {"model_token", "chunk"}:
                    token_run.append(rec)
                else:
                    if token_run:
                        collapsed.append({
                            "timestamp": token_run[0].get("timestamp"),
                            "event": "model_token/chunk",
                            "message": f"... {len(token_run)} token/chunk records collapsed ...",
                            "data": {},
                        })
                        token_run = []
                    collapsed.append(rec)
            if token_run:
                collapsed.append({
                    "timestamp": token_run[0].get("timestamp"),
                    "event": "model_token/chunk",
                    "message": f"... {len(token_run)} token/chunk records collapsed ...",
                    "data": {},
                })
            display_recs = collapsed

        lines.append("")
        lines.append(colorize(f"[{channel}] {len(recs)} records", Colors.BOLD + Colors.BLUE))
        for rec in display_recs:
            prefix = "  "
            if has_error_indicator(rec):
                prefix = colorize("! ", Colors.RED)
            lines.append(prefix + format_record_summary(rec))

    # Reconstruct assistant output if available
    assistant_texts = []
    thinking_texts = []
    for rec in grouped.get("model_output", []):
        data = rec.get("data") or {}
        if "assistant_text" in data:
            assistant_texts.append(data["assistant_text"])
        if "thinking_text" in data:
            thinking_texts.append(data["thinking_text"])

    if assistant_texts or thinking_texts:
        lines.append("")
        lines.append(colorize("Reconstructed model output", Colors.BOLD + Colors.MAGENTA))
        if thinking_texts:
            lines.append(colorize("--- thinking ---", Colors.DIM))
            lines.append(thinking_texts[-1])
        if assistant_texts:
            lines.append(colorize("--- assistant ---", Colors.DIM))
            lines.append(assistant_texts[-1])

    # Tool call and result details
    tool_start = None
    tool_complete = None
    for rec in grouped.get("tools", []):
        if rec.get("event") == "dispatch_start":
            tool_start = rec
        elif rec.get("event") == "dispatch_complete":
            tool_complete = rec
    if tool_start or tool_complete:
        lines.append("")
        lines.append(colorize("Tool call details", Colors.BOLD + Colors.GREEN))
        if tool_start:
            data = tool_start.get("data", {})
            lines.append(f"  tool: {data.get('tool_name')}")
            lines.append(f"  arguments: {json.dumps(data.get('arguments'), indent=2)}")
        if tool_complete:
            data = tool_complete.get("data", {})
            lines.append(f"  success: {data.get('success')}")
            if data.get("error"):
                lines.append(colorize(f"  error: {data['error']}", Colors.RED))
            output = data.get("output")
            if output is not None:
                snippet = str(output)
                if len(snippet) > 800:
                    snippet = snippet[:800] + "\n... [truncated]"
                lines.append(f"  output snippet:\n{snippet}")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
        trace_id = _resolve_trace_id(run_dir, args.trace, args.last_error)
        grouped = find_records_by_trace_id(run_dir, trace_id)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"trace_id": trace_id, "channels": grouped}, indent=2, default=str))
        return 0

    print(_render_text(run_dir, trace_id, grouped, compact=args.compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
