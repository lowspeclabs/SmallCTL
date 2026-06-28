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

from collections import Counter

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
    warn_on_schema_mismatch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace a single call across SmallCTL log channels.")
    parser.add_argument("trace", nargs="?", help="trace_id or step-call suffix")
    parser.add_argument("--run", default="latest", help="Run dir, run id, 'latest', or 'latest-N' (default: latest)")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--last-error", action="store_true", help="Trace the most recent error-ish harness record")
    parser.add_argument("--compact", "-c", action="store_true", help="Collapse model_token / chunk / ui_event records")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    parser.add_argument("--step", type=int, help="Filter records to a specific step number")
    parser.add_argument("--task", help="Filter records to a specific task id")
    parser.add_argument("--event", action="append", default=None, help="Filter to specific event names (repeatable)")
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


def _run_matches_trace(run_dir: Path, trace_arg: str | None) -> bool:
    """Return False when the resolved run directory does not match a full trace id prefix."""
    if not trace_arg:
        return True
    if ":" not in trace_arg or trace_arg.startswith("step-"):
        return True
    prefix = trace_arg.split(":", 1)[0]
    return run_dir.name.startswith(prefix) if prefix else True


def _trace_step(trace_id: str | None) -> int | None:
    if not trace_id:
        return None
    for part in trace_id.split(":"):
        if part.startswith("step-"):
            try:
                return int(part[5:])
            except ValueError:
                return None
    return None


def _trace_task(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    parts = trace_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return None


def _filter_records(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    trace_id: str,
    step: int | None,
    task: str | None,
    events: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Apply step/task/event filters to grouped records."""
    result: dict[str, list[dict[str, Any]]] = {}
    for channel, recs in grouped.items():
        filtered: list[dict[str, Any]] = []
        for rec in recs:
            rec_trace = extract_trace_id(rec) or trace_id
            if step is not None and _trace_step(rec_trace) != step:
                continue
            if task is not None and _trace_task(rec_trace) != task:
                continue
            if events is not None and rec.get("event") not in events:
                continue
            filtered.append(rec)
        result[channel] = filtered
    return result


def _make_collapsed_record(run: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Build a single synthetic record for a collapsed run of records."""
    first_ts = run[0].get("timestamp", "") if run else ""
    last_ts = run[-1].get("timestamp", "") if run else ""
    ts_info = f" first={first_ts}"
    if first_ts != last_ts:
        ts_info += f" last={last_ts}"

    if kind == "token_chunk":
        return {
            "timestamp": first_ts,
            "event": "model_token/chunk",
            "message": f"... {len(run)} token/chunk records collapsed ...",
            "data": {},
            "_collapsed": True,
        }

    if kind == "ui_event":
        kinds = Counter(
            str((r.get("data") or {}).get("event_type") or "unknown") for r in run
        )
        kind_summary = ", ".join(f"{k}={v}" for k, v in kinds.most_common())
        return {
            "timestamp": first_ts,
            "event": "ui_event",
            "message": f"... {len(run)} ui_event records collapsed ({kind_summary}){ts_info}",
            "data": {},
            "_collapsed": True,
        }

    return run[0] if run else {}


def _render_text(run_dir: Path, trace_id: str, grouped: dict[str, list[dict[str, Any]]], compact: bool = False, mismatch_warning: str | None = None) -> str:
    lines: list[str] = []
    lines.append(colorize(f"Trace: {trace_id}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run: {run_dir}")
    if mismatch_warning:
        lines.append(colorize(f"Warning: {mismatch_warning}", Colors.YELLOW))

    for channel in ("harness", "model_output", "chat", "tools"):
        recs = grouped.get(channel, [])
        if not recs:
            continue

        # In compact mode, collapse high-frequency token/chunk/ui_event records
        display_recs = recs
        if compact and channel in {"harness", "model_output", "chat"}:
            collapsed = []
            run: list[dict[str, Any]] = []
            run_kind: str | None = None
            for rec in recs:
                kind: str | None = None
                event = rec.get("event")
                if event in {"model_token", "chunk"}:
                    kind = "token_chunk"
                elif event == "ui_event":
                    kind = "ui_event"

                if kind == run_kind:
                    run.append(rec)
                    continue

                if run:
                    collapsed.append(_make_collapsed_record(run, run_kind))
                    run = []

                run_kind = kind
                if kind is None:
                    collapsed.append(rec)
                    if event == "model_output_degenerate_loop_exhausted":
                        phrase = (rec.get("data") or {}).get("repeated_phrase")
                        if phrase:
                            collapsed.append({
                                "timestamp": rec.get("timestamp"),
                                "event": "degenerate_sample",
                                "message": f"repeated phrase ({len(phrase)} chars): {str(phrase)[:120]}{'...' if len(str(phrase)) > 120 else ''}",
                                "data": {},
                            })
                else:
                    run.append(rec)
            if run:
                collapsed.append(_make_collapsed_record(run, run_kind))
            display_recs = collapsed

        lines.append("")
        lines.append(colorize(f"[{channel}] {len(recs)} records", Colors.BOLD + Colors.BLUE))
        for rec in display_recs:
            prefix = "  "
            if has_error_indicator(rec):
                prefix = colorize("! ", Colors.RED)
            max_width = 1000 if rec.get("_collapsed") else 160
            lines.append(prefix + format_record_summary(rec, max_width=max_width))

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

    # Tool call and result details (support multiple dispatches per trace)
    tools_recs = grouped.get("tools", [])
    starts_by_tid: dict[str, list[dict[str, Any]]] = {}
    paired: list[tuple[dict[str, Any] | None, dict[str, Any]]] = []
    for rec in tools_recs:
        event = rec.get("event", "")
        tid = extract_trace_id(rec) or ""
        if event == "dispatch_start":
            starts_by_tid.setdefault(tid, []).append(rec)
        elif event == "dispatch_complete":
            start = starts_by_tid.get(tid, []).pop(0) if starts_by_tid.get(tid) else None
            if start and not starts_by_tid[tid]:
                starts_by_tid.pop(tid, None)
            paired.append((start, rec))

    # UI event kinds for this trace from harness debug records
    ui_event_kinds: list[str] = []
    for rec in grouped.get("harness", []):
        if rec.get("event") == "ui_event":
            event_type = (rec.get("data") or {}).get("event_type")
            if event_type:
                ui_event_kinds.append(str(event_type))

    if paired:
        lines.append("")
        lines.append(colorize("Tool call details", Colors.BOLD + Colors.GREEN))
        for idx, (tool_start, tool_complete) in enumerate(paired, start=1):
            if len(paired) > 1:
                lines.append(f"  --- dispatch {idx} ---")
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
                if tool_start and data.get("success") is False:
                    start_data = tool_start.get("data", {})
                    tool_name = str(start_data.get("tool_name") or "")
                    if tool_name in {"file_write", "ssh_file_write"}:
                        args = start_data.get("arguments") or {}
                        content = args.get("content") or args.get("text") or args.get("value") or ""
                        if content:
                            preview = str(content)
                            if len(preview) > 400:
                                preview = preview[:400] + "\n... [truncated]"
                            lines.append(colorize("  blocked content preview:", Colors.YELLOW))
                            lines.append(preview)
        if ui_event_kinds:
            lines.append(f"  ui_event kinds: {', '.join(ui_event_kinds[:10])}")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
        for warning in warn_on_schema_mismatch(run_dir):
            print(colorize(f"Warning: {warning}", Colors.YELLOW), file=sys.stderr)
        trace_id = _resolve_trace_id(run_dir, args.trace, args.last_error)
        grouped = find_records_by_trace_id(run_dir, trace_id)
        event_filter = set(args.event) if args.event else None
        grouped = _filter_records(
            grouped,
            trace_id=trace_id,
            step=args.step,
            task=args.task,
            events=event_filter,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    mismatch_warning = None
    if not _run_matches_trace(run_dir, args.trace):
        prefix = args.trace.split(":", 1)[0] if args.trace else ""
        mismatch_warning = (
            f"resolved run {run_dir.name} does not match trace prefix {prefix}; "
            "records may be from a different session"
        )
        print(colorize(f"Warning: {mismatch_warning}", Colors.YELLOW), file=sys.stderr)

    if args.json:
        payload = {"trace_id": trace_id, "channels": grouped}
        if mismatch_warning:
            payload["warning"] = mismatch_warning
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(_render_text(run_dir, trace_id, grouped, compact=args.compact, mismatch_warning=mismatch_warning))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
